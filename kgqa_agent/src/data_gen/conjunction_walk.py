"""Conjunction-type path generation orchestration and checkpointing."""
from __future__ import annotations
import json
import os
import pathlib
import random
from typing import Optional, Dict, List, Set

from kgqa_agent.src.data_gen.conjunction_walk_impl import generate_unique_conjunction_paths
from kgqa_agent.src.tools.virtuoso_kg import VirtuosoKG


def _project_root() -> str:
    """Get project root directory."""
    return str(pathlib.Path(__file__).resolve().parents[3])


def _resolve_repo_relative(maybe_path: Optional[str]) -> str:
    """Resolve a path relative to repo root if not absolute."""
    if not maybe_path:
        return ""
    if os.path.isabs(maybe_path):
        return maybe_path
    return os.path.join(_project_root(), maybe_path)


def _load_sparql_config(sparql_cfg_path: str) -> Dict:
    """Load SPARQL config from YAML file."""
    import yaml
    with open(sparql_cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_predicate_list(json_path: str, cfg_path: Optional[str] = None) -> List[str]:
    """Load predicate list from JSON file."""
    resolved = _resolve_repo_relative(json_path) if json_path else None
    if not resolved or not os.path.exists(resolved):
        if cfg_path:
            # Try relative to config file
            cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
            alt = os.path.join(cfg_dir, json_path)
            if os.path.exists(alt):
                resolved = alt
    if not resolved or not os.path.exists(resolved):
        return []
    with open(resolved, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get('predicates', []) or data.get('relations', []) or []
    return []


def run_conjunction_walk_from_config(
    cfg: Dict,
    *,
    cfg_path: Optional[str] = None,
    save_every: int = 0,
    resume: bool = False,
    checkpoint_out_file: Optional[str] = None,
    checkpoint_out_names: Optional[str] = None,
) -> List[Dict]:
    """Generate conjunction-type paths from config dict and return path dicts."""
    if seed := cfg.get('seed'):
        random.seed(int(seed))
    
    walks = cfg.get('walks', {}) or {}
    n = int(walks.get('num_starts', 10))
    max_attempts = int(walks.get('max_attempts', 4 * n))
    require_unique_centers = bool(walks.get('require_unique_centers', True))
    max_answer_candidates = int(walks.get('max_answer_candidates', 3))
    start_pool_size = int(walks.get('start_pool_size', 16))
    pred_sample_m = int(walks.get('pred_sample_m', 4))
    per_pred_pool_k = int(walks.get('per_pred_pool_k', 4))
    batch_size = int(walks.get('batch_size', 1))
    
    debug_cfg = cfg.get('debug', {}) or {}
    
    sparql_cfg_ref = (cfg.get('sparql', {}) or {}).get('config')
    sparql_cfg = _load_sparql_config(_resolve_repo_relative(sparql_cfg_ref)) if sparql_cfg_ref else (cfg.get('sparql', {}) or {})
    endpoint = sparql_cfg.get('endpoint') or sparql_cfg.get('url') or 'http://localhost:18890/sparql'
    timeout_s = int(sparql_cfg.get('timeout_s', sparql_cfg.get('timeout', 15)))
    graph_uri = sparql_cfg.get('graph_uri') or cfg.get('graph_uri')

    ap_json = cfg.get('allow-predicates-json')
    include_predicate_substrings = None
    if ap_json:
        all_ap = load_predicate_list(ap_json, cfg_path)
        ap_top_k = int(cfg.get('allow-predicates-top-k', 0))
        include_predicate_substrings = all_ap[:ap_top_k] if ap_top_k > 0 else all_ap

    sp_json = cfg.get('start-predicates-json')
    start_predicates = None
    if sp_json:
        all_sp = load_predicate_list(sp_json, cfg_path)
        sp_top_k = int(cfg.get('start-predicates-top-k', 0))
        start_predicates = all_sp[:sp_top_k] if sp_top_k > 0 else all_sp

    kg = VirtuosoKG(endpoint_url=endpoint, timeout_s=timeout_s)

    ap_count = len(include_predicate_substrings) if include_predicate_substrings else 0
    ap_preview = ", ".join((include_predicate_substrings or [])[:5])
    sp_count = len(start_predicates) if start_predicates else 0
    sp_preview = ", ".join((start_predicates or [])[:5])
    print(f"[Conjunction config] endpoint={endpoint} graph_uri={graph_uri} seed={cfg.get('seed')} num_paths={n} max_attempts={max_attempts}")
    print(f"[Conjunction config] max_answer_candidates={max_answer_candidates} require_unique_centers={require_unique_centers}")
    print(f"[Conjunction config] start_pool_size={start_pool_size} pred_sample_m={pred_sample_m} per_pred_pool_k={per_pred_pool_k} batch_size={batch_size}")
    print(f"[Conjunction whitelist] allow_file={ap_json or ''} count={ap_count} preview=[{ap_preview}]")
    print(f"[Conjunction start-predicates] start_file={sp_json or ''} count={sp_count} preview=[{sp_preview}]")

    # Resume state
    existing_records: List[Dict] = []
    existing_name_paths: Set[str] = set()
    existing_path_ids: Set[int] = set()
    next_path_id = 1
    
    def _atomic_write(full_paths: List[Dict]):
        if not checkpoint_out_file or not checkpoint_out_names:
            return
        os.makedirs(os.path.dirname(checkpoint_out_file), exist_ok=True)
        tmp_full = f"{checkpoint_out_file}.tmp"
        with open(tmp_full, "w", encoding="utf-8") as wf:
            json.dump(full_paths, wf, ensure_ascii=False, indent=2)
            wf.flush()
            os.fsync(wf.fileno())
        os.replace(tmp_full, checkpoint_out_file)
        # name-only
        def _format_entity_dict(entities: List[Dict]) -> Dict[str, str]:
            """Convert entity list to simplified dict format: {uri_without_prefix: name}"""
            result = {}
            for ent in entities:
                uri = ent.get("uri", "")
                name = ent.get("name", "")
                if uri and name:
                    # Remove http://rdf.freebase.com/ns/ prefix
                    if uri.startswith("http://rdf.freebase.com/ns/"):
                        uri_short = uri[len("http://rdf.freebase.com/ns/"):]
                    elif uri.startswith("ns:"):
                        uri_short = uri[3:]
                    else:
                        uri_short = uri
                    result[uri_short] = name
            return result
        
        name_only = []
        for p in full_paths:
            topic_entities_list = p.get("topic_entities", [])
            answers_list = p.get("answers", [])
            name_only.append({
                "path_id": p.get("path_id"),
                "name_path1": p.get("name_path1"),
                "name_path2": p.get("name_path2"),
                "path_length": p.get("path_length"),
                "topic_entities": _format_entity_dict(topic_entities_list),
                "answers": _format_entity_dict(answers_list),
            })
        tmp_names = f"{checkpoint_out_names}.tmp"
        with open(tmp_names, "w", encoding="utf-8") as wf2:
            json.dump(name_only, wf2, ensure_ascii=False, indent=2)
            wf2.flush()
            os.fsync(wf2.fileno())
        os.replace(tmp_names, checkpoint_out_names)

    # Load existing output if resume
    if resume and checkpoint_out_file and os.path.exists(checkpoint_out_file):
        try:
            with open(checkpoint_out_file, "r", encoding="utf-8") as rf:
                existing_records = json.load(rf) or []
            
            # Collect existing name_paths and path_ids
            for rec in existing_records:
                # Check both name_path1 and name_path2 for deduplication
                np1 = rec.get("name_path1")
                np2 = rec.get("name_path2")
                if np1 and np2:
                    # Create a unique identifier for deduplication
                    path_key = f"{np1}|||{np2}"
                    existing_name_paths.add(path_key)
                # Also check legacy name_path format for backward compatibility
                np = rec.get("name_path")
                if np:
                    # Parse legacy format: "path1 AND path2"
                    if " AND " in np:
                        parts = np.split(" AND ", 1)
                        np1_legacy = parts[0].strip()
                        np2_legacy = parts[1].strip() if len(parts) > 1 else ""
                        if np1_legacy and np2_legacy:
                            path_key = f"{np1_legacy}|||{np2_legacy}"
                            existing_name_paths.add(path_key)
                    else:
                        existing_name_paths.add(str(np))
                pid = rec.get("path_id")
                if pid is not None:
                    try:
                        existing_path_ids.add(int(pid))
                    except (ValueError, TypeError):
                        pass
            
            # Compute next path_id: max(existing_ids) + 1
            if existing_path_ids:
                next_path_id = max(existing_path_ids) + 1
            else:
                next_path_id = len(existing_records) + 1
                
        except Exception as e:
            print(f"[Conjunction resume] Error loading existing records: {e}")
            existing_records = []
            existing_name_paths = set()
            existing_path_ids = set()
            next_path_id = 1

    print(f"[Conjunction resume] existing={len(existing_records)} paths, existing_path_ids={len(existing_path_ids)}, next_id={next_path_id}")

    # Callback to save incrementally
    new_paths: List[Dict] = []
    def _on_accept(rec: Dict):
        nonlocal next_path_id, existing_path_ids
        
        # Assign unique path_id
        rec["path_id"] = next_path_id
        existing_path_ids.add(next_path_id)
        next_path_id += 1
        
        # path_length is sum of both path lengths (each 1-hop, so total is 2)
        rec["path_length"] = rec.get("hop_count", 2)
        new_paths.append(rec)
        all_paths = existing_records + new_paths
        if save_every > 0 and len(new_paths) % save_every == 0:
            _atomic_write(all_paths)
            print(f"[Conjunction checkpoint] saved {len(all_paths)} paths")

    # Generate paths (use batched version if batch_size > 1)
    from kgqa_agent.src.data_gen.conjunction_walk_impl import (
        generate_unique_conjunction_paths,
        generate_unique_conjunction_paths_batched,
    )
    
    if batch_size > 1:
        results = generate_unique_conjunction_paths_batched(
            endpoint_url=endpoint,
            timeout_s=timeout_s,
            n=n,
            batch_size=batch_size,
            max_attempts=max_attempts,
            require_unique_centers=require_unique_centers,
            max_answer_candidates=max_answer_candidates,
            start_pool_size=start_pool_size,
            pred_sample_m=pred_sample_m,
            per_pred_pool_k=per_pred_pool_k,
            start_predicates=start_predicates,
            on_accept=_on_accept,
            initial_seen=existing_name_paths,
            graph_uri=graph_uri,
            include_predicate_substrings=include_predicate_substrings,
        )
    else:
        results = generate_unique_conjunction_paths(
            kg,
            n=n,
            max_attempts=max_attempts,
            require_unique_centers=require_unique_centers,
            max_answer_candidates=max_answer_candidates,
            start_pool_size=start_pool_size,
            pred_sample_m=pred_sample_m,
            per_pred_pool_k=per_pred_pool_k,
            start_predicates=start_predicates,
            on_accept=_on_accept,
            initial_seen=existing_name_paths,
            graph_uri=graph_uri,
            include_predicate_substrings=include_predicate_substrings,
        )

    # Add path_id and path_length to results (for any remaining paths not handled by callback)
    for i, rec in enumerate(results):
        if "path_id" not in rec:
            # Assign unique path_id
            rec["path_id"] = next_path_id + i
            existing_path_ids.add(rec["path_id"])
            # path_length is sum of both path lengths (each 1-hop, so total is 2)
            rec["path_length"] = rec.get("hop_count", 2)

    # Final save
    all_paths = existing_records + new_paths
    if checkpoint_out_file:
        _atomic_write(all_paths)
        print(f"[Conjunction final] saved {len(all_paths)} paths -> {checkpoint_out_file}")

    return all_paths

