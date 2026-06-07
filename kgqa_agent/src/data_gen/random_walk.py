"""Random-walk orchestration: load config, run walks, write JSON outputs."""
from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Set
import random
import pathlib

from kgqa_agent.src.tools.virtuoso_kg import VirtuosoKG
from kgqa_agent.src.data_gen.random_walk_impl import (
    generate_unique_paths,
    generate_unique_paths_batched,
)


def _project_root() -> str:
    """Repo root from KGQA_REPO_ROOT or file location."""
    env_root = os.getenv('KGQA_REPO_ROOT')
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    return str(pathlib.Path(__file__).resolve().parents[3])


def _resolve_repo_relative(maybe_path: Optional[str]) -> str:
    """Resolve path relative to repo root if not absolute."""
    if not maybe_path:
        raise FileNotFoundError("Empty path")
    return maybe_path if os.path.isabs(maybe_path) else os.path.abspath(os.path.join(_project_root(), maybe_path))


def load_predicate_list(maybe_path: Optional[str], cfg_path: Optional[str] = None) -> List[str]:
    """Load predicate whitelist JSON (dict -> freq-sorted keys, list -> values)."""
    if not maybe_path:
        return []
    with open(_resolve_repo_relative(maybe_path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [k for k, _ in sorted(data.items(), key=lambda kv: kv[1], reverse=True)]
    return [str(x).strip() for x in data if x] if isinstance(data, list) else []


def _load_sparql_config(sparql_cfg_path: str) -> Dict:
    """Load SPARQL config YAML."""
    import yaml
    with open(sparql_cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def run_random_walk_from_config(
    cfg: Dict,
    *,
    cfg_path: Optional[str] = None,
    save_every: int = 0,
    resume: bool = False,
    checkpoint_out_file: Optional[str] = None,
    checkpoint_out_names: Optional[str] = None,
) -> List[Dict]:
    """Generate random walks from config dict and return path dicts."""
    if seed := cfg.get('seed'):
        random.seed(int(seed))
    
    walks = cfg.get('walks', {}) or {}
    n = int(walks.get('num_starts', 10))
    steps_max = int(walks.get('max_hops', 3))
    steps_min = int(walks.get('min_hops', 2))
    max_attempts = int(walks.get('max_attempts', 4 * n))
    start_pool_size = int(walks.get('start_pool_size', 16))
    neigh_pool_size = int(walks.get('neigh_pool_size', 16))
    batch_size = int(walks.get('batch_size', 1))
    start_pred_sample_m = int(walks.get('pred_sample_m', 4))
    per_pred_pool_k = int(walks.get('per_pred_pool_k', 4))
    include_names = bool(walks.get('include_names', True))
    max_index = int(walks.get('max_index', 50))
    
    debug_cfg = cfg.get('debug', {}) or {}
    log_start_progress = bool(debug_cfg.get('start_progress') or os.getenv('RW_START_PROGRESS'))

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
    sp_count = len(start_predicates) if start_predicates else 0
    ap_preview = ", ".join((include_predicate_substrings or [])[:5])
    print(f"[RW config] endpoint={endpoint} graph_uri={graph_uri} seed={seed} steps=[{steps_min},{steps_max}] num_starts={n} max_attempts={max_attempts}")
    print(f"[RW whitelist] allow_file={ap_json or ''} (resolved={_resolve_repo_relative(ap_json) if ap_json else ''}) count={ap_count} preview=[{ap_preview}]")
    print(f"[RW params] start_pred_count={sp_count} pred_sample_m={start_pred_sample_m} per_pred_pool_k={per_pred_pool_k} start_pool_size={start_pool_size} neigh_pool_size={neigh_pool_size}")

    # Resume state
    existing_records: List[Dict] = []
    existing_name_paths: Set[str] = set()
    next_path_id = 1
    def _atomic_write(full_paths: List[Dict]):
        if not checkpoint_out_file or not checkpoint_out_names:
            return
        os.makedirs(os.path.dirname(checkpoint_out_file), exist_ok=True)
        tmp_full = f"{checkpoint_out_file}.tmp"
        with open(tmp_full, "w", encoding="utf-8") as wf:
            json.dump(full_paths, wf, ensure_ascii=False, indent=2)
            wf.flush(); os.fsync(wf.fileno())
        os.replace(tmp_full, checkpoint_out_file)
        # name-only
        name_only = [{
            "path_id": p.get("path_id"),
            "name_path": p.get("name_path"),
            "path_length": p.get("path_length"),
        } for p in full_paths]
        tmp_names = f"{checkpoint_out_names}.tmp"
        with open(tmp_names, "w", encoding="utf-8") as wf2:
            json.dump(name_only, wf2, ensure_ascii=False, indent=2)
            wf2.flush(); os.fsync(wf2.fileno())
        os.replace(tmp_names, checkpoint_out_names)

    # Load existing output if resume
    if resume and checkpoint_out_file and os.path.exists(checkpoint_out_file):
        try:
            with open(checkpoint_out_file, "r", encoding="utf-8") as rf:
                existing_records = json.load(rf) or []
            for rec in existing_records:
                np = rec.get("name_path")
                if np:
                    existing_name_paths.add(str(np))
            # compute next id
            ids = [int(p.get("path_id")) for p in existing_records if isinstance(p.get("path_id"), int)]
            next_path_id = (max(ids) + 1) if ids else (len(existing_records) + 1)
        except Exception:
            existing_records = []
            existing_name_paths = set()
            next_path_id = 1

    # Accumulator for newly generated (this run)
    new_records: List[Dict] = []

    def _on_accept(base_rec: Dict):
        nonlocal next_path_id
        # Skip if already saved (paranoia; should be filtered by initial_seen)
        if base_rec.get("name_path") in existing_name_paths:
            return
        # enrich with ids/length
        # Prefer hop_count if provided, else derive from name_path tokens
        hop_count = base_rec.get("hop_count")
        if isinstance(hop_count, int):
            path_len = max(0, hop_count)
        else:
            np = str(base_rec.get("name_path") or "")
            parts = [t.strip() for t in np.split("->") if t.strip()]
            path_len = max(0, (len(parts) - 1) // 2) if parts else 0
        # For output, use detailed query_steps if available
        out_steps = base_rec.get("query_steps") or base_rec.get("steps") or []
        rec = {**base_rec, "steps": out_steps, "path_id": next_path_id, "path_length": path_len}
        next_path_id += 1
        new_records.append(rec)
        # Periodic checkpoint
        if save_every and (len(new_records) % save_every == 0):
            _atomic_write(existing_records + new_records)

    paths = generate_unique_paths_batched(
        endpoint_url=endpoint,
        timeout_s=timeout_s,
        n=n,
        batch_size=batch_size,
        max_attempts=max_attempts,
        steps_min=steps_min,
        steps_max=steps_max,
        include_names=include_names,
        start_pool_size=start_pool_size,
        neigh_pool_size=neigh_pool_size,
        graph_uri=graph_uri,
        pred_sample_m=start_pred_sample_m,
        per_pred_pool_k=per_pred_pool_k,
        include_predicate_substrings=include_predicate_substrings,
        start_predicates=start_predicates,
        log_start_progress=False,
        on_accept=_on_accept if save_every and checkpoint_out_file else None,
        initial_seen=existing_name_paths if resume else None,
        max_index=max_index,
    )
    # Assign ids to any remaining (if not checkpointed during)
    for p in paths:
        if p.get("name_path") in existing_name_paths:
            # skip, already saved
            continue
        if not any(nr.get("name_path") == p.get("name_path") for nr in new_records):
            hop_count = p.get("hop_count")
            if isinstance(hop_count, int):
                p_len = max(0, hop_count)
            else:
                np = str(p.get("name_path") or "")
                parts = [t.strip() for t in np.split("->") if t.strip()]
                p_len = max(0, (len(parts) - 1) // 2) if parts else 0
            out_steps = p.get("query_steps") or p.get("steps") or []
            new_records.append({**p, "steps": out_steps, "path_id": next_path_id, "path_length": p_len})
            next_path_id += 1

    # Final write if checkpointing requested
    if save_every and checkpoint_out_file:
        _atomic_write(existing_records + new_records)

    # For return value, include all records (existing + new)
    return existing_records + new_records


def save_paths(paths: List[Dict], out_file: str, out_names: str):
    """Write two JSONs: full paths and compact name-only list."""
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)
    name_only = [{
        "path_id": p["path_id"],
        "name_path": p["name_path"],
        "path_length": p["path_length"],
    } for p in paths]
    with open(out_names, "w", encoding="utf-8") as f:
        json.dump(name_only, f, ensure_ascii=False, indent=2)

