from __future__ import annotations
"""Synthesize <information> tool-call traces for golden KG paths.

Each input path is a JSON object with Freebase URIs and names.
For every hop (entity -> relation -> next entity) we now issue one
`get_relations` query followed by a `get_triples` query, mirroring the
tool usage enforced during evaluation. Safety filters discard paths with
oversized relation sets, missing triple evidence, or mismatched golden
targets so only clean, well-aligned traces are written out.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
from kgqa_agent.src.tools.direct_sparql_client import DirectSPARQLKGClient
from kgqa_agent.src.tools.relation_normalizer import normalize_relation


def load_paths(paths_file: str) -> List[Dict[str, Any]]:
    """Load golden paths from JSON."""
    with open(paths_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def _is_conjunction_path(item: Dict[str, Any]) -> bool:
    """Check if a path item is a conjunction type (has name_path1 and name_path2)."""
    return 'name_path1' in item and 'name_path2' in item


def _extract_topic_entity_names(obj: Dict[str, Any]) -> List[str]:
    """Extract topic entity surface names from a QA record."""
    if not isinstance(obj, dict):
        return []
    te = obj.get("topic_entities")
    names: List[str] = []
    if isinstance(te, list):
        for x in te:
            if isinstance(x, str) and x.strip():
                names.append(x.strip())
    elif isinstance(te, dict):
        for k, v in te.items():
            # Prefer name-like string value
            if isinstance(v, str) and v.strip():
                names.append(v.strip())
            elif isinstance(k, str) and k.strip():
                names.append(k.strip())
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _make_augmented_question(question: str, topic_entity_names: List[str]) -> str:
    base = (question or "").strip()
    if not topic_entity_names:
        return base
    extra = " ".join(t for t in topic_entity_names if t)
    if not extra:
        return base
    return (base + " " + extra).strip()


def load_questions_map(qa_file: str) -> Dict[int, Dict[str, Any]]:
    """Build path_id -> {question, topic_entity_names, augmented_question}."""
    mapping: Dict[int, Dict[str, Any]] = {}
    
    # Try to detect file format: JSON array or JSONL
    # First, try reading as JSON (handles JSON arrays)
    try:
        with open(qa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Successfully loaded as JSON
        if isinstance(data, list):
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                pid = obj.get('path_id')
                if pid is None:
                    continue
                q = obj.get('raw_question') or obj.get('question') or ''
                topic_names = _extract_topic_entity_names(obj)
                pid_int = int(pid)
                augmented = _make_augmented_question(q, topic_names)
                mapping[pid_int] = {
                    "question": q,
                    "topic_entity_names": topic_names,
                    "augmented_question": augmented,
                }
        elif isinstance(data, dict):
            for k, v in data.items():
                pid_int = int(k)
                if isinstance(v, dict):
                    q = v.get('raw_question') or v.get('question') or ''
                    topic_names = _extract_topic_entity_names(v)
                else:
                    q = str(v)
                    topic_names = []
                augmented = _make_augmented_question(q, topic_names)
                mapping[pid_int] = {
                    "question": q,
                    "topic_entity_names": topic_names,
                    "augmented_question": augmented,
                }
        return mapping
    except (json.JSONDecodeError, ValueError):
        # If JSON parsing fails, try JSONL format (one JSON object per line)
        pass
    
    # Try JSONL format as fallback
    with open(qa_file, 'r', encoding='utf-8') as f:
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get('path_id')
                    if pid is None:
                        continue
                    q = obj.get('raw_question') or obj.get('question') or ''
                    topic_names = _extract_topic_entity_names(obj)
                    pid_int = int(pid)
                    augmented = _make_augmented_question(q, topic_names)
                    mapping[pid_int] = {
                        "question": q,
                        "topic_entity_names": topic_names,
                        "augmented_question": augmented,
                    }
                except json.JSONDecodeError:
                    # Skip invalid lines
                    continue
        return mapping
    
    return mapping


def uri_to_id(uri: str | None) -> str:
    """Convert a Freebase ns URI to dotted ID (e.g., m.xxx)."""
    if not uri:
        return ""
    return uri.replace("http://rdf.freebase.com/ns/", "").replace("/", ".")


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def ensure_required_in_topk(rel_ids: List[str], required: str, top_k: int) -> List[str]:
    """Ensure the truncated top_k relations contain the required relation.

    If required is already in the first top_k, return as-is.
    If not present and len(top_k_list)==top_k, replace the last one with required.
    If list shorter than top_k, append required.
    """
    rel_ids = unique_preserve_order(rel_ids)
    if top_k <= 0:
        return []
    if not required:
        return rel_ids[:top_k]
    required = required.strip()
    top = rel_ids[:top_k]
    if required in top:
        return top
    if len(top) == top_k:
        return top[:-1] + [required]
    else:
        top.append(required)
        return unique_preserve_order(top)[:top_k]


def _rule1_exceeds_max_num(count: int, max_num: int) -> bool:
    """Rule 1: return True if candidate count violates the max_num cap."""
    return count > max_num


def _normalize_lower(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _build_flatten_to_original_mapping(cvt_info: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build flatten_relation -> original_relation mapping from cvt_info.

    Args:
        cvt_info: CVT info list, each element contains 'flatten_relation' and 'original_relation'

    Returns:
        Mapping dict: {flatten_relation: original_relation}
    """
    mapping = {}
    for info in cvt_info:
        flatten_rel = info.get('flatten_relation')
        original_rel = info.get('original_relation')
        if flatten_rel and original_rel:
            mapping[flatten_rel] = original_rel
    return mapping


def _deduplicate_and_filter_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate and filter triples.

    Args:
        triples: List of triples

    Returns:
        Deduplicated and filtered triple list (removes duplicate triples and self-loops)
    """
    seen = set()
    result = []
    for triple in triples:
        head_id = triple.get('head_id', '')
        tail_id = triple.get('tail_id', '')
        relation = triple.get('relation', '')
        
        # Filter: head_id and tail_id must be different
        if head_id and tail_id and head_id == tail_id:
            continue
        
        # Deduplicate: use (head_id, relation, tail_id) as key
        key = (head_id, relation, tail_id)
        if key not in seen:
            seen.add(key)
            result.append(triple)
    
    return result


def _is_golden_triple(
    triple: Dict[str, Any],
    golden_relation: str,
    golden_name: str,
    golden_id: str,
    flatten_to_original: Optional[Dict[str, str]] = None,
) -> bool:
    rel = triple.get("relation", "") or ""
    # If flatten_to_original mapping is provided, map flatten_relation to original_relation
    if flatten_to_original:
        rel = flatten_to_original.get(rel, rel)
    rel_norm = normalize_relation(rel)
    golden_norm = normalize_relation(golden_relation or "")
    if golden_norm and rel_norm != golden_norm:
        return False

    target_id = (golden_id or "").strip()
    if target_id and target_id in {triple.get("head_id", ""), triple.get("tail_id", "")}:
        return True

    target_name = _normalize_lower(golden_name)
    if not target_name:
        return False
    head_name = _normalize_lower(triple.get("head", ""))
    tail_name = _normalize_lower(triple.get("tail", ""))
    return target_name in {head_name, tail_name}


def _format_triple_results(
    triples: List[Dict[str, Any]],
    golden_relation: str,
    golden_name: str,
    golden_id: str,
    max_results: int,
) -> List[str]:
    """Format triples into observation strings and shuffle them."""
    if max_results is None or max_results <= 0:
        max_results = len(triples)

    lines: List[str] = []
    seen: set[str] = set()

    for triple in triples:
        head = triple.get("head") or triple.get("head_id") or ""
        tail = triple.get("tail") or triple.get("tail_id") or ""
        relation = triple.get("relation") or ""
        line = f"[{head}, {relation}, {tail}]".strip()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)

    random.shuffle(lines)
    return lines[:max_results]


def _contains_golden_triple(
    triples: List[Dict[str, Any]],
    golden_relation: str,
    golden_name: str,
    golden_id: str,
    flatten_to_original: Optional[Dict[str, str]] = None,
) -> bool:
    for triple in triples:
        if _is_golden_triple(triple, golden_relation, golden_name, golden_id, flatten_to_original):
            return True
    return False


def synthesize_information(
    paths_file: str,
    qa_file: str,
    out_file: str,
    kg_server_url: str = "http://localhost:18890",
    relation_limit: int = 20,
    entity_limit: int = 50,
    *,
    save_every: int = 20,
    resume: bool = False,
    max_num: int = 50,
) -> None:
    paths = load_paths(paths_file)
    qmap = load_questions_map(qa_file) if qa_file else {}

    endpoint = kg_server_url if kg_server_url.endswith('/sparql') else f"{kg_server_url}/sparql"
    kg = DirectSPARQLKGClient(sparql_endpoint=endpoint, timeout=120)

    results: List[Dict[str, Any]] = []
    processed_ids: set = set()
    invalid_path_ids: List[int] = []
    relation_candidate_failures = 0
    triple_candidate_failures = 0
    golden_triple_mismatch = 0
    rule_a_failure = 0  # relations > 50
    rule_b_failure = 0  # golden fan-out > 5 (for normal paths) or > 10 (for conjunction paths)

    def _atomic_write(target: str, data: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(target) or '.', exist_ok=True)
        tmp = f"{target}.tmp"
        with open(tmp, 'w', encoding='utf-8') as wf:
            json.dump(data, wf, ensure_ascii=False, indent=2)
            wf.flush(); os.fsync(wf.fileno())
        os.replace(tmp, target)

    def _write_invalid_log():
        if not out_file:
            return
        base, ext = os.path.splitext(out_file)
        invalid_file = f"{base}.invalid.json"
        stats = {
            "relation_candidate_failures": relation_candidate_failures,
            "triple_candidate_failures": triple_candidate_failures,
            "golden_triple_mismatch": golden_triple_mismatch,
            "rule_a_failure": rule_a_failure,
            "rule_b_failure": rule_b_failure,
            "invalid_path_ids": sorted(set(invalid_path_ids)),
        }
        os.makedirs(os.path.dirname(invalid_file) or '.', exist_ok=True)
        tmp = f"{invalid_file}.tmp"
        with open(tmp, 'w', encoding='utf-8') as wf:
            json.dump(stats, wf, ensure_ascii=False, indent=2)
            wf.flush(); os.fsync(wf.fileno())
        os.replace(tmp, invalid_file)

    # If resume requested and out_file exists, load existing results and skip those path_ids
    if resume and os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as rf:
            existing = json.load(rf) or []
        if isinstance(existing, list):
            results.extend(existing)
            for it in existing:
                pid = int(it.get('path_id'))
                processed_ids.add(pid)
            print(f"[synthesize] Resuming: loaded {len(existing)} existing records; will skip their path_id(s)")


    # Build candidate list to improve progress accuracy: only items with path_id in QA
    # and not already processed when resume is enabled.
    candidates: List[Dict[str, Any]] = []
    for it in paths:
        pid = int(it.get('path_id') or 0)
        if pid in qmap and (not resume or pid not in processed_ids):
            candidates.append(it)

    print(f"[synthesize] Will process {len(candidates)}/{len(paths)} paths (filtered by QA ids and resume)")

    new_since_save = 0
    valid_count = 0
    for idx, item in enumerate(tqdm(candidates, total=len(candidates), desc="Synthesizing info")):
        path_id = int(item.get('path_id') or 0)
        # Only process items whose path_id exists in QA file mapping; otherwise skip
        if path_id not in qmap:
            # Optional: log once per skip (kept concise)
            # print(f"[synthesize] skip path_id={path_id}: not found in QA file")
            continue
        qrec = qmap.get(path_id, {})
        question = qrec.get('question', '')
        augmented_question = qrec.get('augmented_question', question)

        # Check if this is a conjunction type path
        if _is_conjunction_path(item):
            # Process conjunction type path
            name_path1 = item.get('name_path1', '')
            name_path2 = item.get('name_path2', '')
            path1_info = item.get('path1', {})
            path2_info = item.get('path2', {})
            center_entity_info = item.get('center_entity', {})
            
            if not path1_info or not path2_info or not center_entity_info:
                relation_candidate_failures += 1
                invalid_path_ids.append(path_id)
                continue
            
            # Extract entities and relations for both paths
            path1_data = {
                'name': path1_info.get('entity', {}).get('name', ''),
                'id': uri_to_id(path1_info.get('entity', {}).get('uri')),
                'relation_id': uri_to_id(path1_info.get('relation', {}).get('uri')) or path1_info.get('relation', {}).get('name', ''),
                'name_path': name_path1,
            }
            
            path2_data = {
                'name': path2_info.get('entity', {}).get('name', ''),
                'id': uri_to_id(path2_info.get('entity', {}).get('uri')),
                'relation_id': uri_to_id(path2_info.get('relation', {}).get('uri')) or path2_info.get('relation', {}).get('name', ''),
                'name_path': name_path2,
            }
            
            center_name = center_entity_info.get('name', '')
            center_id = uri_to_id(center_entity_info.get('uri'))
            
            # Randomly decide which path to process first
            paths_to_process = [path1_data, path2_data]
            random.shuffle(paths_to_process)
            first_path, second_path = paths_to_process
            
            info_steps: List[Dict[str, Any]] = []
            step_index = 0
            valid_path = True
            
            # Process first path (randomly selected)
            first_entity_token = first_path['id'] or first_path['name']
            if not first_entity_token:
                relation_candidate_failures += 1
                valid_path = False
            else:
                # Get relations for first entity
                relation_dicts = kg.get_relations(first_entity_token, question=augmented_question, top_k=max(max_num, relation_limit))
                relation_candidates = unique_preserve_order([r.get('relation', '') for r in relation_dicts if r.get('relation')])
                if first_path['relation_id'] and first_path['relation_id'] not in relation_candidates:
                    relation_candidates.append(first_path['relation_id'])
                
                if not relation_candidates:
                    relation_candidate_failures += 1
                    valid_path = False
                elif len(relation_candidates) > 50:
                    rule_a_failure += 1
                    valid_path = False
                else:
                    selected_relations = relation_candidates[:5]
                    if first_path['relation_id'] and first_path['relation_id'] not in selected_relations:
                        idx_to_replace = random.randint(0, len(selected_relations) - 1)
                        selected_relations[idx_to_replace] = first_path['relation_id']
                    
                    info_steps.append({
                        'step_index': step_index,
                        'query_type': 'get_relations',
                        'current': first_path['name'] or first_entity_token,
                        'args': {'entity': first_entity_token, 'top_k': relation_limit},
                        'results': selected_relations,
                    })
                    step_index += 1
                    
                    # Get triples for first path
                    # For conjunction paths, use limit_per_relation=11 to detect if fan-out > 10
                    result = kg.get_triples(first_entity_token, selected_relations, limit_per_relation=11, return_with_cvt_info=True)
                    if isinstance(result, dict):
                        triples_raw = result.get('triples', [])
                        cvt_info = result.get('cvt_info', [])
                    else:
                        triples_raw = result
                        cvt_info = []
                    if not triples_raw:
                        triple_candidate_failures += 1
                        valid_path = False
                    else:
                        # Build flatten_relation -> original_relation mapping (used for Rule B and grouping)
                        flatten_to_original = _build_flatten_to_original_mapping(cvt_info)
                        golden_triples_count = 0
                        for t in triples_raw:
                            rel = t.get('relation', '')
                            original_rel = flatten_to_original.get(rel, rel)
                            if normalize_relation(original_rel) == normalize_relation(first_path['relation_id']):
                                golden_triples_count += 1
                        # Rule B for conjunction: golden relation fan-out must be <= 10
                        # Note: Rule C (golden_triple_mismatch) is not applied to conjunction paths
                        # because the query logic returns mixed results from multiple relations,
                        # which may cause false negatives
                        if golden_triples_count > 10:
                            rule_b_failure += 1
                            valid_path = False
                        else:
                            
                            # Group by original_relation
                            final_triples = []
                            triples_by_original_rel = {}
                            for t in triples_raw:
                                rel = t.get('relation')
                                # If it's a flatten_relation, map back to original_relation; otherwise use relation itself
                                original_rel = flatten_to_original.get(rel, rel)
                                if original_rel not in triples_by_original_rel:
                                    triples_by_original_rel[original_rel] = []
                                triples_by_original_rel[original_rel].append(t)
                            
                            # Each original_relation keeps at most 5 triples
                            for original_rel, t_list in triples_by_original_rel.items():
                                if len(t_list) > 5:
                                    final_triples.extend(random.sample(t_list, 5))
                                else:
                                    final_triples.extend(t_list)
                            
                            # Deduplicate and filter triples (remove duplicates and triples where head_id == tail_id)
                            final_triples = _deduplicate_and_filter_triples(final_triples)
                            
                            # For conjunction paths: if golden triple is not in results, add it
                            # Note: final_triples may contain flatten_relations, so we need to pass the mapping
                            if not _contains_golden_triple(final_triples, first_path['relation_id'], center_name, center_id, flatten_to_original):
                                golden_triple = {
                                    'head': first_path['name'] or first_entity_token,
                                    'head_id': first_path['id'] or '',
                                    'relation': first_path['relation_id'] or '',
                                    'tail': center_name or '',
                                    'tail_id': center_id or '',
                                }
                                final_triples.append(golden_triple)
                            
                            formatted_triples = _format_triple_results(
                                final_triples,
                                golden_relation=first_path['relation_id'],
                                golden_name=center_name,
                                golden_id=center_id,
                                max_results=25,
                            )
                            
                            if not formatted_triples:
                                triple_candidate_failures += 1
                                valid_path = False
                            else:
                                info_steps.append({
                                    'step_index': step_index,
                                    'query_type': 'get_triples',
                                    'current': first_path['name'] or first_entity_token,
                                    'args': {
                                        'entity': first_entity_token,
                                        'relations': selected_relations,
                                        'limit_per_relation': 5,
                                    },
                                    'results': formatted_triples,
                                })
                                step_index += 1
            
            # Process second path (only if first path is valid)
            if valid_path:
                second_entity_token = second_path['id'] or second_path['name']
                if not second_entity_token:
                    relation_candidate_failures += 1
                    valid_path = False
                else:
                    # Get relations for second entity
                    relation_dicts = kg.get_relations(second_entity_token, question=augmented_question, top_k=max(max_num, relation_limit))
                    relation_candidates = unique_preserve_order([r.get('relation', '') for r in relation_dicts if r.get('relation')])
                    if second_path['relation_id'] and second_path['relation_id'] not in relation_candidates:
                        relation_candidates.append(second_path['relation_id'])
                    
                    if not relation_candidates:
                        relation_candidate_failures += 1
                        valid_path = False
                    elif len(relation_candidates) > 50:
                        rule_a_failure += 1
                        valid_path = False
                    else:
                        selected_relations = relation_candidates[:5]
                        if second_path['relation_id'] and second_path['relation_id'] not in selected_relations:
                            idx_to_replace = random.randint(0, len(selected_relations) - 1)
                            selected_relations[idx_to_replace] = second_path['relation_id']
                        
                        info_steps.append({
                            'step_index': step_index,
                            'query_type': 'get_relations',
                            'current': second_path['name'] or second_entity_token,
                            'args': {'entity': second_entity_token, 'top_k': relation_limit},
                            'results': selected_relations,
                        })
                        step_index += 1
                        
                        # Get triples for second path
                        # For conjunction paths, use limit_per_relation=11 to detect if fan-out > 10
                        result = kg.get_triples(second_entity_token, selected_relations, limit_per_relation=11, return_with_cvt_info=True)
                        if isinstance(result, dict):
                            triples_raw = result.get('triples', [])
                            cvt_info = result.get('cvt_info', [])
                        else:
                            triples_raw = result
                            cvt_info = []
                        if not triples_raw:
                            triple_candidate_failures += 1
                            valid_path = False
                        else:
                            # Build flatten_relation -> original_relation mapping (used for Rule B and grouping)
                            flatten_to_original = _build_flatten_to_original_mapping(cvt_info)
                            golden_triples_count = 0
                            for t in triples_raw:
                                rel = t.get('relation', '')
                                original_rel = flatten_to_original.get(rel, rel)
                                if normalize_relation(original_rel) == normalize_relation(second_path['relation_id']):
                                    golden_triples_count += 1
                            # Rule B for conjunction: golden relation fan-out must be <= 10
                            # Note: Rule C (golden_triple_mismatch) is not applied to conjunction paths
                            # because the query logic returns mixed results from multiple relations,
                            # which may cause false negatives
                            if golden_triples_count > 10:
                                rule_b_failure += 1
                                valid_path = False
                            else:
                                
                                # Group by original_relation
                                final_triples = []
                                triples_by_original_rel = {}
                                for t in triples_raw:
                                    rel = t.get('relation')
                                    # If it's a flatten_relation, map back to original_relation; otherwise use relation itself
                                    original_rel = flatten_to_original.get(rel, rel)
                                    if original_rel not in triples_by_original_rel:
                                        triples_by_original_rel[original_rel] = []
                                    triples_by_original_rel[original_rel].append(t)
                                
                                # Each original_relation keeps at most 5 triples
                                for original_rel, t_list in triples_by_original_rel.items():
                                    if len(t_list) > 5:
                                        final_triples.extend(random.sample(t_list, 5))
                                    else:
                                        final_triples.extend(t_list)
                                
                                # Deduplicate and filter triples (remove duplicates and triples where head_id == tail_id)
                                final_triples = _deduplicate_and_filter_triples(final_triples)
                                
                                # For conjunction paths: if golden triple is not in results, add it
                                # Note: final_triples may contain flatten_relations, so we need to pass the mapping
                                if not _contains_golden_triple(final_triples, second_path['relation_id'], center_name, center_id, flatten_to_original):
                                    golden_triple = {
                                        'head': second_path['name'] or second_entity_token,
                                        'head_id': second_path['id'] or '',
                                        'relation': second_path['relation_id'] or '',
                                        'tail': center_name or '',
                                        'tail_id': center_id or '',
                                    }
                                    final_triples.append(golden_triple)
                                
                                formatted_triples = _format_triple_results(
                                    final_triples,
                                    golden_relation=second_path['relation_id'],
                                    golden_name=center_name,
                                    golden_id=center_id,
                                    max_results=25,
                                )
                                
                                if not formatted_triples:
                                    triple_candidate_failures += 1
                                    valid_path = False
                                else:
                                    info_steps.append({
                                        'step_index': step_index,
                                        'query_type': 'get_triples',
                                        'current': second_path['name'] or second_entity_token,
                                        'args': {
                                            'entity': second_entity_token,
                                            'relations': selected_relations,
                                            'limit_per_relation': 5,
                                        },
                                        'results': formatted_triples,
                                    })
                                    step_index += 1
            
            if not valid_path:
                invalid_path_ids.append(path_id)
                continue
            
            results.append({
                'name_path1': name_path1,
                'name_path2': name_path2,
                'path_id': path_id,
                'question': question,
                'steps': info_steps,
            })
            processed_ids.add(path_id)
            new_since_save += 1
            valid_count += 1
            
            # Periodic checkpoint
            if save_every and new_since_save >= int(save_every):
                _atomic_write(out_file, results)
                _write_invalid_log()
                print(f"[synthesize] checkpoint wrote {len(results)} records -> {out_file}")
                new_since_save = 0
            continue
        
        # Original processing for non-conjunction paths
        name_path = item.get('name_path') or ''
        raw_steps: List[Dict[str, Any]] = item.get('steps', [])
        if not raw_steps:
            relation_candidate_failures += 1
            invalid_path_ids.append(path_id)
            continue

        # Pre-extract entity names/ids and relation ids per hop
        entities: List[Tuple[str, str]] = []  # (entity_name, entity_id)
        rel_ids: List[str] = []  # relation dotted id per hop (len = len(entities)-1)
        for s in raw_steps:
            e_name = s.get('node_name', '')
            e_id = uri_to_id(s.get('node_uri'))
            entities.append((e_name, e_id))
        for s in raw_steps[:-1]:
            r_id = uri_to_id(s.get('relation_uri')) if s.get('relation_uri') else (s.get('relation_name') or '')
            rel_ids.append(r_id)

        info_steps: List[Dict[str, Any]] = []
        step_index = 0

        valid_path = True
        for hop_i in range(len(rel_ids)):
            e_name, e_id = entities[hop_i]
            r_id = rel_ids[hop_i]
            next_name, next_id = entities[hop_i + 1]

            entity_token = e_id or e_name or next_name
            if not entity_token:
                relation_candidate_failures += 1
                valid_path = False
                break

            relation_dicts = kg.get_relations(entity_token, question=augmented_question, top_k=max(max_num, relation_limit))
            relation_candidates = unique_preserve_order([r.get('relation', '') for r in relation_dicts if r.get('relation')])
            if r_id and r_id not in relation_candidates:
                relation_candidates.append(r_id)

            if not relation_candidates:
                relation_candidate_failures += 1
                valid_path = False
                break

            # Rule A: relations > 50
            if len(relation_candidates) > 50:
                rule_a_failure += 1
                valid_path = False
                break

            # Select top 5 relations, ensuring golden relation is included
            selected_relations = relation_candidates[:5]
            if r_id and r_id not in selected_relations:
                # Replace a random one with golden relation
                idx_to_replace = random.randint(0, len(selected_relations) - 1)
                selected_relations[idx_to_replace] = r_id

            info_steps.append({
                'step_index': step_index,
                'query_type': 'get_relations',
                'current': e_name or entity_token,
                'args': {'entity': entity_token, 'top_k': relation_limit},
                'results': selected_relations,
            })
            step_index += 1

            # Query triples with a slightly higher limit to detect fan-out > 5
            # We use 6 to check if it exceeds 5.
            result = kg.get_triples(entity_token, selected_relations, limit_per_relation=6, return_with_cvt_info=True)
            if isinstance(result, dict):
                triples_raw = result.get('triples', [])
                cvt_info = result.get('cvt_info', [])
            else:
                triples_raw = result
                cvt_info = []
            
            if not triples_raw:
                triple_candidate_failures += 1
                valid_path = False
                break

            # Rule B: Golden relation fan-out > 5
            # Build flatten_relation -> original_relation mapping for Rule B check
            flatten_to_original = _build_flatten_to_original_mapping(cvt_info)
            golden_triples_count = 0
            for t in triples_raw:
                rel = t.get('relation', '')
                original_rel = flatten_to_original.get(rel, rel)
                if normalize_relation(original_rel) == normalize_relation(r_id):
                    golden_triples_count += 1
            
            if golden_triples_count > 5:
                rule_b_failure += 1
                valid_path = False
                break

            # Build flatten_relation -> original_relation mapping (used for Rule B and Rule C)
            flatten_to_original = _build_flatten_to_original_mapping(cvt_info)
            
            if not _contains_golden_triple(triples_raw, r_id, next_name, next_id, flatten_to_original):
                golden_triple_mismatch += 1
                valid_path = False
                break
            
            # Group by original_relation
            final_triples = []
            triples_by_original_rel = {}
            for t in triples_raw:
                rel = t.get('relation')
                # If it's a flatten_relation, map back to original_relation; otherwise use relation itself
                original_rel = flatten_to_original.get(rel, rel)
                if original_rel not in triples_by_original_rel:
                    triples_by_original_rel[original_rel] = []
                triples_by_original_rel[original_rel].append(t)
            
            # Each original_relation keeps at most 5 triples
            for original_rel, t_list in triples_by_original_rel.items():
                if len(t_list) > 5:
                    final_triples.extend(random.sample(t_list, 5))
                else:
                    final_triples.extend(t_list)
            
            # Deduplicate and filter triples (remove duplicates and triples where head_id == tail_id)
            final_triples = _deduplicate_and_filter_triples(final_triples)
            
            # Ensure golden triple is in final_triples (similar to conjunction paths)
            if not _contains_golden_triple(final_triples, r_id, next_name, next_id, flatten_to_original):
                golden_triple = {
                    'head': e_name or entity_token,
                    'head_id': e_id or '',
                    'relation': r_id or '',
                    'tail': next_name or '',
                    'tail_id': next_id or '',
                }
                final_triples.append(golden_triple)

            formatted_triples = _format_triple_results(
                final_triples,
                golden_relation=r_id,
                golden_name=next_name,
                golden_id=next_id,
                max_results=25, # Max 5 relations * 5 triples
            )

            if not formatted_triples:
                triple_candidate_failures += 1
                valid_path = False
                break

            info_steps.append({
                'step_index': step_index,
                'query_type': 'get_triples',
                'current': e_name or entity_token,
                'args': {
                    'entity': entity_token,
                    'relations': selected_relations,
                    'limit_per_relation': 5,
                },
                'results': formatted_triples,
            })
            step_index += 1

        if not valid_path:
            # Invalid paths are skipped entirely and never written out
            invalid_path_ids.append(path_id)
            continue

        results.append({
            'name_path': name_path,
            'path_id': path_id,
            'question': question,
            'steps': info_steps,
        })
        processed_ids.add(path_id)
        new_since_save += 1
        valid_count += 1

        # Periodic checkpoint: write out every `save_every` new entries
        if save_every and new_since_save >= int(save_every):
            _atomic_write(out_file, results)
            _write_invalid_log()
            print(f"[synthesize] checkpoint wrote {len(results)} records -> {out_file}")
            new_since_save = 0

    # Final write (atomic)
    _atomic_write(out_file, results)
    _write_invalid_log()
    print(f"[synthesize] final write {len(results)} valid records -> {out_file}")
    print(f"[synthesize] processed_paths={len(candidates)}, valid_paths={valid_count}")
    print(
        "[synthesize] relation_candidate_failures=", relation_candidate_failures,
        "triple_candidate_failures=", triple_candidate_failures,
        "golden_triple_mismatch=", golden_triple_mismatch,
    )


def main():
    parser = argparse.ArgumentParser(description="Synthesize <information> sequences for KG paths")
    parser.add_argument('--paths-file', required=True, help='JSON file with golden paths (entity->relation->entity)')
    parser.add_argument('--qa-file', required=True, help='QA JSONL/JSON mapping path_id to question')
    parser.add_argument('--out-file', required=True, help='Output JSON file to write information sequences')
    parser.add_argument('--kg-server-url', default='http://localhost:18890', help='Virtuoso base URL or SPARQL endpoint URL')
    parser.add_argument('--relation-limit', type=int, default=20, help='Max relations to keep per step after Rule1; golden relation is always preserved')
    parser.add_argument('--entity-limit', type=int, default=200, help='Max entities to fetch per relation query (keep all)')
    parser.add_argument('--save-every', type=int, default=20, help='Checkpoint: write output every N new records')
    parser.add_argument('--resume', action='store_true', help='If set, load existing out-file and skip already processed path_id entries')
    parser.add_argument('--max-num', type=int, default=50, help='Max allowed number of results per step and require unique last entity')
    args = parser.parse_args()

    synthesize_information(
        paths_file=args.paths_file,
        qa_file=args.qa_file,
        out_file=args.out_file,
        kg_server_url=args.kg_server_url,
        relation_limit=args.relation_limit,
        entity_limit=args.entity_limit,
        save_every=args.save_every,
        resume=args.resume,
        max_num=args.max_num,
    )


if __name__ == '__main__':
    main()
