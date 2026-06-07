"""Random-walk utilities for knowledge graph exploration.

Features:
- Start-node sampling with predicate filters and optional degree constraints
- Neighbor sampling with predicate whitelist and basic readability checks
- Sequential/batched walk generation and path name formatting
"""
from __future__ import annotations
import random
import time
from typing import Optional, List, Dict, Set, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from kgqa_agent.src.tools.virtuoso_kg import VirtuosoKG


def _short_label(uri: str) -> str:
    """Last segment of a URI after '#' or '/'."""
    tail = uri.rsplit('#', 1)[-1]
    return tail.rsplit('/', 1)[-1]


def _relation_label(uri: str) -> str:
    """Predicate label as dotted Freebase ID (strip only HTTP/ns prefix)."""
    if not uri:
        return "N/A"
    prefix = "http://rdf.freebase.com/ns/"
    if uri.startswith(prefix):
        # Ensure '/' variants are normalized to dots (defensive)
        tail = uri[len(prefix):]
        return tail.replace('/', '.')
    if uri.startswith('ns:'):
        return uri[3:]
    # Fallback: keep last segment as-is (no dot/middle-dot/underscore conversion)
    return _short_label(uri)


def _is_readable_name(name: Optional[str], uri: Optional[str]) -> bool:
    """Heuristic: prefer real names over bare Freebase IDs."""
    if not name or name == "N/A":
        return False
    short = _short_label(uri or "")
    if name == short and short.startswith(("m.", "g.", "base.", "type.", "common.")):
        return False
    return True


def get_any_start_node_fast(kg: VirtuosoKG, max_offset: int = 8, graph_uri: Optional[str] = None):
    """Quickly fetch any IRI subject from the graph."""
    for off in [random.randint(0, max_offset) for _ in range(2)] + [0]:
        where = _wrap_graph_where(graph_uri, "?s ?p ?o . FILTER(isIRI(?s))")
        res = kg.execute_query(f"SELECT ?s WHERE {{ {where} }} LIMIT 1 OFFSET {off}")
        if res and res[0].get("s", {}).get("value"):
            return res[0]["s"]["value"]
    return None


def _wrap_graph_where(graph_uri: Optional[str], inner_where: str) -> str:
    """Wrap WHERE in GRAPH <uri> if provided."""
    if graph_uri:
        return f"GRAPH <{graph_uri}> {{ {inner_where} }}"
    return inner_where


def get_start_node_light_random(
    kg: VirtuosoKG,
    pool_size: int = 16,
    max_offset: int = 1024,
    graph_uri: Optional[str] = None,
    pred_sample_m: int = 4,
    per_pred_pool_k: int = 4,
    start_predicates: Optional[List[str]] = None,
    log_progress: bool = False,
):
    """Sample a start node from predicate-filtered pools (optionally degree-filtered)."""
    predicate_pool = start_predicates or []
    if not predicate_pool:
        return get_any_start_node_fast(kg, max_offset=0, graph_uri=graph_uri)

    preds = random.sample(predicate_pool, min(len(predicate_pool), pred_sample_m))
    candidates = []
    unique_subjects: Set[str] = set()
    
    for pred in preds:
        inner = f"?s <{pred}> ?o . FILTER(isIRI(?s))\nOPTIONAL {{ ?s ns:type.object.name ?nm . FILTER(LANGMATCHES(LANG(?nm),'en')) }}"
        where = _wrap_graph_where(graph_uri, inner)
        q = f"PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?s ?nm\nWHERE {{ {where} }}\nLIMIT {per_pred_pool_k}"
        res = kg.execute_query(q)
        
        if res:
            for b in res:
                if v := b.get("s", {}).get("value"):
                    nm = b.get("nm", {}).get("value") if b.get("nm") else None
                    candidates.append((v, pred, nm))
                    unique_subjects.add(v)
                    if pool_size and len(unique_subjects) >= pool_size:
                        break
        if pool_size and len(unique_subjects) >= pool_size:
            break

    seen: Dict[str, Dict] = {}
    for v, pred, nm in candidates:
        if v not in seen:
            seen[v] = {"pred": pred, "nm": nm}
        elif not seen[v].get("nm") and nm:
            seen[v]["nm"] = nm
    
    cand_list = [(uri, meta["pred"], meta.get("nm")) for uri, meta in seen.items()]
    if cand_list:
        
        named_pairs = [(uri, pred, nm) for uri, pred, nm in cand_list if _is_readable_name(nm, uri)]
        if named_pairs:
            cand_list = named_pairs
        
        weighted = []
        for uri, pred, nm in cand_list:
            weighted.extend([uri] * (2 if _is_readable_name(nm, uri) else 1))
        return random.choice(weighted)

    if graph_uri:
        q2 = f"SELECT ?s WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o . FILTER(ISIRI(?s)) }} }} LIMIT 1"
        if res2 := kg.execute_query(q2):
            if res2[0].get("s", {}).get("value"):
                return res2[0]["s"]["value"]
    return get_any_start_node_fast(kg, max_offset=0, graph_uri=graph_uri)


def sample_start_pool(
    kg: VirtuosoKG,
    pool_size: int,
    graph_uri: Optional[str],
    pred_sample_m: int,
    per_pred_pool_k: int,
    start_predicates: Optional[List[str]] = None,
    log_progress: bool = False,
) -> List[Tuple[str, Optional[str]]]:
    """Build a pool of up to pool_size (uri, name) start candidates."""
    predicate_pool = start_predicates or []
    if not predicate_pool:
        uri = get_any_start_node_fast(kg, max_offset=0, graph_uri=graph_uri)
        return [(uri, kg.get_entity_name(uri, graph_uri=graph_uri))] if uri else []

    preds = random.sample(predicate_pool, min(len(predicate_pool), pred_sample_m))
    candidates: List[Tuple[str, Optional[str]]] = []
    unique_subjects: Set[str] = set()
    
    for pred in preds:
        inner = f"?s <{pred}> ?o . FILTER(isIRI(?s))\nOPTIONAL {{ ?s ns:type.object.name ?nm . FILTER(LANGMATCHES(LANG(?nm),'en')) }}"
        where = _wrap_graph_where(graph_uri, inner)
        q = f"PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?s ?nm\nWHERE {{ {where} }}\nLIMIT {per_pred_pool_k}"
        res = kg.execute_query(q)
        
        if res:
            for b in res:
                if (v := b.get("s", {}).get("value")) and v not in unique_subjects:
                    nm = b.get("nm", {}).get("value") if b.get("nm") else None
                    unique_subjects.add(v)
                    candidates.append((v, nm))
                    if pool_size and len(unique_subjects) >= pool_size:
                        break
        if pool_size and len(unique_subjects) >= pool_size:
            break

    named = [(u, nm) for u, nm in candidates if _is_readable_name(nm, u)]
    return named or candidates


def get_neighbor_light_random(
    kg: VirtuosoKG,
    current_node,
    pool_size: int = 16,
    max_offset: int = 64, 
    attempts: int = 2,
    graph_uri: Optional[str] = None,
    include_names: bool = False,
    include_predicate_substrings: Optional[List[str]] = None,
    exclude_node_uris: Optional[Set[str]] = None,
):
    """Sample a neighbor from outgoing edges with optional filters."""
    for _ in range(attempts):
        incl = include_predicate_substrings or []
        incl_filter = ""
        values_clause = ""
        
        if incl:
            include_are_uris = all(s.strip().startswith(("http://", "https://", "urn:")) for s in incl)
            if include_are_uris:
                sample = random.sample(incl, 80) if len(incl) > 80 else incl
                values_clause = f"VALUES ?p {{ {' '.join(f'<{u}>' for u in sample)} }}"
            else:
                ors = " || ".join([f"CONTAINS(STR(?p), '{s.replace(chr(39), chr(92)+chr(39)).replace(chr(34), chr(92)+chr(34))}')" for s in incl])
                incl_filter = f" && ({ors})"
        
        name_opt = "OPTIONAL { ?o ns:type.object.name ?nm . FILTER(LANGMATCHES(LANG(?nm),'en')) }" if include_names else ""
        inner_core = f"<{current_node}> ?p ?o ."
        inner_core += f" {values_clause}" if values_clause else ""
        inner = f"{inner_core} FILTER(isIRI(?o){incl_filter}) {name_opt}"
        where = _wrap_graph_where(graph_uri, inner)
        q = f"PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?o ?p ?nm\nWHERE {{ {where} }}\nLIMIT {pool_size}"
        res = kg.execute_query(q)
        
        if res:
            parsed = [(b.get("o", {}).get("value"), b.get("p", {}).get("value"), 
                      b.get("nm", {}).get("value") if b.get("nm") else None) 
                     for b in res if b.get("o", {}).get("value") and b.get("p", {}).get("value")]
            
            if exclude_node_uris:
                parsed = [tpl for tpl in parsed if tpl[0] not in exclude_node_uris]
            
            # degree filtering removed
            
            named_pool = [t for t in parsed if t[2]]
            choice = random.choice(named_pool) if named_pool else (random.choice(parsed) if parsed else None)
            
            if choice:
                o, p, nm = choice
                return {"neighbor_uri": o, "relation_uri": p, "neighbor_name": nm}
    
    for fallback_q in [
        f"SELECT ?o ?p WHERE {{ GRAPH <{graph_uri}> {{ <{current_node}> ?p ?o . FILTER(isIRI(?o)) }} }} LIMIT 1" if graph_uri else None,
        f"SELECT ?o ?p WHERE {{ <{current_node}> ?p ?o . FILTER(isIRI(?o)) }} LIMIT 1"
    ]:
        if fallback_q:
            if res := kg.execute_query(fallback_q):
                if (o := res[0].get("o", {}).get("value")) and (p := res[0].get("p", {}).get("value")):
                    if not exclude_node_uris or o not in exclude_node_uris:
                        return {"neighbor_uri": o, "relation_uri": p}
    return None


def perform_random_walk(
    kg: VirtuosoKG,
    steps_min=1,
    steps_max=2,
    include_names=False,
    start_pool_size=16,
    start_max_offset=1024,
    neigh_pool_size=16,
    neigh_max_offset=64,
    graph_uri: Optional[str] = None,
    verbose: bool = True,
    pred_sample_m: int = 4,
    per_pred_pool_k: int = 4,
    include_predicate_substrings: Optional[List[str]] = None,
    start_predicates: Optional[List[str]] = None,
    log_start_progress: bool = False,
    start_node_uri: Optional[str] = None,
    start_node_name: Optional[str] = None,
    *,
    max_index: int = 50,
):
    """Execute a single random walk from start node, rejecting unreadable names.
    
    Args:
        steps_min/max: Walk length range
        start_node_uri/name: Optional pre-selected start (else sampled)
        verbose: Print step-by-step progress
    
    Returns:
        List of step dicts with node_uri, node_name, relation_uri, relation_name,
        or empty list if walk fails validation
    """
    if not start_node_uri:
        start_node_uri = get_start_node_light_random(
            kg, pool_size=start_pool_size, max_offset=start_max_offset, graph_uri=graph_uri,
            pred_sample_m=pred_sample_m, per_pred_pool_k=per_pred_pool_k, start_predicates=start_predicates,
            log_progress=log_start_progress
        )
    if not start_node_uri:
        print("Failed to find a start node.")
        return []

    start_name = start_node_name or kg.get_entity_name(start_node_uri, graph_uri=graph_uri)
    if not _is_readable_name(start_name, start_node_uri):
        if verbose:
            print("Start node lacks a readable name; abandoning this path.")
        return []

    # Original node/edge path structure (kept for name_path construction and compatibility)
    path = [{"node_uri": start_node_uri, "node_name": start_name, "relation_uri": None, "relation_name": None}]
    # Detailed query steps to persist (relations/entities per hop)
    query_steps: List[Dict] = []
    current_node_uri = start_node_uri
    current_node_name = start_name
    visited: Set[str] = {start_node_uri}
    k_steps = random.randint(steps_min, steps_max)
    
    if verbose:
        print(f"Executing {k_steps} random-walk steps (min-cost mode). Start: {start_node_uri}")

    for i in range(k_steps):
        step_start_time = time.time()
        # 1) Relations query (tail): collect DISTINCT relations of current node (respect filters)
        incl = include_predicate_substrings or []
        incl_filter = ""
        values_clause = ""
        if incl:
            include_are_uris = all(s.strip().startswith(("http://", "https://", "urn:")) for s in incl)
            if include_are_uris:
                sample = random.sample(incl, 80) if len(incl) > 80 else incl
                values_clause = f"VALUES ?p {{ {' '.join(f'<{u}>' for u in sample)} }}"
            else:
                ors = " || ".join([f"CONTAINS(STR(?p), '{s.replace(chr(39), chr(92)+chr(39)).replace(chr(34), chr(92)+chr(34))}')" for s in incl])
                incl_filter = f" && ({ors})"

        inner_rel = f"<{current_node_uri}> ?p ?o . FILTER(isIRI(?o){incl_filter})"
        if values_clause:
            inner_rel = f"{inner_rel} {values_clause}"
        where_rel = _wrap_graph_where(graph_uri, inner_rel)
        q_rel = f"SELECT DISTINCT ?p WHERE {{ {where_rel} }} LIMIT {max_index + 1}"
        rel_rows = kg.execute_query(q_rel) or []
        rel_uris: List[str] = [r.get("p", {}).get("value") for r in rel_rows if r.get("p", {}).get("value")]
        # If more than max_index relations, invalidate path
        if len(rel_uris) > max_index:
            if verbose:
                print(f"Step {i+1}: relations count {len(rel_uris)} > max_index={max_index}; invalid path.")
            return []  # invalid
        # Build mapping relation_uri -> relation_name (dotted)
        rel_map: Dict[str, str] = {u: _relation_label(u) for u in rel_uris}
        if not rel_map:
            if verbose:
                print("No relations found; walk ends.")
            break
        # Pick one relation (prefer ones with readable label; here labels are dotted IDs always available)
        relation_uri = random.choice(list(rel_map.keys()))
        relation_name = rel_map.get(relation_uri) or _relation_label(relation_uri)
        # Record relation query step (tail)
        query_steps.append({
            "step_index": str(len(query_steps) + 1),
            "node_uri": current_node_uri,
            "node_name": current_node_name,
            "relation_uri": relation_uri,
            "relation_name": relation_name,
            "type": "tail",
            "query_results": {ru: rn for ru, rn in rel_map.items()},
        })

        # 2) Entities query (tail): for chosen relation, collect neighbor entities
        name_opt = "OPTIONAL { ?o ns:type.object.name ?nm . FILTER(LANGMATCHES(LANG(?nm),'en')) }" if include_names else ""
        inner_ent = f"<{current_node_uri}> <{relation_uri}> ?o . FILTER(isIRI(?o)) {name_opt}"
        where_ent = _wrap_graph_where(graph_uri, inner_ent)
        q_ent = ("PREFIX ns: <http://rdf.freebase.com/ns/>\n" f"SELECT ?o ?nm WHERE {{ {where_ent} }} LIMIT {max_index + 1}")
        ent_rows = kg.execute_query(q_ent) or []
        ent_list: List[Tuple[str, Optional[str]]] = []
        for b in ent_rows:
            o = b.get("o", {}).get("value")
            if not o:
                continue
            nm = b.get("nm", {}).get("value") if b.get("nm") else None
            ent_list.append((o, nm))
        if len(ent_list) > max_index:
            if verbose:
                print(f"Step {i+1}: entity results {len(ent_list)} > max_index={max_index}; invalid path.")
            return []
        # Map entity_uri -> entity_name (prefer readable; fallback to short label)
        ent_map: Dict[str, str] = {}
        for uri, nm in ent_list:
            name = nm or kg.get_entity_name(uri, graph_uri=graph_uri) if include_names else (nm or None)
            if not name:
                name = _short_label(uri)
            ent_map[uri] = name
        if not ent_map:
            if verbose:
                print("No entity results for chosen relation; walk ends.")
            break
        # Select neighbor (prefer readable name)
        readable = [u for u in ent_map.keys() if _is_readable_name(ent_map[u], u)]
        neighbor_uri = random.choice(readable) if readable else random.choice(list(ent_map.keys()))
        neighbor_name = ent_map.get(neighbor_uri)
        # Record entity query step (tail)
        query_steps.append({
            "step_index": str(len(query_steps) + 1),
            "node_uri": current_node_uri,
            "node_name": current_node_name,
            "relation_uri": relation_uri,
            "relation_name": relation_name,
            "type": "tail",
            "query_results": {eu: ent_map[eu] for eu in ent_map.keys()},
        })

        # If this is the final intended hop, enforce unique entity
        if i == k_steps - 1 and len(ent_map) != 1:
            if verbose:
                print(f"Final hop has {len(ent_map)} entity candidates (require 1); invalid path.")
            return []

        # Commit hop to the original path nodes
        path[-1].update({"relation_uri": relation_uri, "relation_name": relation_name})
        path.append({"node_uri": neighbor_uri, "node_name": neighbor_name, "relation_uri": None, "relation_name": None})
        if neighbor_uri:
            visited.add(neighbor_uri)
            current_node_uri = neighbor_uri
            current_node_name = neighbor_name
        else:
            break

        if verbose:
            print(f"Step {i+1}: {relation_uri} -> {neighbor_uri} ({time.time() - step_start_time:.2f}s)")

    path_length = len(path) - 1
    if path_length < steps_min:
        if verbose:
            print(f"Path length < {steps_min}; discarding this path.")
        return []

    # Attach query_steps to the path object by returning a dict wrapper for compatibility up-stack
    return {"path_nodes": path, "query_steps": query_steps, "hop_count": (len(path) - 1)}


def build_name_path(path: List[Dict]) -> str:
    """Format as: entity -> relation -> entity -> ..."""
    if not path:
        return ""
    segments = [path[0].get("node_name") or _short_label(path[0].get("node_uri", ""))]
    for i in range(len(path) - 1):
        rel_name = path[i].get("relation_name") or _short_label(path[i].get("relation_uri", ""))
        nxt = path[i + 1]
        nname = nxt.get("node_name") or _short_label(nxt.get("node_uri", ""))
        segments.extend([rel_name, nname])
    return " -> ".join(segments)


def generate_unique_paths(
    kg: VirtuosoKG,
    n: int = 10,
    max_attempts: int = 200,
    require_unique_starts: bool = True,
    *,
    on_accept: Optional[Callable[[Dict], None]] = None,
    initial_seen: Optional[Set[str]] = None,
    **walk_kwargs,
) -> List[Dict]:
    """Generate n unique random walks sequentially.
    
    Args:
        kg: Knowledge graph connection
        n: Target number of unique paths
        max_attempts: Abort after this many tries
        require_unique_starts: Enforce distinct start nodes across paths
        **walk_kwargs: Forwarded to perform_random_walk
    
    Returns:
        List of path dicts with 'name_path' and 'steps' keys
    """
    from tqdm import tqdm

    seen = set(initial_seen or set())
    starts = set()
    results = []
    attempts = 0
    
    with tqdm(total=n, desc="Generating paths", unit="paths") as pbar:
        while len(results) < n and attempts < max_attempts:
            attempts += 1
            p = perform_random_walk(kg, verbose=False, **walk_kwargs)
            if not p:
                continue
            # Support new return shape: dict with path_nodes/query_steps/hop_count
            if isinstance(p, dict):
                path_nodes = p.get("path_nodes") or []
                query_steps = p.get("query_steps") or []
                hop_count = int(p.get("hop_count") or max(0, len(path_nodes) - 1))
            else:
                path_nodes = p
                query_steps = []
                hop_count = max(0, len(path_nodes) - 1)

            if not path_nodes:
                continue

            start_uri = path_nodes[0].get("node_uri") if path_nodes else None
            if require_unique_starts and start_uri:
                if start_uri in starts:
                    continue
                starts.add(start_uri)

            name_path = build_name_path(path_nodes)
            if not name_path or name_path in seen:
                continue
            
            print(f"[RW accepted] path_len={hop_count} start={start_uri} pred_sample_m={walk_kwargs.get('pred_sample_m')} per_pred_pool_k={walk_kwargs.get('per_pred_pool_k')} start_pool_size={walk_kwargs.get('start_pool_size')} neigh_pool_size={walk_kwargs.get('neigh_pool_size')}")
            seen.add(name_path)
            rec = {"name_path": name_path, "steps": path_nodes, "query_steps": query_steps, "hop_count": hop_count}
            results.append(rec)
            if on_accept:
                try:
                    on_accept(rec)
                except Exception:
                    # Do not fail path generation if callback errors
                    pass
            pbar.update(1)
    
    return results


def generate_unique_paths_batched(
    endpoint_url: str,
    timeout_s: int,
    n: int,
    batch_size: int,
    max_attempts: int,
    *,
    steps_min: int,
    steps_max: int,
    include_names: bool,
    start_pool_size: int,
    neigh_pool_size: int,
    graph_uri: Optional[str],
    pred_sample_m: int,
    per_pred_pool_k: int,
    include_predicate_substrings: Optional[List[str]],
    start_predicates: Optional[List[str]],
    log_start_progress: bool = False,
    on_accept: Optional[Callable[[Dict], None]] = None,
    initial_seen: Optional[Set[str]] = None,
    max_index: int = 50,
) -> List[Dict]:
    """Generate n unique walks in parallel batches with pooled start nodes.
    
    Each batch:
    1. Sample start pool from predicate-filtered subjects
    2. Distribute starts across workers with per-thread KG connections
    3. Deduplicate results by name_path
    
    Args:
        endpoint_url: SPARQL endpoint
        batch_size: Parallel workers per batch
        n: Target unique paths
        max_attempts: Total attempts budget
    
    Returns:
        List of path dicts with 'name_path' and 'steps' keys
    """
    from tqdm import tqdm

    results: List[Dict] = []
    seen: Set[str] = set(initial_seen or set())
    attempts = 0
    ctrl_kg = VirtuosoKG(endpoint_url=endpoint_url, timeout_s=timeout_s)

    with tqdm(total=n, desc="Generating paths", unit="paths") as pbar:
        while len(results) < n and attempts < max_attempts:
            start_pool = sample_start_pool(
                ctrl_kg, pool_size=start_pool_size, graph_uri=graph_uri,
                pred_sample_m=pred_sample_m, per_pred_pool_k=per_pred_pool_k, start_predicates=start_predicates,
                log_progress=False
            )

            if not start_pool:
                attempts += batch_size
                continue

            starts_for_batch = random.sample(start_pool, batch_size) if len(start_pool) >= batch_size else [random.choice(start_pool) for _ in range(batch_size)]

            with ThreadPoolExecutor(max_workers=batch_size) as ex:
                def _worker(start_uri, start_nm):
                    kg_local = VirtuosoKG(endpoint_url=endpoint_url, timeout_s=timeout_s)
                    return perform_random_walk(
                        kg_local, steps_min=steps_min, steps_max=steps_max, include_names=include_names,
                        start_pool_size=start_pool_size, neigh_pool_size=neigh_pool_size, graph_uri=graph_uri,
                        pred_sample_m=pred_sample_m, per_pred_pool_k=per_pred_pool_k,
                        include_predicate_substrings=include_predicate_substrings, start_predicates=start_predicates,
                        log_start_progress=False, verbose=False, start_node_uri=start_uri, start_node_name=start_nm,
                        max_index=max_index,
                    )
                
                futures = [ex.submit(_worker, uri, nm) for uri, nm in starts_for_batch]
                
                for fut in as_completed(futures):
                    attempts += 1
                    p = fut.result()
                    if not p:
                        continue
                    # Support new return shape (dict) and legacy list fallback
                    if isinstance(p, dict):
                        path_nodes = p.get("path_nodes") or []
                        query_steps = p.get("query_steps") or []
                        hop_count = int(p.get("hop_count") or max(0, len(path_nodes) - 1))
                    else:
                        path_nodes = p or []
                        query_steps = []
                        hop_count = max(0, len(path_nodes) - 1)

                    if not path_nodes:
                        continue

                    name_path = build_name_path(path_nodes)
                    if not name_path or name_path in seen:
                        continue

                    start_uri = path_nodes[0].get("node_uri") if path_nodes else None
                    print(f"[RW accepted] path_len={hop_count} start={start_uri} pred_sample_m={pred_sample_m} per_pred_pool_k={per_pred_pool_k} start_pool_size={start_pool_size} neigh_pool_size={neigh_pool_size}")
                    seen.add(name_path)
                    rec = {"name_path": name_path, "steps": path_nodes, "query_steps": query_steps, "hop_count": hop_count}
                    results.append(rec)
                    if on_accept:
                        try:
                            on_accept(rec)
                        except Exception:
                            pass
                    pbar.update(1)
                    if len(results) >= n:
                        break

    return results
