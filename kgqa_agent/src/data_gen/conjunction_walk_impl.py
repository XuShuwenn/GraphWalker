"""Conjunction-type path generation utilities.

Generates paths where two different entities connect to the same target entity
via different relations:
  entity1 -> relationA -> entity2
  entity3 -> relationB -> entity2
"""
from __future__ import annotations
import random
from typing import Optional, List, Dict, Set, Tuple
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
        tail = uri[len(prefix):]
        return tail.replace('/', '.')
    if uri.startswith('ns:'):
        return uri[3:]
    return _short_label(uri)


def _is_readable_name(name: Optional[str], uri: Optional[str]) -> bool:
    """Heuristic: prefer real names over bare Freebase IDs."""
    if not name or name == "N/A":
        return False
    short = _short_label(uri or "")
    if name == short and short.startswith(("m.", "g.", "base.", "type.", "common.")):
        return False
    return True


def _wrap_graph_where(graph_uri: Optional[str], inner_where: str) -> str:
    """Wrap WHERE in GRAPH <uri> if provided."""
    if graph_uri:
        return f"GRAPH <{graph_uri}> {{ {inner_where} }}"
    return inner_where


def get_center_entity_with_incoming_relations(
    kg: VirtuosoKG,
    graph_uri: Optional[str] = None,
    min_incoming_relations: int = 2,
    max_attempts: int = 50,
    include_predicate_substrings: Optional[List[str]] = None,
) -> Optional[Tuple[str, str, List[str]]]:
    """Find a center entity that has at least min_incoming_relations incoming relations.
    
    Returns:
        Tuple of (center_entity_uri, center_entity_name, list_of_incoming_relation_uris)
        or None if not found
    """
    for _ in range(max_attempts):
        # Sample a random entity as center
        where = _wrap_graph_where(graph_uri, "?s ?p ?o . FILTER(isIRI(?s))")
        q = f"SELECT ?s WHERE {{ {where} }} LIMIT 1 OFFSET {random.randint(0, 1000)}"
        res = kg.execute_query(q)
        
        if not res or not res[0].get("s", {}).get("value"):
            continue
        
        center_uri = res[0]["s"]["value"]
        
        # Get center entity name
        name_where = _wrap_graph_where(
            graph_uri,
            f"<{center_uri}> ns:type.object.name ?name . FILTER(LANGMATCHES(LANG(?name), 'en'))"
        )
        name_q = f"PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?name WHERE {{ {name_where} }} LIMIT 1"
        name_res = kg.execute_query(name_q)
        center_name = None
        if name_res and name_res[0].get("name", {}).get("value"):
            center_name = name_res[0]["name"]["value"]
        
        # Check if center has readable name
        if not _is_readable_name(center_name, center_uri):
            continue
        
        # Query incoming relations
        incl_filter = ""
        values_clause = ""
        if include_predicate_substrings:
            incl = include_predicate_substrings
            include_are_uris = all(s.strip().startswith(("http://", "https://", "urn:")) for s in incl)
            if include_are_uris:
                sample = random.sample(incl, 80) if len(incl) > 80 else incl
                values_clause = f"VALUES ?p {{ {' '.join(f'<{u}>' for u in sample)} }}"
            else:
                ors = " || ".join([
                    f"CONTAINS(STR(?p), '{s.replace(chr(39), chr(92)+chr(39)).replace(chr(34), chr(92)+chr(34))}')"
                    for s in incl
                ])
                incl_filter = f" && ({ors})"
        
        inner_rel = f"?head ?p <{center_uri}> . FILTER(isIRI(?head) && isIRI(?p){incl_filter})"
        if values_clause:
            inner_rel = f"{inner_rel} {values_clause}"
        where_rel = _wrap_graph_where(graph_uri, inner_rel)
        q_rel = f"SELECT DISTINCT ?p WHERE {{ {where_rel} }} LIMIT 100"
        rel_rows = kg.execute_query(q_rel) or []
        rel_uris: List[str] = [r.get("p", {}).get("value") for r in rel_rows if r.get("p", {}).get("value")]
        
        if len(rel_uris) >= min_incoming_relations:
            return (center_uri, center_name or center_uri, rel_uris)
    
    return None


def find_entities_via_relation(
    kg: VirtuosoKG,
    relation_uri: str,
    target_entity_uri: str,
    graph_uri: Optional[str] = None,
    limit: int = 10,
    require_readable_name: bool = True,
) -> List[Tuple[str, str]]:
    """Find entities that connect to target_entity via relation_uri.
    
    Returns:
        List of (entity_uri, entity_name) tuples
    """
    inner = f"?head <{relation_uri}> <{target_entity_uri}> . FILTER(isIRI(?head))"
    if require_readable_name:
        inner += "\nOPTIONAL { ?head ns:type.object.name ?name . FILTER(LANGMATCHES(LANG(?name), 'en')) }"
    where = _wrap_graph_where(graph_uri, inner)
    q = f"PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?head ?name WHERE {{ {where} }} LIMIT {limit}"
    res = kg.execute_query(q) or []
    
    entities = []
    for row in res:
        head_uri = row.get("head", {}).get("value")
        head_name = row.get("name", {}).get("value") if row.get("name") else None
        if not head_uri:
            continue
        if require_readable_name:
            if not _is_readable_name(head_name, head_uri):
                continue
        entities.append((head_uri, head_name or head_uri))
    
    return entities


def generate_conjunction_path(
    kg: VirtuosoKG,
    graph_uri: Optional[str] = None,
    include_predicate_substrings: Optional[List[str]] = None,
    max_attempts: int = 100,
    max_answer_candidates: int = 3,
    start_pool_size: int = 16,
    pred_sample_m: int = 4,
    per_pred_pool_k: int = 4,
    start_predicates: Optional[List[str]] = None,
    verbose: bool = False,
) -> Optional[Dict]:
    """Generate a single conjunction-type path.
    
    Args:
        max_answer_candidates: Maximum number of answer candidates (center entities) allowed.
            Only paths with <= max_answer_candidates answers are considered valid.
    
    Returns:
        Dict with keys:
        - path_nodes: List of node dicts with node_uri, node_name, relation_uri, relation_name
        - name_path: String representation of the path
        - topic_entities: List of topic entities [entity1, entity3]
        - answers: List of answer entities [entity2] (center entities)
        - center_entity: The center entity that both paths connect to (for backward compatibility)
        - path1: entity1 -> relationA -> center
        - path2: entity3 -> relationB -> center
        or None if generation fails
    """
    # Step 1: Sample start entity (entity1) using predicate-based sampling if available
    # Build a pool of start candidates
    start_pool: List[Tuple[str, Optional[str]]] = []
    
    if start_predicates:
        # Use predicate-based sampling (similar to random_walk)
        preds = random.sample(start_predicates, min(len(start_predicates), pred_sample_m))
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
                        start_pool.append((v, nm))
                        if start_pool_size and len(unique_subjects) >= start_pool_size:
                            break
            if start_pool_size and len(unique_subjects) >= start_pool_size:
                break
        
        # Filter for readable names
        readable_pool = [(u, nm) for u, nm in start_pool if _is_readable_name(nm, u)]
        if readable_pool:
            start_pool = readable_pool
        # If no readable names, keep all candidates (better than nothing)
    else:
        # Fallback to random sampling
        from kgqa_agent.src.data_gen.random_walk_impl import get_any_start_node_fast
        for _ in range(min(start_pool_size, 10)):
            entity_uri = get_any_start_node_fast(kg, max_offset=1000, graph_uri=graph_uri)
            if entity_uri:
                name_where = _wrap_graph_where(
                    graph_uri,
                    f"<{entity_uri}> ns:type.object.name ?name . FILTER(LANGMATCHES(LANG(?name), 'en'))"
                )
                name_q = f"PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?name WHERE {{ {name_where} }} LIMIT 1"
                name_res = kg.execute_query(name_q)
                entity_name = None
                if name_res and name_res[0].get("name", {}).get("value"):
                    entity_name = name_res[0]["name"]["value"]
                if _is_readable_name(entity_name, entity_uri):
                    start_pool.append((entity_uri, entity_name))
    
    if not start_pool:
        if verbose:
            print("Failed to build start entity pool.")
        return None
    
    # Step 2: Try entities from the start pool
    attempts = 0
    max_rel_attempts = max_attempts
    
    while attempts < max_rel_attempts:
        attempts += 1
        
        # Sample an entity from the start pool
        entity1_uri, entity1_name = random.choice(start_pool)
        
        # Get outgoing relations from entity1
        incl_filter = ""
        values_clause = ""
        if include_predicate_substrings:
            incl = include_predicate_substrings
            include_are_uris = all(s.strip().startswith(("http://", "https://", "urn:")) for s in incl)
            if include_are_uris:
                sample = random.sample(incl, 80) if len(incl) > 80 else incl
                values_clause = f"VALUES ?p {{ {' '.join(f'<{u}>' for u in sample)} }}"
            else:
                ors = " || ".join([
                    f"CONTAINS(STR(?p), '{s.replace(chr(39), chr(92)+chr(39)).replace(chr(34), chr(92)+chr(34))}')"
                    for s in incl
                ])
                incl_filter = f" && ({ors})"
        
        inner_rel = f"<{entity1_uri}> ?p ?o . FILTER(isIRI(?o) && isIRI(?p){incl_filter})"
        if values_clause:
            inner_rel = f"{inner_rel} {values_clause}"
        where_rel = _wrap_graph_where(graph_uri, inner_rel)
        q_rel = f"SELECT DISTINCT ?p ?o WHERE {{ {where_rel} }} LIMIT 20"
        rel_rows = kg.execute_query(q_rel) or []
        
        if not rel_rows:
            continue
        
        # Try each relation from entity1
        for row in random.sample(rel_rows, min(len(rel_rows), 5)):
            rel1_uri = row.get("p", {}).get("value")
            center_uri = row.get("o", {}).get("value")
            
            if not rel1_uri or not center_uri:
                continue
            
            # Get center entity name
            center_name_where = _wrap_graph_where(
                graph_uri,
                f"<{center_uri}> ns:type.object.name ?name . FILTER(LANGMATCHES(LANG(?name), 'en'))"
            )
            center_name_q = f"PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?name WHERE {{ {center_name_where} }} LIMIT 1"
            center_name_res = kg.execute_query(center_name_q)
            center_name = None
            if center_name_res and center_name_res[0].get("name", {}).get("value"):
                center_name = center_name_res[0]["name"]["value"]
            
            if not _is_readable_name(center_name, center_uri):
                continue
            
            # Now find all entities that connect to this center via different relations
            # We need to find entity3 and relation2 such that entity3 -> relation2 -> center
            # and count how many such (entity3, relation2) pairs exist
            
            # Get all incoming relations to center
            center_incoming_where = _wrap_graph_where(
                graph_uri,
                f"?head ?p <{center_uri}> . FILTER(isIRI(?head) && isIRI(?p) && ?p != <{rel1_uri}>)"
            )
            if include_predicate_substrings:
                incl = include_predicate_substrings
                include_are_uris = all(s.strip().startswith(("http://", "https://", "urn:")) for s in incl)
                if include_are_uris:
                    sample = random.sample(incl, 80) if len(incl) > 80 else incl
                    values_clause = f"VALUES ?p {{ {' '.join(f'<{u}>' for u in sample)} }}"
                    center_incoming_where = _wrap_graph_where(
                        graph_uri,
                        f"?head ?p <{center_uri}> . FILTER(isIRI(?head) && isIRI(?p) && ?p != <{rel1_uri}>) {values_clause}"
                    )
                else:
                    ors = " || ".join([
                        f"CONTAINS(STR(?p), '{s.replace(chr(39), chr(92)+chr(39)).replace(chr(34), chr(92)+chr(34))}')"
                        for s in incl
                    ])
                    center_incoming_where = _wrap_graph_where(
                        graph_uri,
                        f"?head ?p <{center_uri}> . FILTER(isIRI(?head) && isIRI(?p) && ?p != <{rel1_uri}> && ({ors}))"
                    )
            
            center_incoming_q = f"SELECT DISTINCT ?head ?p WHERE {{ {center_incoming_where} }} LIMIT 50"
            center_incoming_res = kg.execute_query(center_incoming_q) or []
            
            if not center_incoming_res:
                continue
            
            # Count valid (entity3, relation2) pairs with readable names
            valid_pairs = []
            for pair_row in center_incoming_res:
                entity3_uri = pair_row.get("head", {}).get("value")
                rel2_uri = pair_row.get("p", {}).get("value")
                
                if not entity3_uri or not rel2_uri or entity3_uri == entity1_uri:
                    continue
                
                # Get entity3 name
                entity3_name_where = _wrap_graph_where(
                    graph_uri,
                    f"<{entity3_uri}> ns:type.object.name ?name . FILTER(LANGMATCHES(LANG(?name), 'en'))"
                )
                entity3_name_q = f"PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?name WHERE {{ {entity3_name_where} }} LIMIT 1"
                entity3_name_res = kg.execute_query(entity3_name_q)
                entity3_name = None
                if entity3_name_res and entity3_name_res[0].get("name", {}).get("value"):
                    entity3_name = entity3_name_res[0]["name"]["value"]
                
                if _is_readable_name(entity3_name, entity3_uri):
                    valid_pairs.append((entity3_uri, entity3_name, rel2_uri))
            
            # Check if number of valid pairs (answer candidates) is within limit
            if len(valid_pairs) == 0 or len(valid_pairs) > max_answer_candidates:
                continue
            
            # Select one valid pair
            entity3_uri, entity3_name, rel2_uri = random.choice(valid_pairs)
            
            # Build path representation
            rel1_name = _relation_label(rel1_uri)
            rel2_name = _relation_label(rel2_uri)
            
            # Create path nodes
            path_nodes = [
                {
                    "node_uri": entity1_uri,
                    "node_name": entity1_name,
                    "relation_uri": rel1_uri,
                    "relation_name": rel1_name,
                },
                {
                    "node_uri": center_uri,
                    "node_name": center_name,
                    "relation_uri": None,
                    "relation_name": None,
                },
                {
                    "node_uri": entity3_uri,
                    "node_name": entity3_name,
                    "relation_uri": rel2_uri,
                    "relation_name": rel2_name,
                },
                {
                    "node_uri": center_uri,
                    "node_name": center_name,
                    "relation_uri": None,
                    "relation_name": None,
                },
            ]
            
            # Build name paths: separate into name_path1 and name_path2
            name_path1 = f"{entity1_name} -> {rel1_name} -> {center_name}"
            name_path2 = f"{entity3_name} -> {rel2_name} -> {center_name}"
            
            if verbose:
                print(f"Generated conjunction path:")
                print(f"  Path1: {name_path1}")
                print(f"  Path2: {name_path2}")
                print(f"  Topic entities: [{entity1_name}, {entity3_name}]")
                print(f"  Answers: [{center_name}] (candidates: {len(valid_pairs)})")
            
            return {
                "path_nodes": path_nodes,
                "name_path1": name_path1,
                "name_path2": name_path2,
                "topic_entities": [
                    {"uri": entity1_uri, "name": entity1_name},
                    {"uri": entity3_uri, "name": entity3_name},
                ],
                "answers": [
                    {"uri": center_uri, "name": center_name},
                ],
                "answer_candidates_count": len(valid_pairs),
                # For backward compatibility
                "center_entity": {"uri": center_uri, "name": center_name},
                "path1": {
                    "entity": {"uri": entity1_uri, "name": entity1_name},
                    "relation": {"uri": rel1_uri, "name": rel1_name},
                },
                "path2": {
                    "entity": {"uri": entity3_uri, "name": entity3_name},
                    "relation": {"uri": rel2_uri, "name": rel2_name},
                },
                "hop_count": 2,  # Total hops: path1 (1 hop) + path2 (1 hop) = 2
            }
    
    if verbose:
        print("Failed to find valid entity pairs for conjunction path.")
    return None


def generate_unique_conjunction_paths(
    kg: VirtuosoKG,
    n: int = 10,
    max_attempts: int = 200,
    require_unique_centers: bool = True,
    max_answer_candidates: int = 3,
    start_pool_size: int = 16,
    pred_sample_m: int = 4,
    per_pred_pool_k: int = 4,
    start_predicates: Optional[List[str]] = None,
    *,
    on_accept: Optional[Callable[[Dict], None]] = None,
    initial_seen: Optional[Set[str]] = None,
    graph_uri: Optional[str] = None,
    include_predicate_substrings: Optional[List[str]] = None,
) -> List[Dict]:
    """Generate n unique conjunction-type paths.
    
    Args:
        kg: VirtuosoKG instance
        n: Number of paths to generate
        max_attempts: Maximum attempts to generate paths
        require_unique_centers: If True, each path must have a unique center entity
        on_accept: Optional callback when a path is accepted
        initial_seen: Set of name_paths already seen (for deduplication)
        graph_uri: Optional graph URI
        include_predicate_substrings: Optional predicate filter list
    
    Returns:
        List of path dicts
    """
    from tqdm import tqdm
    
    seen = set(initial_seen) if initial_seen else set()
    centers = set()
    results = []
    attempts = 0
    
    with tqdm(total=n, desc="Generating conjunction paths", unit="paths") as pbar:
        while len(results) < n and attempts < max_attempts:
            attempts += 1
            p = generate_conjunction_path(
                kg,
                graph_uri=graph_uri,
                include_predicate_substrings=include_predicate_substrings,
                max_attempts=50,
                max_answer_candidates=max_answer_candidates,
                start_pool_size=start_pool_size,
                pred_sample_m=pred_sample_m,
                per_pred_pool_k=per_pred_pool_k,
                start_predicates=start_predicates,
                verbose=False,
            )
            
            if not p:
                continue
            
            # Use name_path1 and name_path2 for deduplication
            name_path1 = p.get("name_path1")
            name_path2 = p.get("name_path2")
            # Fallback to legacy name_path format for backward compatibility
            if not name_path1 or not name_path2:
                name_path = p.get("name_path")
                if not name_path:
                    continue
                # Parse legacy format: "path1 AND path2"
                if " AND " in name_path:
                    parts = name_path.split(" AND ", 1)
                    name_path1 = parts[0].strip()
                    name_path2 = parts[1].strip() if len(parts) > 1 else ""
                else:
                    continue
            
            # Create a unique identifier for deduplication
            path_key = f"{name_path1}|||{name_path2}"
            if path_key in seen:
                continue
            
            # Use answers[0] if available, otherwise fall back to center_entity
            answers = p.get("answers", [])
            center_uri = answers[0].get("uri") if answers else p.get("center_entity", {}).get("uri")
            if require_unique_centers and center_uri:
                if center_uri in centers:
                    continue
                centers.add(center_uri)
            
            seen.add(path_key)
            results.append(p)
            
            if on_accept:
                try:
                    on_accept(p)
                except Exception:
                    pass
            
            pbar.update(1)
    
    return results


def generate_unique_conjunction_paths_batched(
    endpoint_url: str,
    timeout_s: int,
    n: int,
    batch_size: int,
    max_attempts: int,
    require_unique_centers: bool = True,
    max_answer_candidates: int = 3,
    start_pool_size: int = 16,
    pred_sample_m: int = 4,
    per_pred_pool_k: int = 4,
    start_predicates: Optional[List[str]] = None,
    *,
    on_accept: Optional[Callable[[Dict], None]] = None,
    initial_seen: Optional[Set[str]] = None,
    graph_uri: Optional[str] = None,
    include_predicate_substrings: Optional[List[str]] = None,
) -> List[Dict]:
    """Generate n unique conjunction-type paths in parallel batches.
    
    Each batch runs multiple workers in parallel, each independently trying to generate
    conjunction paths. Results are deduplicated by path_key (name_path1|||name_path2).
    
    Args:
        endpoint_url: SPARQL endpoint
        timeout_s: Query timeout
        n: Target unique paths
        batch_size: Parallel workers per batch
        max_attempts: Total attempts budget
        require_unique_centers: If True, each path must have a unique center entity
        max_answer_candidates: Maximum number of answer candidates allowed
        start_pool_size: Size of start entity candidate pool
        pred_sample_m: Number of predicates to sample
        per_pred_pool_k: Number of entities per predicate
        start_predicates: Optional list of predicates for start entity sampling
        on_accept: Optional callback when a path is accepted
        initial_seen: Set of path_keys already seen (for deduplication)
        graph_uri: Optional graph URI
        include_predicate_substrings: Optional predicate filter list
    
    Returns:
        List of path dicts
    """
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    results: List[Dict] = []
    seen: Set[str] = set(initial_seen or set())
    centers: Set[str] = set()
    attempts = 0
    seen_lock = threading.Lock()
    centers_lock = threading.Lock()
    results_lock = threading.Lock()
    
    with tqdm(total=n, desc="Generating conjunction paths", unit="paths") as pbar:
        while len(results) < n and attempts < max_attempts:
            with ThreadPoolExecutor(max_workers=batch_size) as ex:
                def _worker():
                    kg_local = VirtuosoKG(endpoint_url=endpoint_url, timeout_s=timeout_s)
                    # Each worker independently tries to generate a conjunction path
                    p = generate_conjunction_path(
                        kg_local,
                        graph_uri=graph_uri,
                        include_predicate_substrings=include_predicate_substrings,
                        max_attempts=20,  # Fewer attempts per call since we're in parallel
                        max_answer_candidates=max_answer_candidates,
                        start_pool_size=start_pool_size,
                        pred_sample_m=pred_sample_m,
                        per_pred_pool_k=per_pred_pool_k,
                        start_predicates=start_predicates,
                        verbose=False,
                    )
                    return p
                
                futures = [ex.submit(_worker) for _ in range(batch_size)]
                
                for fut in as_completed(futures):
                    attempts += 1
                    p = fut.result()
                    if not p:
                        continue
                    
                    # Use name_path1 and name_path2 for deduplication
                    name_path1 = p.get("name_path1")
                    name_path2 = p.get("name_path2")
                    # Fallback to legacy name_path format
                    if not name_path1 or not name_path2:
                        name_path = p.get("name_path")
                        if not name_path:
                            continue
                        if " AND " in name_path:
                            parts = name_path.split(" AND ", 1)
                            name_path1 = parts[0].strip()
                            name_path2 = parts[1].strip() if len(parts) > 1 else ""
                        else:
                            continue
                    
                    # Create a unique identifier for deduplication
                    path_key = f"{name_path1}|||{name_path2}"
                    
                    # Thread-safe check and update
                    with seen_lock:
                        if path_key in seen:
                            continue
                        seen.add(path_key)
                    
                    # Check unique centers if required
                    answers = p.get("answers", [])
                    center_uri = answers[0].get("uri") if answers else p.get("center_entity", {}).get("uri")
                    if require_unique_centers and center_uri:
                        with centers_lock:
                            if center_uri in centers:
                                continue
                            centers.add(center_uri)
                    
                    # Thread-safe append
                    should_break = False
                    with results_lock:
                        if len(results) >= n:
                            should_break = True
                        else:
                            results.append(p)
                            if on_accept:
                                try:
                                    on_accept(p)
                                except Exception:
                                    pass
                            pbar.update(1)
                    
                    if should_break or len(results) >= n:
                        break
                
                if len(results) >= n:
                    break
    
    return results

