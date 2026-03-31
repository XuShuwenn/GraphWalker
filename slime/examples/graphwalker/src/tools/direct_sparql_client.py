"""Direct SPARQL client with entity name resolution and BM25-based ranking."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging
import os
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from rank_bm25 import BM25Okapi

from .entity_resolver import EntityResolver
from .relation_normalizer import normalize_relation

logger = logging.getLogger(__name__)

# Performance logger for SPARQL operations
_sparql_perf_logger = logging.getLogger("graphwalker.sparql_performance")
_sparql_perf_logger.setLevel(logging.INFO)
if not _sparql_perf_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[SPARQL-PERF] %(message)s'))
    _sparql_perf_logger.addHandler(handler)


def _setup_sparql_no_proxy():
    """Setup environment to bypass proxy for SPARQL requests."""
    # Add localhost and 127.0.0.1 to no_proxy to bypass proxy for local SPARQL endpoints
    no_proxy = os.environ.get('no_proxy', '')
    no_proxy_list = [x.strip() for x in no_proxy.split(',') if x.strip()]
    for host in ['localhost', '127.0.0.1', '0.0.0.0']:
        if host not in no_proxy_list:
            no_proxy_list.append(host)
    os.environ['no_proxy'] = ','.join(no_proxy_list)
    # Also set NO_PROXY (uppercase) for compatibility
    os.environ['NO_PROXY'] = os.environ['no_proxy']


class DirectSPARQLKGClient:
    """SPARQL client: resolves entity names, queries Virtuoso, ranks results."""
    
    def __init__(self, sparql_endpoint: str = "http://localhost:18890/sparql", timeout: int = 10, 
                 llm_filter_callback: Optional[Any] = None,
                 dataset_type: str = "cwq",
                 full_list: bool = False):
        """Init client with endpoint and timeout; set up resolver and caches.
        
        Args:
            sparql_endpoint: SPARQL endpoint URL
            timeout: Query timeout in seconds
            llm_filter_callback: Optional callback function for LLM-based relation filtering.
                                Should accept (relations: List[Dict[str, str]], question: str, entity_name: str, use_flatten_prompt: bool = False)
                                and return filtered List[Dict[str, str]]
                                If use_flatten_prompt=True, should use flatten relation filter prompt
            dataset_type: Dataset type (cwq, webqsp, grailqa) to determine which whitelist to use
            full_list: If True, use the merged full whitelist instead of dataset-specific one
        """
        self.sparql_endpoint = sparql_endpoint
        self.timeout = timeout
        # Setup no_proxy for local SPARQL endpoints
        _setup_sparql_no_proxy()
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(timeout)
        self.resolver = EntityResolver(sparql_endpoint, timeout=timeout)
        # Load and enable whitelist filtering
        self._relation_whitelist = self._load_relation_whitelist(dataset_type, full_list)
        self._llm_filter_callback = llm_filter_callback
        self._rel_cache: Dict[str, List[str]] = {}
        # CVT node mapping: {flatten_relation: (original_relation, cvt_node_id, cvt_relation, direction)}
        # direction: 'out' for outgoing (entity -> relation -> cvt -> cvt_rel -> tail)
        #            'in' for incoming (head -> cvt_rel -> cvt -> relation -> entity)
        self._cvt_mapping: Dict[str, tuple] = {}
        # Pending flatten relations per entity: {entity_id: [flatten_relations]}
        self._pending_flatten_relations: Dict[str, List[str]] = {}
        # Pending flatten candidates for async LLM filtering: {entity_id: {candidates, question, entity_name}}
        self._pending_flatten_candidates: Dict[str, Dict] = {}

    def _load_relation_whitelist(self, dataset_type: str = "cwq", full_list: bool = False) -> Optional[set]:
        """Load predicate whitelist based on dataset type and full_list flag.
        
        Args:
            dataset_type: Dataset type (cwq, webqsp, grailqa)
            full_list: If True, use merged full whitelist; otherwise use dataset-specific whitelist
            
        Returns:
            Set of dotted relation IDs or None
        """
        import os, json
        
        # Calculate path to graphwalker directory
        # From src/tools/direct_sparql_client.py -> src/tools -> src -> graphwalker
        here = os.path.dirname(__file__)
        graphwalker_root = os.path.abspath(os.path.join(here, "..", ".."))
        
        # Whitelist files are now in graphwalker/datasets/predicates_white_list/
        whitelist_dir = os.path.join(graphwalker_root, "datasets", "predicates_white_list")
        
        # Determine whitelist path
        if full_list:
            wl_path = os.path.join(whitelist_dir, "filtered_full_white_list.json")
        else:
            # Map dataset type to specific whitelist file
            whitelist_map = {
                "cwq": os.path.join(whitelist_dir, "filtered_cwq_white_list.json"),
                "webqsp": os.path.join(whitelist_dir, "filtered_webqsp_white_list.json"),
                "grailqa": os.path.join(whitelist_dir, "filtered_grailqa_white_list.json"),
                "simpleqa": os.path.join(whitelist_dir, "filtered_cwq_white_list.json"),  # SimpleQA uses CWQ whitelist
            }
            wl_path = whitelist_map.get(dataset_type.lower())
            if not wl_path:
                logger.warning(f"Unknown dataset type '{dataset_type}', falling back to CWQ whitelist")
                wl_path = whitelist_map["cwq"]
        
        if not os.path.exists(wl_path):
            logger.warning(f"Whitelist file not found: {wl_path}, predicate filtering disabled")
            return None
            
        with open(wl_path, "r", encoding="utf-8") as f:
            uris = json.load(f)

        prefix = "http://rdf.freebase.com/ns/"
        dotted = {u[len(prefix):].replace("/", ".") for u in uris if u.startswith(prefix)}
        logger.info(f"Loaded {len(dotted)} relations from whitelist: {os.path.basename(wl_path)}")
        return dotted

    def _apply_relation_whitelist(self, relations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter relations by whitelist; fallback to original if all filtered out."""
        if not relations or not self._relation_whitelist:
            return relations
        filtered = [r for r in relations if r.get("relation") in self._relation_whitelist]
        # If filtering removes everything, fall back to original to avoid total loss
        return filtered or relations
    
    def _is_cvt_node(self, tail_id: str, tail_name: str) -> bool:
        """Check if a node is a CVT node (no entity name, ID starting with m. or g.)."""
        # CVT nodes typically have no name or name equals ID, and ID starts with m. or g.
        if not (tail_id.startswith('m.') or tail_id.startswith('g.')):
            return False
        # If tail_name is empty or equals tail_id, it's likely a CVT node
        if not tail_name or tail_name == tail_id:
            return True
        # Additional check: if name is just the MID format, it's a CVT
        if tail_name.replace('/', '.').replace(':', '.') == tail_id:
            return True
        return False
    
    def _is_meaningless_pattern_relation(self, relation: str) -> bool:
        """Check if a relation is a meaningless pattern relation that should be skipped.
        
        These are metadata relations that don't provide meaningful information for the model,
        such as type.object.type, government.government_position_held, etc.
        """
        # Patterns to skip (exact matches or prefixes)
        skip_patterns = [
            # 1. 用于描述类型体系(Type System)的关系
            "type.object.type",
            "type.type.instance",
            "type.type.properties",
            "type.property.schema",
            # 2. CVT 上用于声明 schema / 结构化元数据的关系
            "common.topic.article",
            "common.topic.image",
            "common.image_source",
            "common.document",
            "common.webpage",
            "common.license",
            "common.topic.description",
            # 3. "schema-only" CVT 关系（不会构成真实语义路径）
            "freebase.type_hints.included_types",
            "freebase.property_hints.best_value_properties",
            "freebase.property_hints.mediator",
            "freebase.property_hints.unit",
            "freebase.property_hints.expected_type",
            "freebase.type_hints.enumeration"
            # 4. 所有“type.object.*”下的关系（层级、域、权限等）
            "type.object.key",             # Key 信息
            "type.object.permission",      # 权限信息
            "type.object.profile",         # 元描述
            "type.object.service"          # 服务类元数据
            # 5. 用于 CVT 内部结构组织的"辅助关系"
            # 这些关系不会作为 path hop，用来连接真正实体
            "common.topic.topic_equivalent_webpage",
            "common.topic.topic_equivalent",
        ]
        
        for pattern in skip_patterns:
            if relation == pattern or relation.startswith(pattern + "."):
                return True
        return False
    
    def _get_common_prefix(self, rel1: str, rel2: str) -> str:
        """Get the common prefix of two relations (dot-separated)."""
        parts1 = rel1.split('.')
        parts2 = rel2.split('.')
        common = []
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                common.append(p1)
            else:
                break
        return '.'.join(common) if common else ""
    
    def _flatten_relation(self, rel1: str, rel2: str) -> str:
        """Flatten two relations by removing common prefix and concatenating."""
        common_prefix = self._get_common_prefix(rel1, rel2)
        if common_prefix:
            # Remove common prefix from rel1
            rel1_suffix = rel1[len(common_prefix):].lstrip('.')
            # Concatenate: rel1_suffix + rel2
            return f"{rel1_suffix}.{rel2}"
        else:
            # No common prefix, just concatenate
            return f"{rel1}.{rel2}"
    
    def _tokenize(self, text: str) -> List[str]:
        """Lowercase and split on whitespace/punctuation."""
        if not text:
            return []
        # Replace common punctuation with spaces
        for char in '.,!?;:()[]{}"\'/\\':
            text = text.replace(char, ' ')
        return [t.lower() for t in text.split() if t.strip()]
    
    def rank_by_similarity(self, items: List[Dict[str, str]], question: str, text_field: str = "name") -> List[Dict[str, str]]:
        """Rank items by BM25 similarity to question using a text field."""
        if not items or not question:
            return items
        
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return items
        
        docs_tokens = [
            self._tokenize(item.get(text_field, "") or item.get("entity", "") or item.get("relation", ""))
            for item in items
        ]
        if not any(docs_tokens):
            return items

        bm25 = BM25Okapi(docs_tokens)
        scores = bm25.get_scores(query_tokens)
        scored_items = sorted(zip(scores, items), key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items]
    
    def _resolve_entity(self, entity: str) -> Optional[str]:
        """Return entity ID for a name or pass through if already an ID.
        
        Includes fast-fail heuristics to skip entities that are clearly not entity names,
        avoiding expensive SPARQL queries for invalid inputs.
        """
        if not entity:
            return None
        
        if entity.startswith(('m.', 'g.', 'en.')):
            return entity
        
        # Fast-fail heuristics: skip entities that are clearly not entity names
        # 1. Very long strings (>100 chars) are likely not entity names
        if len(entity) > 100:
            logger.debug(f"Skipping entity resolution for long string: '{entity[:50]}...'")
            return None
        
        
        entity_id = self.resolver.resolve(entity)
        if entity_id:
            return entity_id
        else:
            logger.warning(f"Failed to resolve entity: '{entity}'")
            return None
    
    def get_relations(self, entity: str, limit: int = 100, question: str = "", top_k: int = 10, 
                     include_flatten: bool = True) -> List[Dict[str, str]]:
        """Get all relations (head and tail) for an entity; whitelist-filter and BM25-rank.
        
        Args:
            entity: Entity name or ID
            limit: Max relations to query
            question: Question text for BM25 ranking
            top_k: Top K relations to return
            include_flatten: Whether to include pending flatten relations found in get_triples
        """
        # Resolve entity name to ID
        entity_id = self._resolve_entity(entity)
        if not entity_id:
            logger.warning(f"Cannot resolve entity '{entity}', returning empty results")
            return []
        
        try:
            # Try cache first
            if entity_id in self._rel_cache:
                cached = self._rel_cache[entity_id]
                relations = [{"relation": rid} for rid in cached]
            else:
                # Split into two queries to avoid timeout on complex UNIONs
                rel_ids = set()
                
                # 1. Outgoing relations
                query_out = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT DISTINCT ?relation WHERE {{
                    ns:{entity_id} ?relation ?tail .
                    FILTER(isIRI(?relation))
                    FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/"))
                }} LIMIT {limit}
                """
                sparql_start = time.time()
                self.sparql.setQuery(query_out)
                results_out = self.sparql.query().convert()
                sparql_time = time.time() - sparql_start
                if sparql_time > 0.5:  # Only log slow queries (>500ms)
                    _sparql_perf_logger.info(f"SPARQL query (outgoing relations for {entity_id}): {sparql_time:.2f}s")
                for b in results_out.get("results", {}).get("bindings", []):
                    val = b.get("relation", {}).get("value")
                    if val:
                        rel_ids.add(val.replace("http://rdf.freebase.com/ns/", "").replace("/", "."))

                # 2. Incoming relations
                query_in = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT DISTINCT ?relation WHERE {{
                    ?head ?relation ns:{entity_id} .
                    FILTER(isIRI(?relation))
                    FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/"))
                }} LIMIT {limit}
                """
                sparql_start = time.time()
                self.sparql.setQuery(query_in)
                results_in = self.sparql.query().convert()
                sparql_time = time.time() - sparql_start
                if sparql_time > 0.5:  # Only log slow queries (>500ms)
                    _sparql_perf_logger.info(f"SPARQL query (incoming relations for {entity_id}): {sparql_time:.2f}s")
                for b in results_in.get("results", {}).get("bindings", []):
                    val = b.get("relation", {}).get("value")
                    if val:
                        rel_ids.add(val.replace("http://rdf.freebase.com/ns/", "").replace("/", "."))
                
                rel_ids_list = list(rel_ids)
                relations = [{"relation": rid} for rid in rel_ids_list]
                self._rel_cache[entity_id] = rel_ids_list
            
            # Add pending flatten relations for this specific entity if requested
            # Note: Do NOT clear _pending_flatten_relations here - they should persist for the entire question
            if include_flatten and entity_id in self._pending_flatten_relations:
                entity_flatten_rels = self._pending_flatten_relations[entity_id]
                for flatten_rel in entity_flatten_rels:
                    if flatten_rel not in [r.get("relation") for r in relations]:
                        relations.append({"relation": flatten_rel})
            
            # Apply whitelist; fallback if empty
            relations = self._apply_relation_whitelist(relations)
            
            # Filter out meaningless pattern relations before BM25 ranking
            relations = [r for r in relations if not self._is_meaningless_pattern_relation(r.get("relation", ""))]
            
            # Rank by BM25 similarity if question provided
            if question and relations:
                relations = self.rank_by_similarity(relations, question, "relation")
            
            result = relations[:top_k]
            return result
            
        except Exception as e:
            logger.error(f"Error in get_relations: {e}")
            return []

    def get_triples(self, entity: str, relations: List[str], limit_per_relation: int = 20, question: str = "", return_with_cvt_info: bool = False) -> Any:
        """Get triples for entity and a list of relations. 
        
        Args:
            entity: Entity name or ID
            relations: List of relation names
            limit_per_relation: Max triples per relation
            question: Question text (for future use)
            return_with_cvt_info: If True, returns dict with 'triples' and 'cvt_info'.
                                 If False (default), returns only triples list (backward compatible).
        
        Returns:
            If return_with_cvt_info=False: List of dicts {'head': name, 'head_id': id, 'relation':..., 'tail': name, 'tail_id': id}
            If return_with_cvt_info=True: Dict with:
                - 'triples': list of triples
                - 'cvt_info': list of CVT information dicts (if any CVT nodes were found)
        
        Handles CVT nodes automatically:
        - Detects CVT nodes (nodes without entity names)
        - Automatically flattens relations through CVT nodes
        - Maintains mapping for flatten relations
        - Resolves flatten relations to two-hop queries when used
        """
        import random
        entity_id = self._resolve_entity(entity)
        if not entity_id:
            # Return format based on return_with_cvt_info parameter
            if return_with_cvt_info:
                return {"triples": [], "cvt_info": []}
            else:
                return []
            
        # Get entity name for consistent display
        entity_name = entity
        # Try to get English name, fallback to any name
        query_name = f"""
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?name WHERE {{
            ns:{entity_id} ns:type.object.name ?name .
        }} LIMIT 50
        """
        try:
            self.sparql.setQuery(query_name)
            res = self.sparql.query().convert()
            if res:
                bindings = res.get("results", {}).get("bindings", [])
                if bindings:
                    best_name = None
                    for b in bindings:
                        val = b.get("name", {}).get("value", "")
                        lang = b.get("name", {}).get("xml:lang", "")
                        if lang == 'en':
                            best_name = val
                            break
                        if not best_name: # Keep first as fallback
                            best_name = val
                    
                    if best_name:
                        entity_name = best_name
        except Exception:
            # Query failed (network error, timeout, etc.), use original entity name
            pass

        # Deduplicate relations list to avoid querying the same relation multiple times
        # Preserve order while removing duplicates
        seen_relations = []
        seen_set = set()
        for rel in relations:
            if rel not in seen_set:
                seen_relations.append(rel)
                seen_set.add(rel)
        relations = seen_relations

        all_triples = []
        flatten_relations = []  # Collect flatten relations to return via get_relations
        cvt_info = []  # Collect CVT node information: [{original_rel, cvt_node_id, cvt_rel, flatten_rel, flattened_triples}, ...]
        flatten_rel_counter = {}  # Track flatten relation name usage to handle duplicates
        
        # Collect all flatten candidates from ALL relations before processing
        # This ensures we process all CVT nodes together, not per-relation
        all_flatten_candidates_out = []  # Collect all flatten candidates from all CVT nodes (outgoing)
        all_flatten_candidates_in = []  # Collect all flatten candidates from all CVT nodes (incoming)
        
        for relation in relations:
            # Check if this is a flatten relation that needs to be resolved
            if relation in self._cvt_mapping:
                mapping = self._cvt_mapping[relation]
                if len(mapping) == 4:
                    original_rel, cvt_node_id, cvt_rel, direction = mapping
                else:
                    # Backward compatibility: old format without direction
                    original_rel, cvt_node_id, cvt_rel = mapping
                    direction = 'out'
                
                # Resolve flatten relation to two-hop query
                triples_for_rel = self._query_flatten_relation(
                    entity_id, entity_name, original_rel, cvt_node_id, cvt_rel, relation, limit_per_relation, direction
                )
                all_triples.extend(triples_for_rel)
                continue
            
            relation_id = normalize_relation(relation)
            triples_for_rel = []
            cvt_nodes_found_out = []  # Track CVT nodes found in outgoing relations
            cvt_nodes_found_in = []  # Track CVT nodes found in incoming relations
            
            # Outgoing: (entity, relation, ?tail)
            try:
                # Use GROUP BY to get one row per tail, prioritizing English names
                query_tail = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT ?tail (SAMPLE(?name_en) as ?preferred_name) (SAMPLE(?name_any) as ?fallback_name) WHERE {{
                    ns:{entity_id} ns:{relation_id} ?tail .
                    OPTIONAL {{ ?tail ns:type.object.name ?name_en . FILTER(LANGMATCHES(LANG(?name_en), 'en')) }}
                    OPTIONAL {{ ?tail ns:type.object.name ?name_any . }}
                }} GROUP BY ?tail LIMIT {limit_per_relation * 2} 
                """
                self.sparql.setQuery(query_tail)
                results = self.sparql.query().convert()
                bindings = results.get("results", {}).get("bindings", [])
                for b in bindings:
                    tail_uri = b.get("tail", {}).get("value", "")
                    if tail_uri:
                        tail_id = tail_uri.replace("http://rdf.freebase.com/ns/", "").replace("/", ".")
                        name = b.get("preferred_name", {}).get("value", "") or b.get("fallback_name", {}).get("value", "")
                        
                        # Check if this is a CVT node
                        if self._is_cvt_node(tail_id, name):
                            cvt_nodes_found_out.append((tail_id, relation))
                        else:
                            triple_data = {
                                "head": entity_name,
                                "head_id": entity_id,
                                "relation": relation,
                                "tail": name if name else tail_id,
                                "tail_id": tail_id
                            }
                            triples_for_rel.append(triple_data)
            except Exception as e:
                logger.error(f"Error querying tail triples for {relation}: {e}")

            # Incoming: (?head, relation, entity)
            try:
                query_head = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT ?head (SAMPLE(?name_en) as ?preferred_name) (SAMPLE(?name_any) as ?fallback_name) WHERE {{
                    ?head ns:{relation_id} ns:{entity_id} .
                    OPTIONAL {{ ?head ns:type.object.name ?name_en . FILTER(LANGMATCHES(LANG(?name_en), 'en')) }}
                    OPTIONAL {{ ?head ns:type.object.name ?name_any . }}
                }} GROUP BY ?head LIMIT {limit_per_relation * 4}
                """
                self.sparql.setQuery(query_head)
                results = self.sparql.query().convert()
                bindings = results.get("results", {}).get("bindings", [])
                for b in bindings:
                    head_uri = b.get("head", {}).get("value", "")
                    if head_uri:
                        head_id = head_uri.replace("http://rdf.freebase.com/ns/", "").replace("/", ".")
                        name = b.get("preferred_name", {}).get("value", "") or b.get("fallback_name", {}).get("value", "")
                        
                        # Check if this is a CVT node
                        if self._is_cvt_node(head_id, name):
                            cvt_nodes_found_in.append((head_id, relation))
                        else:
                            triple_data = {
                                "head": name if name else head_id,
                                "head_id": head_id,
                                "relation": relation,
                                "tail": entity_name,
                                "tail_id": entity_id
                            }
                            triples_for_rel.append(triple_data)
            except Exception as e:
                logger.error(f"Error querying head triples for {relation}: {e}")
            
            # Collect CVT nodes for this relation (will be processed globally after all relations)
            # Process CVT nodes in outgoing relations: collect candidates
            for cvt_node_id, original_rel in cvt_nodes_found_out:
                try:
                    # Get outgoing relations from CVT node
                    query_cvt_rels = f"""
                    PREFIX ns: <http://rdf.freebase.com/ns/>
                    SELECT DISTINCT ?relation WHERE {{
                        ns:{cvt_node_id} ?relation ?tail .
                        FILTER(isIRI(?relation))
                        FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/"))
                    }} LIMIT 50
                    """
                    self.sparql.setQuery(query_cvt_rels)
                    cvt_results = self.sparql.query().convert()
                    cvt_bindings = cvt_results.get("results", {}).get("bindings", [])
                    
                    # Collect CVT relations and apply whitelist filtering before flattening
                    cvt_relations_raw = []
                    for cvt_b in cvt_bindings:
                        cvt_rel_uri = cvt_b.get("relation", {}).get("value", "")
                        if cvt_rel_uri:
                            cvt_rel_id = cvt_rel_uri.replace("http://rdf.freebase.com/ns/", "").replace("/", ".")
                            # Skip meaningless pattern relations
                            if self._is_meaningless_pattern_relation(cvt_rel_id):
                                continue
                            cvt_relations_raw.append(cvt_rel_id)
                    
                    # Apply whitelist filtering to CVT relations before flattening
                    if self._relation_whitelist and cvt_relations_raw:
                        cvt_relations_filtered = [r for r in cvt_relations_raw if r in self._relation_whitelist]
                        if not cvt_relations_filtered:
                            # If whitelist filters out all relations, fall back to original
                            cvt_relations_filtered = cvt_relations_raw
                        cvt_relations = cvt_relations_filtered
                    else:
                        cvt_relations = cvt_relations_raw
                    
                    # Create flatten relations for each CVT relation
                    for cvt_rel_id in cvt_relations:
                        base_flatten_rel = self._flatten_relation(original_rel, cvt_rel_id)
                        
                        # Handle duplicate flatten relation names by adding numeric suffix
                        if base_flatten_rel not in flatten_rel_counter:
                            flatten_rel_counter[base_flatten_rel] = 0
                        flatten_rel_counter[base_flatten_rel] += 1
                        flatten_rel = f"{base_flatten_rel}_{flatten_rel_counter[base_flatten_rel]}"
                        
                        # Collect candidate for global ranking
                        all_flatten_candidates_out.append({
                            "flatten_relation": flatten_rel,
                            "original_relation": original_rel,
                            "cvt_node_id": cvt_node_id,
                            "cvt_relation": cvt_rel_id,
                            "direction": "out"
                        })
                except Exception as e:
                    logger.error(f"Error processing outgoing CVT node {cvt_node_id}: {e}")
                    continue
            
            # Collect CVT nodes for this relation (will be processed globally after all relations)
            # Process CVT nodes in incoming relations: collect candidates
            for cvt_node_id, original_rel in cvt_nodes_found_in:
                try:
                    # Get incoming relations to CVT node
                    query_cvt_rels = f"""
                    PREFIX ns: <http://rdf.freebase.com/ns/>
                    SELECT DISTINCT ?relation WHERE {{
                        ?head ?relation ns:{cvt_node_id} .
                        FILTER(isIRI(?relation))
                        FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/"))
                    }} LIMIT 50
                    """
                    self.sparql.setQuery(query_cvt_rels)
                    cvt_results = self.sparql.query().convert()
                    cvt_bindings = cvt_results.get("results", {}).get("bindings", [])
                    
                    # Collect CVT relations and apply whitelist filtering before flattening
                    cvt_relations_raw = []
                    for cvt_b in cvt_bindings:
                        cvt_rel_uri = cvt_b.get("relation", {}).get("value", "")
                        if cvt_rel_uri:
                            cvt_rel_id = cvt_rel_uri.replace("http://rdf.freebase.com/ns/", "").replace("/", ".")
                            # Skip meaningless pattern relations
                            if self._is_meaningless_pattern_relation(cvt_rel_id):
                                continue
                            cvt_relations_raw.append(cvt_rel_id)
                    
                    # Apply whitelist filtering to CVT relations before flattening
                    if self._relation_whitelist and cvt_relations_raw:
                        cvt_relations_filtered = [r for r in cvt_relations_raw if r in self._relation_whitelist]
                        if not cvt_relations_filtered:
                            # If whitelist filters out all relations, fall back to original
                            cvt_relations_filtered = cvt_relations_raw
                        cvt_relations = cvt_relations_filtered
                    else:
                        cvt_relations = cvt_relations_raw
                    
                    # Create flatten relations for each CVT relation
                    for cvt_rel_id in cvt_relations:
                        # Create flatten relation (cvt_rel comes first, then original_rel)
                        # For incoming: head -> cvt_rel -> cvt -> original_rel -> entity
                        base_flatten_rel = self._flatten_relation(cvt_rel_id, original_rel)
                        
                        # Handle duplicate flatten relation names by adding numeric suffix
                        if base_flatten_rel not in flatten_rel_counter:
                            flatten_rel_counter[base_flatten_rel] = 0
                        flatten_rel_counter[base_flatten_rel] += 1
                        flatten_rel = f"{base_flatten_rel}_{flatten_rel_counter[base_flatten_rel]}"
                        
                        # Collect candidate for global ranking
                        all_flatten_candidates_in.append({
                            "flatten_relation": flatten_rel,
                            "original_relation": original_rel,
                            "cvt_node_id": cvt_node_id,
                            "cvt_relation": cvt_rel_id,
                            "direction": "in"
                        })
                except Exception as e:
                    logger.error(f"Error processing incoming CVT node {cvt_node_id}: {e}")
                    continue
            
            # Add regular triples for this relation (CVT flatten triples will be added globally later)
            # Randomly sample: adjust limit based on whether CVT nodes were found
            # If CVT nodes were found (needs flattening), use limit 15; otherwise use limit 5
            has_cvt_nodes = len(cvt_nodes_found_out) > 0 or len(cvt_nodes_found_in) > 0
            effective_limit = 15 if has_cvt_nodes else 5
            
            if len(triples_for_rel) > effective_limit:
                triples_for_rel = random.sample(triples_for_rel, effective_limit)
            
            all_triples.extend(triples_for_rel)
        
        # After processing all relations, process all CVT flatten candidates globally
        # Combine all flatten candidates (outgoing + incoming) and process globally
        all_flatten_candidates = all_flatten_candidates_out + all_flatten_candidates_in
        
        # Global ranking and filtering for all flatten candidates from all relations
        if all_flatten_candidates:
            # Step 1: BM25 ranking to get top-80
            if question:
                query_text = f"{question} {entity_name}".strip()
                ranked_candidates = self.rank_by_similarity(
                    [{"relation": c["flatten_relation"]} for c in all_flatten_candidates],
                    query_text,
                    "relation"
                )
                # Get top-80
                top_80_flatten_rels = {r["relation"] for r in ranked_candidates[:80]}
                top_80_candidates = [c for c in all_flatten_candidates if c["flatten_relation"] in top_80_flatten_rels]
            else:
                # No question provided, just take first 80
                top_80_candidates = all_flatten_candidates[:80]
            
            # Step 2: Query triples for top-80 candidates and filter out meaningless ones
            # A triple is meaningless if head_id == tail_id (exact match)
            # Collect candidates that produce at least one meaningful triple
            meaningful_candidates = []  # Candidates that produce meaningful triples
            
            for candidate in top_80_candidates:
                flatten_rel = candidate["flatten_relation"]
                original_rel = candidate["original_relation"]
                cvt_node_id = candidate["cvt_node_id"]
                cvt_rel_id = candidate["cvt_relation"]
                direction = candidate["direction"]
                
                # Query flattened triples (use limit_per_relation from function parameter)
                flattened_triples = self._query_flatten_relation(
                    entity_id, entity_name, original_rel, cvt_node_id, cvt_rel_id, 
                    flatten_rel, limit_per_relation=limit_per_relation, direction=direction
                )
                
                # Filter out triples where head_id == tail_id (meaningless)
                meaningful_triples = []
                for triple in flattened_triples:
                    head_id = triple.get("head_id", "")
                    tail_id = triple.get("tail_id", "")
                    # Keep only triples where head and tail are different
                    if head_id and tail_id and head_id != tail_id:
                        meaningful_triples.append(triple)
                
                # Only keep this candidate if it produces at least one meaningful triple
                if meaningful_triples:
                    meaningful_candidates.append(candidate)
                    logger.debug(f"Found meaningful flatten relation: {flatten_rel} (direction: {direction}, {len(meaningful_triples)} meaningful triples)")
                else:
                    logger.debug(f"Skipping flatten relation {flatten_rel}: all triples have identical head and tail entities (head_id == tail_id)")
            
            # Step 3: Prepare candidates for LLM filter (to be called externally if needed)
            # Store pending flatten candidates for external async filtering
            # This avoids blocking the thread with sync LLM API calls
            final_candidates = []
            
            # If LLM filter callback is available, store candidates for external filtering
            # The caller can then use async filter if available
            if self._llm_filter_callback and meaningful_candidates:
                # Store for potential external async filtering
                self._pending_flatten_candidates[entity_id] = {
                    "candidates": meaningful_candidates,
                    "question": question or "",
                    "entity_name": entity_name
                }
                # For now, just take first 8 as fallback
                # The external async filter can override this if called
                final_candidates = meaningful_candidates[:8]
            else:
                # No LLM filter available, just take first 8 meaningful candidates
                final_candidates = meaningful_candidates[:8]
            
            # Step 4: Query triples for final top-8 candidates and merge with regular triples
            flatten_triples_all = []  # Collect all flatten triples
            for candidate in final_candidates:
                flatten_rel = candidate["flatten_relation"]
                original_rel = candidate["original_relation"]
                cvt_node_id = candidate["cvt_node_id"]
                cvt_rel_id = candidate["cvt_relation"]
                direction = candidate["direction"]
                
                # Query flattened triples again (use limit_per_relation from function parameter)
                flattened_triples = self._query_flatten_relation(
                    entity_id, entity_name, original_rel, cvt_node_id, cvt_rel_id, 
                    flatten_rel, limit_per_relation=limit_per_relation, direction=direction
                )
                
                # Filter out triples where head_id == tail_id (meaningless)
                meaningful_triples = []
                for triple in flattened_triples:
                    head_id = triple.get("head_id", "")
                    tail_id = triple.get("tail_id", "")
                    # Keep only triples where head and tail are different
                    if head_id and tail_id and head_id != tail_id:
                        meaningful_triples.append(triple)
                
                # Only process if we have at least one meaningful triple
                # (This can happen if query results differ between Step 2 and Step 4)
                if meaningful_triples:
                    # Store mapping
                    self._cvt_mapping[flatten_rel] = (original_rel, cvt_node_id, cvt_rel_id, direction)
                    flatten_relations.append(flatten_rel)
                    logger.debug(f"Final flatten relation: {flatten_rel} (direction: {direction}, {len(meaningful_triples)} meaningful triples)")
                    
                    # Add meaningful triples
                    flatten_triples_all.extend(meaningful_triples)
                    
                    # Collect CVT information
                    cvt_info.append({
                        "original_relation": original_rel,
                        "cvt_node_id": cvt_node_id,
                        "cvt_relation": cvt_rel_id,
                        "flatten_relation": flatten_rel,
                        "direction": direction,
                        "flattened_triples": meaningful_triples
                    })
                else:
                    logger.warning(f"Final flatten relation {flatten_rel} produced no meaningful triples in Step 4 (all head_id == tail_id), skipping")
            
            logger.info(f"Selected {len(flatten_relations)} meaningful flatten relations (target: 8) from {len(meaningful_candidates)} candidates for entity {entity_id}")
            
            # Add all flatten triples to all_triples (merged with regular triples)
            all_triples.extend(flatten_triples_all)
            
        # If we found flatten relations, store them per entity for later retrieval via get_relations
        if flatten_relations:
            logger.info(f"Found {len(flatten_relations)} flatten relations for {entity_id}: {flatten_relations[:5]}...")
            # Store flatten relations per entity
            if entity_id not in self._pending_flatten_relations:
                self._pending_flatten_relations[entity_id] = []
            # Add new flatten relations (avoid duplicates)
            existing = set(self._pending_flatten_relations[entity_id])
            for flatten_rel in flatten_relations:
                if flatten_rel not in existing:
                    self._pending_flatten_relations[entity_id].append(flatten_rel)
        
        # Return format based on return_with_cvt_info parameter
        if return_with_cvt_info:
            return {
                "triples": all_triples,
                "cvt_info": cvt_info if cvt_info else []
            }
        else:
            # Backward compatible: return only triples list
            return all_triples

    def _query_flatten_relation(self, entity_id: str, entity_name: str, original_rel: str, 
                                cvt_node_id: str, cvt_rel: str, flatten_rel: str, 
                                limit_per_relation: int, direction: str = 'out') -> List[Dict[str, str]]:
        """Query a flatten relation by resolving it to a two-hop query.
        
        Args:
            entity_id: The entity ID
            entity_name: The entity name
            original_rel: The original relation (relation1 for outgoing, relation2 for incoming)
            cvt_node_id: The CVT node ID
            cvt_rel: The CVT relation (cvt_out_rel for outgoing, cvt_in_rel for incoming)
            flatten_rel: The flatten relation name to return
            limit_per_relation: Limit for results
            direction: 'out' for outgoing (entity -> original_rel -> cvt -> cvt_rel -> tail)
                      'in' for incoming (head -> cvt_rel -> cvt -> original_rel -> entity)
        """
        triples = []
        try:
            if direction == 'out':
                # Outgoing: entity -> original_rel -> cvt_node -> cvt_rel -> final_tail
                query = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT ?final_tail (SAMPLE(?name_en) as ?preferred_name) (SAMPLE(?name_any) as ?fallback_name) WHERE {{
                    ns:{entity_id} ns:{normalize_relation(original_rel)} ?cvt .
                    FILTER(?cvt = ns:{cvt_node_id})
                    ?cvt ns:{normalize_relation(cvt_rel)} ?final_tail .
                    OPTIONAL {{ ?final_tail ns:type.object.name ?name_en . FILTER(LANGMATCHES(LANG(?name_en), 'en')) }}
                    OPTIONAL {{ ?final_tail ns:type.object.name ?name_any . }}
                }} GROUP BY ?final_tail LIMIT {limit_per_relation * 2}
                """
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                bindings = results.get("results", {}).get("bindings", [])
                
                for b in bindings:
                    tail_uri = b.get("final_tail", {}).get("value", "")
                    if tail_uri:
                        tail_id = tail_uri.replace("http://rdf.freebase.com/ns/", "").replace("/", ".")
                        name = b.get("preferred_name", {}).get("value", "") or b.get("fallback_name", {}).get("value", "")
                        
                        triple_data = {
                            "head": entity_name,
                            "head_id": entity_id,
                            "relation": flatten_rel,  # Return the flatten relation name
                            "tail": name if name else tail_id,
                            "tail_id": tail_id
                        }
                        triples.append(triple_data)
            else:
                # Incoming: final_head -> cvt_rel -> cvt_node -> original_rel -> entity
                query = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT ?final_head (SAMPLE(?name_en) as ?preferred_name) (SAMPLE(?name_any) as ?fallback_name) WHERE {{
                    ?final_head ns:{normalize_relation(cvt_rel)} ?cvt .
                    FILTER(?cvt = ns:{cvt_node_id})
                    ?cvt ns:{normalize_relation(original_rel)} ns:{entity_id} .
                    OPTIONAL {{ ?final_head ns:type.object.name ?name_en . FILTER(LANGMATCHES(LANG(?name_en), 'en')) }}
                    OPTIONAL {{ ?final_head ns:type.object.name ?name_any . }}
                }} GROUP BY ?final_head LIMIT {limit_per_relation * 2}
                """
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                bindings = results.get("results", {}).get("bindings", [])
                
                for b in bindings:
                    head_uri = b.get("final_head", {}).get("value", "")
                    if head_uri:
                        head_id = head_uri.replace("http://rdf.freebase.com/ns/", "").replace("/", ".")
                        name = b.get("preferred_name", {}).get("value", "") or b.get("fallback_name", {}).get("value", "")
                        
                        triple_data = {
                            "head": name if name else head_id,
                            "head_id": head_id,
                            "relation": flatten_rel,  # Return the flatten relation name
                            "tail": entity_name,
                            "tail_id": entity_id
                        }
                        triples.append(triple_data)
        except Exception as e:
            logger.error(f"Error querying flatten relation {flatten_rel} (direction={direction}): {e}")
        
        # Limit to limit_per_relation (for flatten relations, this should be 3)
        if len(triples) > limit_per_relation:
            import random
            triples = random.sample(triples, limit_per_relation)
        
        return triples
    
    def get_pending_flatten_relations(self, entity_id: Optional[str] = None) -> List[str]:
        """Get pending flatten relations found during get_triples calls.
        
        Args:
            entity_id: If provided, only return flatten relations for this entity.
                     If None, return all flatten relations from all entities.
        
        Note: Does NOT clear them - they persist for the entire question.
        Use clear_pending_flatten_relations() to clear when question is complete.
        """
        if not hasattr(self, '_pending_flatten_relations'):
            return []
        
        if entity_id is not None:
            # Return only for this specific entity
            return self._pending_flatten_relations.get(entity_id, []).copy()
        else:
            # Return all flatten relations from all entities
            all_relations = []
            for rels in self._pending_flatten_relations.values():
                all_relations.extend(rels)
            return all_relations
    
    def clear_pending_flatten_relations(self):
        """Clear pending flatten relations. Call this when a question is complete."""
        if hasattr(self, '_pending_flatten_relations'):
            self._pending_flatten_relations = {}
    
    def get_pending_flatten_candidates(self, entity_id: str) -> Optional[Dict]:
        """
        Get pending flatten candidates that need async LLM filtering.
        
        Returns:
            Dict with keys: 'candidates', 'question', 'entity_name'
            or None if no candidates pending
        """
        if not hasattr(self, '_pending_flatten_candidates'):
            return None
        return self._pending_flatten_candidates.get(entity_id)
    
    def apply_filtered_flatten_candidates(self, entity_id: str, filtered_candidates: List[Dict]):
        """
        Apply filtered flatten candidates after async LLM filtering.
        This should be called after get_pending_flatten_candidates() and external async filtering.
        
        Args:
            entity_id: The entity ID
            filtered_candidates: The filtered and ranked candidates (already top-8)
        """
        # Store filtered candidates - they will be used when triples are re-queried
        # For now, we'll update the pending flatten relations with filtered ones
        if filtered_candidates:
            filtered_rels = [c["flatten_relation"] for c in filtered_candidates[:8]]
            self._pending_flatten_relations[entity_id] = filtered_rels
    
    def clear_pending_flatten_candidates(self):
        """Clear pending flatten candidates after processing."""
        if hasattr(self, '_pending_flatten_candidates'):
            self._pending_flatten_candidates = {}

    def format_relations_for_prompt(self, relations: List[Dict[str, str]]) -> str:
        """Format relations for prompt."""
        if not relations:
            return "No relations found."
        return "\n".join(f"{rel.get('relation', '')}" for rel in relations)
