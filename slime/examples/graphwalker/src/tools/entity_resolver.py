"""Resolve entity names (e.g., "Barack Obama") to Freebase IDs (e.g., "m.02mjmr")."""
from __future__ import annotations
import logging
import os
import time
from typing import Optional, List, Dict
from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)

# Performance logger for entity resolution
_entity_perf_logger = logging.getLogger("graphwalker.entity_performance")
_entity_perf_logger.setLevel(logging.INFO)
if not _entity_perf_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[ENTITY-PERF] %(message)s'))
    _entity_perf_logger.addHandler(handler)


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

class EntityResolver:
    """Resolve entity names to Freebase IDs via SPARQL label search.

    Note: In the main KG-augmented flow we prefer a per-question name↔mid registry
    maintained elsewhere (do not reimplement here). This resolver is a fallback
    for direct SPARQL use cases.
    """
    
    def __init__(self, sparql_endpoint: str, timeout: int = 10):
        """Init resolver with endpoint and timeout; keep a small in-memory cache."""
        self.endpoint = sparql_endpoint
        # Setup no_proxy for local SPARQL endpoints
        _setup_sparql_no_proxy()
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(timeout)
        self._cache: Dict[str, Optional[str]] = {}
        logger.debug(f"✓ Entity resolver initialized: {self.endpoint}")
    
    def resolve(self, entity_name: str, limit: int = 10) -> Optional[str]:
        """Resolve a name to a Freebase ID using exact English label matches.

        Strategy: query type.object.name for exact @en matches, group by entity,
        pick the candidate having the most associated types (simple, non-hardcoded
        heuristic). Fallback to a lowercase-equality variant if needed.
        """
        # Check cache first
        cache_key = entity_name.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try type.object.name first (more reliable for entity names)
        query = f"""
        PREFIX ns: <http://rdf.freebase.com/ns/>
        
        SELECT DISTINCT ?entity ?type WHERE {{
          ?entity ns:type.object.name "{entity_name}"@en .
          ?entity ns:type.object.type ?type .
        }}
        LIMIT {limit * 10}
        """
        try:
            sparql_start = time.time()
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            sparql_time = time.time() - sparql_start
            if sparql_time > 1.0:  # Only log slow queries (>1s)
                _entity_perf_logger.info(f"Entity resolution SPARQL query for '{entity_name}': {sparql_time:.2f}s")
            
            bindings = results.get("results", {}).get("bindings", []) if results else []
            if bindings:
                entity_types: Dict[str, List[str]] = {}
                for b in bindings:
                    eid = b.get("entity", {}).get("value", "")
                    eid = eid.split("/ns/")[-1] if "/ns/" in eid else eid
                    etype = b.get("type", {}).get("value", "").split("/ns/")[-1]
                    entity_types.setdefault(eid, []).append(etype)
                # Pick entity with the most types (simple heuristic)
                best_entity = max(entity_types.items(), key=lambda kv: len(kv[1]))[0]
                self._cache[cache_key] = best_entity
                logger.debug(f"Resolved '{entity_name}' -> {best_entity} (by type count)")
                return best_entity
            
            # Fallback: lowercased equality on name
            query_alt = f"""
            PREFIX ns: <http://rdf.freebase.com/ns/>
            
            SELECT DISTINCT ?entity ?name ?type WHERE {{
              ?entity ns:type.object.name ?name .
              ?entity ns:type.object.type ?type .
              FILTER (LCASE(STR(?name)) = "{entity_name.lower()}")
            }}
            LIMIT {limit * 5}
            """
            
            sparql_start = time.time()
            self.sparql.setQuery(query_alt)
            results = self.sparql.query().convert()
            sparql_time = time.time() - sparql_start
            if sparql_time > 1.0:  # Only log slow queries (>1s)
                _entity_perf_logger.info(f"Entity resolution fallback SPARQL query for '{entity_name}': {sparql_time:.2f}s")
            bindings = results.get("results", {}).get("bindings", []) if results else []
            if bindings:
                entity_types: Dict[str, List[str]] = {}
                for b in bindings:
                    eid = b.get("entity", {}).get("value", "")
                    eid = eid.split("/ns/")[-1] if "/ns/" in eid else eid
                    etype = b.get("type", {}).get("value", "").split("/ns/")[-1]
                    entity_types.setdefault(eid, []).append(etype)
                best_entity = max(entity_types.items(), key=lambda kv: len(kv[1]))[0]
                self._cache[cache_key] = best_entity
                return best_entity
            
            self._cache[cache_key] = None
            return None
            
        except Exception as e:
            logger.error(f"Entity resolution failed for '{entity_name}': {e}")
            return None
    