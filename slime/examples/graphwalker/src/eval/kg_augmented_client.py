"""KG-Augmented Model Client

Wrap a base model client, intercept <kg-query> calls during generation, execute
them on the KG (SPARQL) backend, and feed results back to the model.
"""
from __future__ import annotations
import asyncio
import re
import logging
import json
import random
import time
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Performance logger for KG operations
_perf_logger = logging.getLogger("graphwalker.kg_performance")
_perf_logger.setLevel(logging.INFO)
if not _perf_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[KG-PERF] %(message)s'))
    _perf_logger.addHandler(handler)

# Concurrency control for KG queries
SEMAPHORE = asyncio.Semaphore(128)
_thread_pool_executor = None

def _setup_thread_pool():
    """Setup thread pool executor for asyncio.to_thread calls."""
    global _thread_pool_executor
    if _thread_pool_executor is None:
        max_workers = 128
        _thread_pool_executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            loop = asyncio.get_event_loop()
            loop.set_default_executor(_thread_pool_executor)
            _perf_logger.info(f"Thread pool executor configured with max_workers={max_workers}")
        except RuntimeError:
            # Event loop not yet created, will set later
            pass
    return _thread_pool_executor

from .model_client import BaseModelClient, ModelConfig
try:
    from kgqa_agent.prompts.prompts import build_continuation_prompt, FORCE_ANSWER_PROMPT
    from kgqa_agent.prompts.filter_prompt import __doc__ as filter_prompt_template
    from kgqa_agent.prompts.flatten_rel_filter_prompt import __doc__ as flatten_rel_filter_prompt_template
except ImportError:
    # Fallback: when running from slime training (PYTHONPATH includes graphwalker/)
    from prompts.prompts import build_continuation_prompt, FORCE_ANSWER_PROMPT
    from prompts.filter_prompt import __doc__ as filter_prompt_template
    from prompts.flatten_rel_filter_prompt import __doc__ as flatten_rel_filter_prompt_template
from ..tools.relation_normalizer import normalize_relation

logger = logging.getLogger(__name__)


class KGAugmentedModelClient(BaseModelClient):
    """Model client that supports interactive KG queries during generation."""
    
    def __init__(self, 
                 base_client: BaseModelClient,
                 kg_server_url: str = "http://localhost:18890",
                 max_calls: int = 10,
                 kg_top_k: int = 10,
                 kg_timeout: int = 10,
                 filter_client: Optional[BaseModelClient] = None,
                 dataset_type: str = "cwq",
                 full_list: bool = False):
        """Initialize client.
        Args:
            base_client: Underlying model client.
            kg_server_url: KG server base URL or SPARQL endpoint.
            max_calls: Max number of KG queries per question.
            kg_top_k: Top-K relations to show per list call.
            kg_timeout: Timeout for KG SPARQL queries in seconds.
            filter_client: Optional shared filter client for relation filtering.
            If None, creates a new one (for backward compatibility).
            dataset_type: Dataset type (cwq, webqsp, grailqa) for whitelist selection.
            full_list: If True, use merged full whitelist instead of dataset-specific one.
        """
        # Setup thread pool executor on first initialization
        _setup_thread_pool()
        
        self.base_client = base_client
        self.max_calls = max_calls
        self.kg_top_k = kg_top_k
        self._trace: List[Dict[str, Any]] = []
        self._seen_relations_set: set[str] = set()
        self._last_relations_text: str = ""
        self._last_entities_text: str = ""
        self._entity_registry: Dict[str, str] = {}
        self._initial_entity_names: List[str] = []
        # Statistics for monitoring filter fallback frequency
        self._filter_total_calls: int = 0
        self._filter_fallback_count: int = 0
        self._filter_parse_fail_count: int = 0
        # Use shared filter client if provided, otherwise create a new one
        if filter_client is not None:
            self.filter_client = filter_client
        else:
            # Initialize filter client (using default OpenAI API for filtering)
            self.filter_client = BaseModelClient(ModelConfig(
                model="deepseek-chat",  # Default filter model
                temperature=0.0,  # Deterministic filtering
                max_tokens=1024,
                stop=["\n\n"],
            ))

        from ..tools.direct_sparql_client import DirectSPARQLKGClient
        endpoint = kg_server_url if kg_server_url.endswith("/sparql") else f"{kg_server_url}/sparql"
        # Pass LLM filter callback to DirectSPARQLKGClient for CVT flatten relation filtering
        self.kg_client = DirectSPARQLKGClient(
            sparql_endpoint=endpoint,
            timeout=kg_timeout,
            llm_filter_callback=self._filter_relations_with_llm,
            dataset_type=dataset_type,
            full_list=full_list
        )
    
    def _extract_kg_query(self, text: str) -> Optional[str]:
        """Extract <kg-query>...</kg-query> content.
        
        Uses a robust regex that avoids matching "fake" tags in instructions/think blocks.
        The negative lookahead (?!<kg-query>) ensures we don't match content that contains
        another <kg-query> tag, which prevents matching when the model mentions the tag
        in its reasoning (e.g., "using the <kg-query> tag").
        """
        # Use the same robust regex as in _interactive_generate to avoid matching
        # tags mentioned in think blocks or instructions
        match = re.search(r'<kg-query>((?:(?!<kg-query>).)*?)</kg-query>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_think(self, text: str) -> Optional[str]:
        """Extract <think>...</think> content."""
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_answer_tag(self, text: str) -> Optional[str]:
        """Extract <answer>...</answer> content."""
        # Use regex that avoids matching across intermediate <answer> tags
        # and prefer the last occurrence if multiple exist.
        matches = re.findall(r'<answer>((?:(?!<answer>).)*?)</answer>', text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    # ==================== Async versions (for slime training) ====================
    # Training uses only _parse_and_execute_query_async; sync _parse_and_execute_query was removed.

    async def _parse_and_execute_query_async(self, query_str: str, question: str = "") -> tuple[bool, str]:
        """Async version of _parse_and_execute_query.
        
        Uses async LLM filter calls so the event loop is not blocked while waiting
        for the filter model response. KG server calls (SPARQL) are offloaded to a
        thread pool via asyncio.to_thread to also avoid blocking.
        
        Returns:
            tuple[bool, str]: (success, result)
        """
        if not self.kg_client:
            return (False, "[KG query service not available]")

        augmented_question = self._augment_question(question)

        def entity_error(entity: str) -> str:
            msg = (
                "Invalid entity. Use an entity returned by the previous step "
                "and copy it exactly."
            )
            if self._last_entities_text:
                msg += "\n\nLast entities:\n\n" + self._last_entities_text
            return msg

        def relation_choice_error(relation: str) -> str:
            msg = (
                f"The relation '{relation}' is not in the latest predicate list. "
                "Choose predicates from the list below:\n\n" 
                f"{self._last_relations_text}"
            )
            return msg

        # get_relations(entity)
        match = re.match(r'get_relations\s*\(\s*(["\'])(.*?)\1\s*\)', query_str.strip(), re.DOTALL)
        if match:
            entity = match.group(2).strip()
            
            entity_resolve_start = time.time()
            entity_resolved = self._resolve_and_register_entity(entity)
            entity_resolve_time = time.time() - entity_resolve_start
            if entity_resolve_time > 0.1:
                _perf_logger.info(f"Entity resolution '{entity}' -> '{entity_resolved}': {entity_resolve_time:.2f}s")
            
            if not entity_resolved:
                return (False, entity_error(entity))
            
            # Run KG query in thread pool to avoid blocking event loop
            get_relations_start = time.time()
            async with SEMAPHORE:
                relations = await asyncio.to_thread(
                    self.kg_client.get_relations,
                    entity_resolved,
                    question=augmented_question,
                    top_k=self.kg_top_k * 3,
                )
            get_relations_time = time.time() - get_relations_start
            _perf_logger.info(f"get_relations('{entity_resolved}'): {get_relations_time:.2f}s, found {len(relations)} relations")
            
            if len(relations) > 10:
                random.shuffle(relations)
                # Use async LLM filter (non-blocking)
                llm_filter_start = time.time()
                relations = await self._filter_relations_with_llm_async(relations, question, entity)
                llm_filter_time = time.time() - llm_filter_start
                _perf_logger.info(f"LLM filter async ({len(relations)} candidates): {llm_filter_time:.2f}s")

            relations = relations[:self.kg_top_k]
            
            formatted = self.kg_client.format_relations_for_prompt(relations)
            self._seen_relations_set.update({r.get("relation", "") for r in relations if r.get("relation")})
            self._last_relations_text = formatted
            self._last_entities_text = ""
            return (True, formatted)

        # get_triples(entity, [rel1, rel2, ...])
        match = re.match(r'get_triples\s*\(\s*(["\'])(.*?)\1\s*,\s*\[(.*?)\]\s*\)', query_str.strip(), re.DOTALL)
        if match:
            entity = match.group(2).strip()
            relations_str = match.group(3).strip()
            
            relations = []
            seen_relations = set()
            for r in relations_str.split(','):
                r = r.strip()
                if (r.startswith('"') and r.endswith('"')) or (r.startswith("'") and r.endswith("'")):
                    rel = r[1:-1]
                    if rel not in seen_relations:
                        relations.append(rel)
                        seen_relations.add(rel)
            
            relations = relations[:4]

            entity_resolve_start = time.time()
            entity_resolved = self._resolve_and_register_entity(entity)
            entity_resolve_time = time.time() - entity_resolve_start
            if entity_resolve_time > 0.1:
                _perf_logger.info(f"Entity resolution '{entity}' -> '{entity_resolved}': {entity_resolve_time:.2f}s")
            
            if not entity_resolved:
                return (False, entity_error(entity))
            
            if self._seen_relations_set:
                for r in relations:
                    norm_rel = normalize_relation(r)
                    if norm_rel not in self._seen_relations_set:
                        return (False, relation_choice_error(r))

            # Run get_triples in thread pool
            # Note: Now get_triples stores pending flatten candidates instead of filtering them
            get_triples_start = time.time()
            async with SEMAPHORE:
                result = await asyncio.to_thread(
                    self.kg_client.get_triples,
                    entity_resolved,
                    relations,
                    limit_per_relation=5,
                    question=augmented_question,
                    return_with_cvt_info=True,
                )
            get_triples_time = time.time() - get_triples_start
            _perf_logger.info(f"get_triples('{entity_resolved}', {len(relations)} relations): {get_triples_time:.2f}s, found {len(result.get('triples', []))} triples")
            
            triples = result["triples"]
            cvt_info = result["cvt_info"]
            
            # Check if there are pending flatten candidates that need async filtering
            pending_candidates_info = self.kg_client.get_pending_flatten_candidates(entity_resolved)
            if pending_candidates_info and pending_candidates_info.get("candidates"):
                # Apply async LLM filter to CVT flatten candidates (non-blocking!)
                candidates = pending_candidates_info["candidates"]
                question_text = pending_candidates_info["question"]
                entity_name = pending_candidates_info["entity_name"]
                
                # Prepare relations for LLM filter
                relations_for_llm = [{"relation": c["flatten_relation"]} for c in candidates]
                
                cvt_filter_start = time.time()
                try:
                    # Use async filter with use_flatten_prompt=True
                    filtered_relations = await self._filter_relations_with_llm_async(
                        relations_for_llm, question_text, entity_name, use_flatten_prompt=True
                    )
                    
                    # Build filtered candidates (top-8)
                    filtered_flatten_rels = {r["relation"] for r in filtered_relations}
                    candidate_map = {c["flatten_relation"]: c for c in candidates}
                    filtered_candidates = []
                    for r in filtered_relations[:8]:
                        rel_name = r["relation"]
                        if rel_name in candidate_map:
                            filtered_candidates.append(candidate_map[rel_name])
                    
                    # Apply filtered candidates back to kg_client
                    if filtered_candidates:
                        filtered_rels = [c["flatten_relation"] for c in filtered_candidates]
                        # Update pending flatten relations with filtered results
                        self.kg_client._pending_flatten_relations[entity_resolved] = filtered_rels
                        
                    cvt_filter_time = time.time() - cvt_filter_start
                    _perf_logger.info(f"CVT flatten LLM filter async ({len(candidates)} -> {len(filtered_candidates)}): {cvt_filter_time:.2f}s")
                except Exception as e:
                    _perf_logger.warning(f"CVT flatten async filter failed: {e}, using first 8 candidates")
                
                # Clear pending candidates
                self.kg_client.clear_pending_flatten_candidates()
            
            flatten_relations = self.kg_client.get_pending_flatten_relations(entity_resolved)
            
            if flatten_relations:
                flatten_rel_dicts = [{"relation": rel} for rel in flatten_relations]
                other_relations = [{"relation": rel} for rel in relations[1:]]
                combined_relations = other_relations + flatten_rel_dicts
                
                if augmented_question and combined_relations:
                    rank_by_similarity_start = time.time()
                    combined_relations = self.kg_client.rank_by_similarity(
                        combined_relations, augmented_question, "relation"
                    )
                    rank_by_similarity_time = time.time() - rank_by_similarity_start
                    _perf_logger.info(f"rank_by_similarity: {rank_by_similarity_time:.2f}s")
                
                # Use async LLM filter for flatten relations (non-blocking)
                if len(combined_relations) > 10:
                    random.shuffle(combined_relations)
                    combined_filter_start = time.time()
                    combined_relations = await self._filter_relations_with_llm_async(
                        combined_relations, question, entity
                    )
                    combined_filter_time = time.time() - combined_filter_start
                    _perf_logger.info(f"Combined relations LLM filter async ({len(combined_relations)} candidates): {combined_filter_time:.2f}s")
                
                for rel_dict in combined_relations:
                    rel = rel_dict.get("relation", "")
                    if rel:
                        self._seen_relations_set.add(normalize_relation(rel))
                
            new_entities = []
            for t in triples:
                new_entities.append({'name': t['head'], 'id': t['head_id']})
                new_entities.append({'name': t['tail'], 'id': t['tail_id']})
            self._register_entities(new_entities)

            formatted_triples = []
            for t in triples:
                head_name = t['head']
                tail_name = t['tail']
                formatted_triples.append(f"[{head_name}, {t['relation']}, {tail_name}]")
            
            formatted = "\n".join(formatted_triples)
            if not formatted:
                formatted = "No triples found."
            
            self._last_entities_text = formatted
            self._last_relations_text = ""
            return (True, formatted)

        return (False, f"[Could not parse query: {query_str}]")

    async def _filter_relations_with_llm_async(self, relations: List[Dict[str, str]], question: str, entity_name: str,
                                                use_flatten_prompt: bool = False) -> List[Dict[str, str]]:
        """Async version of _filter_relations_with_llm.
        
        Uses filter_client.agenerate() (AsyncOpenAI) so the event loop is not blocked
        while waiting for the filter LLM API response. This allows other coroutines
        (e.g., SGLang requests for other samples) to execute concurrently.
        """
        if not relations:
            return []
        
        self._filter_total_calls += 1
        
        rel_strs = [r['relation'] for r in relations]
        
        prompt_template = (
            flatten_rel_filter_prompt_template if use_flatten_prompt 
            else filter_prompt_template
        )
        
        prompt = (
            f"{prompt_template}\n"
            f"Question: {question}\n"
            f"Topic Entity: [\"{entity_name}\"]\n"
            f"Relations: {json.dumps(rel_strs)}\n\n"
            f"Your Selections: "
        )
        
        MAX_RETRIES = 3
        MAX_BACKOFF_SECONDS = 8.0
        last_exception: Optional[Exception] = None
        
        for attempt in range(MAX_RETRIES):
            try:
                # Async LLM call: does NOT block the event loop
                response = await self.filter_client.agenerate(prompt)
                
                json_list_match = re.search(r'\[.*?\]', response, re.DOTALL)
                
                if json_list_match:
                    try:
                        selected_rels = json.loads(json_list_match.group(0))
                        
                        if not isinstance(selected_rels, list):
                            raise ValueError(f"Expected list, got {type(selected_rels).__name__}")
                        
                        rel_map = {r['relation']: r for r in relations}
                        ordered_filtered = []
                        
                        for selected_rel in selected_rels:
                            if isinstance(selected_rel, str) and selected_rel in rel_map:
                                ordered_filtered.append(rel_map[selected_rel])
                        
                        if ordered_filtered:
                            return ordered_filtered
                        
                    except (json.JSONDecodeError, ValueError, TypeError) as parse_error:
                        logger.debug(
                            f"JSON parse error in async filter response (attempt {attempt + 1}/{MAX_RETRIES}): "
                            f"{type(parse_error).__name__}: {parse_error}"
                        )
                        self._filter_parse_fail_count += 1
                        break
                else:
                    logger.debug(
                        f"No JSON list found in async filter response (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    self._filter_parse_fail_count += 1
                    break
                    
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                error_str = str(e).lower()
                
                is_retryable = (
                    'timeout' in error_type.lower() or 'timeout' in error_str or
                    'ratelimit' in error_type.lower() or 'rate limit' in error_str or
                    'apiconnection' in error_type.lower() or 'connection' in error_str or
                    'apierror' in error_type.lower()
                )
                
                if is_retryable and attempt < MAX_RETRIES - 1:
                    backoff_time = min(2 ** (attempt + 1), MAX_BACKOFF_SECONDS)
                    logger.debug(
                        f"Async filter LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): "
                        f"{error_type}: {e}. Retrying in {backoff_time:.1f}s..."
                    )
                    await asyncio.sleep(backoff_time)  # Non-blocking sleep
                    continue
                else:
                    logger.debug(
                        f"Async filter LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): "
                        f"{error_type}: {e}. Using fallback."
                    )
                    break
        
        self._filter_fallback_count += 1
        if last_exception:
            logger.debug(
                f"Async filter fallback for entity '{entity_name}': "
                f"{type(last_exception).__name__}: {last_exception}"
            )
        
        return relations[:self.kg_top_k]

    # ==================== Helper methods for entity/question handling ====================

    def _seed_registry_from_topic_entities(self, topic_entities: Dict[str, str]) -> None:
        """Seed registry with provided mapping of id->name or name->id."""
        if not topic_entities:
            return
        def looks_like_mid(s: str) -> bool:
            return isinstance(s, str) and s.startswith(("m.", "g.", "en."))
        for k, v in topic_entities.items():
            if looks_like_mid(k):
                mid, name = k, v
            elif looks_like_mid(v):
                mid, name = v, k
            else:
                # Unknown direction; skip explicit mapping
                mid, name = None, None
            if mid and name:
                self._entity_registry[name.lower()] = mid
                self._entity_registry[mid] = name

    def _extract_initial_entity_names(self, topic_entities: Dict[str, str]) -> List[str]:
        names: List[str] = []
        def looks_like_mid(s: str) -> bool:
            return isinstance(s, str) and s.startswith(("m.", "g.", "en."))
        for k, v in (topic_entities or {}).items():
            if looks_like_mid(k) and isinstance(v, str) and v:
                names.append(v.strip())
            elif looks_like_mid(v) and isinstance(k, str) and k:
                names.append(k.strip())
        # Deduplicate preserving order
        seen = set()
        out: List[str] = []
        for n in names:
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _augment_question(self, question: str) -> str:
        if not self._initial_entity_names:
            return question or ""
        return (question + " " + " ".join(self._initial_entity_names)).strip()

    def _register_entities(self, entities: List[Dict[str, Any]]) -> None:
        if not entities:
            return
        for ent in entities:
            mid = ent.get('entity') or ent.get('id')
            name = ent.get('name') or ent.get('label')
            if isinstance(mid, str) and mid:
                if isinstance(name, str) and name:
                    self._entity_registry[name.lower()] = mid
                    self._entity_registry[mid] = name
                else:
                    self._entity_registry[mid] = mid

    def _resolve_entity_for_query(self, entity: str) -> Optional[str]:
        if not entity:
            return None
        if entity.startswith(('m.', 'g.', 'en.')):
            return entity
        return self._entity_registry.get(entity.lower())
    
    def _resolve_and_register_entity(self, entity: str) -> Optional[str]:
        """解析实体并注册到实体注册表。如果registry中没有，通过kg_client解析。"""
        entity_resolved = self._resolve_entity_for_query(entity)
        if not entity_resolved and self.kg_client:
            entity_resolved = self.kg_client._resolve_entity(entity)
            if entity_resolved:
                self._entity_registry[entity.lower()] = entity_resolved
                self._entity_registry[entity_resolved] = entity
        return entity_resolved

    # ==================== Synchronous filter (kept for backward compatibility) ====================
    # Note: Training code uses _filter_relations_with_llm_async instead.
    # This sync version is kept for DirectSPARQLKGClient callback and potential eval/debug usage.

    def _filter_relations_with_llm(self, relations: List[Dict[str, str]], question: str, entity_name: str, 
                                   use_flatten_prompt: bool = False) -> List[Dict[str, str]]:
        """Filter relations using a secondary LLM with retry mechanism.
        
        This method uses an LLM to filter and rank relations based on their relevance
        to the question. It includes automatic retry logic for transient errors.
        
        Args:
            relations: List of relation dicts to filter (each dict has a 'relation' key)
            question: Question text for context
            entity_name: Entity name for context
            use_flatten_prompt: If True, use flatten_rel_filter_prompt_template for CVT flatten relations;
                              If False, use filter_prompt_template for regular relations
        
        Returns:
            Filtered and reordered list of relation dicts (up to kg_top_k relations)
            Falls back to top-k from original list if filtering fails
        """
        if not relations:
            return []
        
        # Track total filter calls for statistics
        self._filter_total_calls += 1
        
        # Extract relation strings for the prompt
        rel_strs = [r['relation'] for r in relations]
        
        # Select appropriate prompt template
        prompt_template = (
            flatten_rel_filter_prompt_template if use_flatten_prompt 
            else filter_prompt_template
        )
        
        # Build the prompt
        prompt = (
            f"{prompt_template}\n"
            f"Question: {question}\n"
            f"Topic Entity: [\"{entity_name}\"]\n"
            f"Relations: {json.dumps(rel_strs)}\n\n"
            f"Your Selections: "
        )
        
        # Retry configuration
        MAX_RETRIES = 3
        MAX_BACKOFF_SECONDS = 8.0
        last_exception: Optional[Exception] = None
        
        for attempt in range(MAX_RETRIES):
            try:
                # Call the filter LLM
                response = self.filter_client.generate(prompt)
                
                # Parse response: expect a JSON list of relation strings
                # Look for JSON list pattern in the response
                json_list_match = re.search(r'\[.*?\]', response, re.DOTALL)
                
                if json_list_match:
                    try:
                        # Parse the JSON list
                        selected_rels = json.loads(json_list_match.group(0))
                        
                        # Validate that selected_rels is a list
                        if not isinstance(selected_rels, list):
                            raise ValueError(f"Expected list, got {type(selected_rels).__name__}")
                        
                        # Filter and reorder original relations based on LLM selection
                        rel_map = {r['relation']: r for r in relations}
                        ordered_filtered = []
                        
                        for selected_rel in selected_rels:
                            if isinstance(selected_rel, str) and selected_rel in rel_map:
                                ordered_filtered.append(rel_map[selected_rel])
                        
                        # Success: return filtered relations if we found any matches
                        if ordered_filtered:
                            return ordered_filtered
                        
                        # If no matches found, continue to fallback
                        # (This can happen if LLM returns relations not in the original list)
                        
                    except (json.JSONDecodeError, ValueError, TypeError) as parse_error:
                        # JSON parsing failed - don't retry on parse errors
                        logger.debug(
                            f"JSON parse error in filter response (attempt {attempt + 1}/{MAX_RETRIES}): "
                            f"{type(parse_error).__name__}: {parse_error}"
                        )
                        self._filter_parse_fail_count += 1
                        break
                else:
                    # No JSON list found in response - don't retry on format errors
                    logger.debug(
                        f"No JSON list found in filter response (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    self._filter_parse_fail_count += 1
                    break
                    
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                error_str = str(e).lower()
                
                # Determine if this error is retryable
                # Retry on: timeouts, rate limits, connection errors, and general API errors
                is_retryable = (
                    'timeout' in error_type.lower() or 'timeout' in error_str or
                    'ratelimit' in error_type.lower() or 'rate limit' in error_str or
                    'apiconnection' in error_type.lower() or 'connection' in error_str or
                    'apierror' in error_type.lower()
                )
                
                if is_retryable and attempt < MAX_RETRIES - 1:
                    # Exponential backoff: 1s, 2s, 4s (capped at MAX_BACKOFF_SECONDS)
                    backoff_time = min(2 ** (attempt + 1), MAX_BACKOFF_SECONDS)
                    logger.debug(
                        f"Filter LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): "
                        f"{error_type}: {e}. Retrying in {backoff_time:.1f}s..."
                    )
                    time.sleep(backoff_time)
                    continue
                else:
                    # Non-retryable error or max retries reached
                    logger.debug(
                        f"Filter LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): "
                        f"{error_type}: {e}. Using fallback."
                    )
                    break
        
        # Fallback: return top-k relations from original list (no filtering)
        self._filter_fallback_count += 1
        if last_exception:
            logger.debug(
                f"Filter fallback for entity '{entity_name}': "
                f"{type(last_exception).__name__}: {last_exception}"
            )
        
        return relations[:self.kg_top_k]
