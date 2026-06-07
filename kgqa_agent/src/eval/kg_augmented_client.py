"""KG-Augmented Model Client

Wrap a base model client, intercept <kg-query> calls during generation, execute
them on the KG (SPARQL) backend, and feed results back to the model.
"""
from __future__ import annotations
import re
import logging
import json
import random
import time
from typing import Optional, List, Dict, Any

from .model_client import BaseModelClient, ModelConfig
from kgqa_agent.prompts.prompts import build_continuation_prompt, FORCE_ANSWER_PROMPT
from kgqa_agent.prompts.filter_prompt import __doc__ as filter_prompt_template
from kgqa_agent.prompts.flatten_rel_filter_prompt import __doc__ as flatten_rel_filter_prompt_template
from ..tools.relation_normalizer import normalize_relation

logger = logging.getLogger(__name__)


class KGAugmentedModelClient(BaseModelClient):
    """Model client that supports interactive KG queries during generation."""

    @staticmethod
    def _looks_like_mid(s: str) -> bool:
        """Check if string looks like a Freebase ID."""
        return isinstance(s, str) and s.startswith(("m.", "g.", "en."))

    def __init__(self,
                 base_client: BaseModelClient,
                 kg_server_url: str = "http://localhost:18890",
                 max_calls: int = 10,
                 kg_top_k: int = 10,
                 filter_client: Optional[BaseModelClient] = None,
                 dataset_type: str = "cwq",
                 full_list: bool = False):
        """Initialize client.
        Args:
            base_client: Underlying model client.
            kg_server_url: KG server base URL or SPARQL endpoint.
            max_calls: Max number of KG queries per question.
            kg_top_k: Top-K relations to show per list call.
            filter_client: Optional shared filter client for relation filtering.
            If None, creates a new one (for backward compatibility).
            dataset_type: Dataset type (cwq, webqsp, grailqa) for whitelist selection.
            full_list: If True, use merged full whitelist instead of dataset-specific one.
        """
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
                model="Qwen2.5-7B-Instruct",  # Default filter model
                temperature=0.0,  # Deterministic filtering
                max_tokens=1024,
                stop=["\n\n"],
            ))

        from ..tools.direct_sparql_client import DirectSPARQLKGClient
        endpoint = kg_server_url if kg_server_url.endswith("/sparql") else f"{kg_server_url}/sparql"
        # Pass LLM filter callback to DirectSPARQLKGClient for CVT flatten relation filtering
        # Use increased timeout (120s) to reduce timeout-related exceptions
        self.kg_client = DirectSPARQLKGClient(
            sparql_endpoint=endpoint,
            timeout=120,
            llm_filter_callback=self._filter_relations_with_llm,
            dataset_type=dataset_type,
            full_list=full_list
        )
    
    def _extract_kg_query(self, text: str) -> Optional[str]:
        """Extract <kg-query>...</kg-query> content."""
        match = re.search(r'<kg-query>(.*?)</kg-query>', text, re.DOTALL)
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

    def _parse_and_execute_query(self, query_str: str, question: str = "") -> str:
        """Parse and execute a KG query and return formatted results or an error."""
        if not self.kg_client:
            return "[KG query service not available]"

        augmented_question = self._augment_question(question)

        def entity_error(entity: str) -> str:
            msg = (
                "Invalid entity. Use an entity returned by the previous step "
                "and copy it exactly."
            )
            if self._last_entities_text:
                msg += "\n\nLast entities (names only):\n\n" + self._last_entities_text
            self._trace.append({"type": "error", "content": msg, "entity": entity})
            return msg

        def relation_choice_error(relation: str) -> str:
            msg = (
                f"The relation '{relation}' is not in the latest predicate list. "
                "Choose predicates from the list below:\n\n" 
                f"{self._last_relations_text}"
            )
            self._trace.append({"type": "error", "content": msg, "relation": relation})
            return msg

        # get_relations(entity)
        match = re.match(r'get_relations\s*\(\s*(["\'])(.*?)\1\s*\)', query_str.strip(), re.DOTALL)
        if match:
            entity = match.group(2).strip()
            entity_resolved = self._resolve_and_register_entity(entity)
            if not entity_resolved:
                return entity_error(entity)
            
            relations = self.kg_client.get_relations(
                entity_resolved,
                question=augmented_question,
                top_k=self.kg_top_k * 3,
            )
            
            # If there are many candidates, use LLM to re-rank/filter them.
            # Otherwise return the BM25 candidates directly to avoid extra calls.
            if len(relations) > 10:
                # Shuffle to avoid position bias before filtering
                random.shuffle(relations)
                # Filter with LLM
                relations = self._filter_relations_with_llm(relations, question, entity)

            # Limit to top-k for returning to the model
            relations = relations[:self.kg_top_k]
            
            formatted = self.kg_client.format_relations_for_prompt(relations)
            self._seen_relations_set.update({r.get("relation", "") for r in relations if r.get("relation")})
            self._last_relations_text = formatted
            self._last_entities_text = ""
            self._trace.append({
                "type": "tool_call",
                "tool": "get_relations",
                "args": {"entity": entity, "entity_resolved": entity_resolved},
                "result_count": len(relations),
                "tool_output": formatted,
            })
            return formatted

        # get_triples(entity, [rel1, rel2, ...])
        match = re.match(r'get_triples\s*\(\s*(["\'])(.*?)\1\s*,\s*\[(.*?)\]\s*\)', query_str.strip(), re.DOTALL)
        if match:
            entity = match.group(2).strip()
            relations_str = match.group(3).strip()
            
            # Parse relations list
            relations = []
            seen_relations = set()  # Track seen relations to avoid duplicates
            for r in relations_str.split(','):
                r = r.strip()
                if (r.startswith('"') and r.endswith('"')) or (r.startswith("'") and r.endswith("'")):
                    rel = r[1:-1]
                    # Deduplicate: only add if not seen before
                    if rel not in seen_relations:
                        relations.append(rel)
                        seen_relations.add(rel)
            
            # Truncate to top 4 relations as per instruction
            relations = relations[:4]

            entity_resolved = self._resolve_and_register_entity(entity)
            if not entity_resolved:
                return entity_error(entity)
            
            # Validate relations
            if self._seen_relations_set:
                for r in relations:
                    norm_rel = normalize_relation(r)
                    if norm_rel not in self._seen_relations_set:
                        return relation_choice_error(r)

            result = self.kg_client.get_triples(
                entity_resolved,
                relations,
                limit_per_relation=5,
                question=augmented_question,
                return_with_cvt_info=True  # Request CVT information for trace recording
            )
            
            # Extract triples and CVT information from result
            triples = result["triples"]
            cvt_info = result["cvt_info"]
            
            # Check for pending flatten relations discovered during get_triples for this entity
            flatten_relations = self.kg_client.get_pending_flatten_relations(entity_resolved)
            
            # If we have flatten relations, format them as relations list for BM25 ranking and LLM filter
            if flatten_relations:
                # Create relation dicts for flatten relations
                flatten_rel_dicts = [{"relation": rel} for rel in flatten_relations]
                # Combine with original relations (excluding the one that caused CVT)
                other_relations = [{"relation": rel} for rel in relations[1:]]  # Skip first relation that caused CVT
                combined_relations = other_relations + flatten_rel_dicts
                
                # Apply BM25 ranking if question provided
                if augmented_question and combined_relations:
                    combined_relations = self.kg_client.rank_by_similarity(
                        combined_relations, augmented_question, "relation"
                    )
                
                # Filter with LLM if there are many candidates
                if len(combined_relations) > 10:
                    random.shuffle(combined_relations)
                    combined_relations = self._filter_relations_with_llm(
                        combined_relations, question, entity
                    )
                
                # Update seen relations set
                for rel_dict in combined_relations:
                    rel = rel_dict.get("relation", "")
                    if rel:
                        self._seen_relations_set.add(normalize_relation(rel))
                
            
            # Register new entities found in triples
            new_entities = []
            for t in triples:
                # t is now a dict: {'head': name, 'head_id': id, 'relation':..., 'tail': name, 'tail_id': id}
                new_entities.append({'name': t['head'], 'id': t['head_id']})
                new_entities.append({'name': t['tail'], 'id': t['tail_id']})
            self._register_entities(new_entities)

            # Format triples
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
            
            # Build trace entry
            trace_entry = {
                "type": "tool_call",
                "tool": "get_triples",
                "args": {"entity": entity, "entity_resolved": entity_resolved, "relations": relations},
                "result_count": len(triples),
                "tool_output": formatted,
            }
            
            # Add CVT information if any CVT nodes were found
            if cvt_info:
                trace_entry["cvt_info"] = []
                for cvt in cvt_info:
                    # Format flattened triples for display
                    flattened_triples_formatted = []
                    for ft in cvt["flattened_triples"]:
                        flattened_triples_formatted.append({
                            "head": ft["head"],
                            "head_id": ft["head_id"],
                            "relation": ft["relation"],
                            "tail": ft["tail"],
                            "tail_id": ft["tail_id"]
                        })
                    
                    trace_entry["cvt_info"].append({
                        "original_relation": cvt["original_relation"],
                        "cvt_node_id": cvt["cvt_node_id"],
                        "cvt_relation": cvt["cvt_relation"],
                        "flatten_relation": cvt["flatten_relation"],
                        "direction": cvt["direction"],
                        "flattened_triples": flattened_triples_formatted
                    })
            
            self._trace.append(trace_entry)
            return formatted

        return f"[Could not parse query: {query_str}]"
    
    def _interactive_generate(self, 
                             initial_prompt: str,
                             system: Optional[str] = None,
                             question: str = "",
                             topic_entities: Optional[Dict[str, str]] = None,
                             **gen_kwargs) -> str:
        """Interactive generation with KG support using true multi-turn conversation format."""
        # Reset per-sample trace and entity registry
        self._trace = []
        self._entity_registry = {}
        # Reset last-round state
        self._seen_relations_set = set()
        self._last_relations_text = ""
        self._last_entities_text = ""
        # Clear pending flatten relations from previous question
        if self.kg_client:
            self.kg_client.clear_pending_flatten_relations()
        # Seed registry (prefer provided topic_entities with explicit IDs)
        if topic_entities:
            self._seed_registry_from_topic_entities(topic_entities)
            self._initial_entity_names = self._extract_initial_entity_names(topic_entities)
        else:
            self._initial_entity_names = []
        
        # Initialize messages list for true multi-turn conversation
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": initial_prompt})
        
        calls = 0

        # Record only one think/kg-query/answer per round:
        # - If base model outputs multiple tags, parser takes first match only
            # Loop ensures: prompt -> response(with first think/kg-query/answer) -> next round or end
        
        # Force stop token to prevent model from self-answering
        # Assumes base_client supports stop parameter (vLLMModelClient does)
        if "stop" not in gen_kwargs:
            gen_kwargs["stop"] = []
        if "</kg-query>" not in gen_kwargs["stop"]:
            gen_kwargs["stop"].append("</kg-query>")

        while calls < self.max_calls:
            # Log current prompt (for trace)
            current_prompt_text = messages[-1]["content"] if messages else initial_prompt
            self._trace.append({
                "type": "prompt",
                "tag": "prompt",
                "content": current_prompt_text,
            })

            # Generate response using messages list (true multi-turn format)
            response = self.base_client.generate_from_messages(messages, **gen_kwargs)
            
            # Fix truncated </kg-query> tag caused by stop token
            if "</kg-query>" in gen_kwargs["stop"] and "<kg-query>" in response and "</kg-query>" not in response:
                response += "</kg-query>"

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": response})

            # 1) Record only first think content in this round<think> 内容
            think_content = self._extract_think(response)
            if think_content:
                self._trace.append({
                    "type": "think",
                    "tag": "think",
                    "content": think_content,
                })

            # Check positions of query and answer to decide which one to process
            # We use re.search directly to get match objects for position comparison
            # Use robust regex to avoid matching "fake" tags in instructions/think blocks
            kg_query_match = re.search(r'<kg-query>((?:(?!<kg-query>).)*?)</kg-query>', response, re.DOTALL)
            answer_match = re.search(r'<answer>((?:(?!<answer>).)*?)</answer>', response, re.DOTALL)
            
            kg_query_start = kg_query_match.start() if kg_query_match else float('inf')
            answer_start = answer_match.start() if answer_match else float('inf')

            # 2) Process tag that comes first
            if kg_query_match and kg_query_start < answer_start:
                # Found a query before any answer
                kg_query = kg_query_match.group(1).strip()
                
                response_truncated = response[:kg_query_match.end()]
                if messages and messages[-1]["role"] == "assistant":
                    messages[-1]["content"] = response_truncated
                else:
                    messages.append({"role": "assistant", "content": response_truncated})
                
                calls += 1
                self._trace.append({
                    "type": "kg_query",
                    "tag": "kg-query",
                    "content": kg_query,
                    "call_number": calls,
                })

                query_results = self._parse_and_execute_query(kg_query, question=question)

                self._trace.append({
                    "type": "information",
                    "tag": "information",
                    "content": query_results,
                    "call_number": calls,
                })

                continuation = build_continuation_prompt(query_results)
                # Add new user message with continuation (true multi-turn format)
                messages.append({"role": "user", "content": continuation})
            
            elif answer_match:
                # Found answer (and no query before it)
                answer_content = answer_match.group(1).strip()
                
                # Truncate response to keep only up to the answer tag to avoid hallucinated content
                response_truncated = response[:answer_match.end()]
                # Update the last assistant message with truncated response
                if messages and messages[-1]["role"] == "assistant":
                    messages[-1]["content"] = response_truncated
                
                self._trace.append({
                    "type": "answer",
                    "tag": "answer",
                    "content": answer_content,
                })
                return response_truncated
            
            else:
                # Neither answer nor kg-query found - format error.
                # Give model chance to correct format and output query or answer.
                retry_count = 0
                max_retries = 3
                error_msg = "Your response did not contain a valid <kg-query> or <answer> tag. Please continue by outputting a valid <kg-query> or <answer>."
                
                # Use messages list for retries (true multi-turn format)
                retry_messages = messages.copy()
                final_response = response
                valid_action_taken = False

                while retry_count < max_retries:
                    # Add error message as user message
                    retry_messages.append({"role": "user", "content": error_msg})
                    
                    self._trace.append({
                        "type": "prompt",
                        "tag": "prompt",
                        "content": error_msg,
                        "retry": retry_count + 1
                    })

                    final_response = self.base_client.generate_from_messages(retry_messages, **gen_kwargs)
                    retry_messages.append({"role": "assistant", "content": final_response})

                    # 1. Check for Answer
                    answer_content = self._extract_answer_tag(final_response)
                    if answer_content:
                        answer_match_retry = re.search(r'<answer>((?:(?!<answer>).)*?)</answer>', final_response, re.DOTALL)
                        final_response_truncated = final_response[:answer_match_retry.end()] if answer_match_retry else final_response
                        if retry_messages and retry_messages[-1]["role"] == "assistant":
                            retry_messages[-1]["content"] = final_response_truncated
                        self._trace.append({
                            "type": "answer",
                            "tag": "answer",
                            "content": answer_content,
                        })
                        return final_response_truncated
                    
                    # 2. Check for Query
                    kg_query = self._extract_kg_query(final_response)
                    if kg_query:
                        calls += 1
                        self._trace.append({
                            "type": "kg_query",
                            "tag": "kg-query",
                            "content": kg_query,
                            "call_number": calls,
                        })

                        query_results = self._parse_and_execute_query(kg_query, question=question)

                        self._trace.append({
                            "type": "information",
                            "tag": "information",
                            "content": query_results,
                            "call_number": calls,
                        })

                        continuation = build_continuation_prompt(query_results)
                        messages = [msg for msg in retry_messages if not (msg.get("role") == "user" and error_msg in msg.get("content", ""))]
                        if messages and messages[-1]["role"] == "assistant":
                            kg_query_match_retry = re.search(r'<kg-query>((?:(?!<kg-query>).)*?)</kg-query>', messages[-1].get("content", ""), re.DOTALL)
                            if kg_query_match_retry:
                                messages[-1]["content"] = messages[-1]["content"][:kg_query_match_retry.end()]
                        messages.append({"role": "user", "content": continuation})
                        valid_action_taken = True
                        break

                    retry_count += 1

                if valid_action_taken:
                    continue

                answer_match_final = re.search(r'<answer>((?:(?!<answer>).)*?)</answer>', final_response, re.DOTALL)
                if answer_match_final:
                    final_response = final_response[:answer_match_final.end()]
                self._trace.append({
                    "type": "final_response",
                    "content": final_response,
                })
                return final_response

        # After max_calls: trigger forced answer logic one more time
        self._trace.append({
            "type": "max_calls_reached",
            "limit": self.max_calls,
        })

        if messages and len(messages) >= 2 and messages[-1]["role"] == "assistant":
            last_response = messages[-1]["content"]
            answer_content = self._extract_answer_tag(last_response)
            if answer_content:
                answer_match_last = re.search(r'<answer>((?:(?!<answer>).)*?)</answer>', last_response, re.DOTALL)
                last_response_truncated = last_response[:answer_match_last.end()] if answer_match_last else last_response
                messages[-1]["content"] = last_response_truncated
                self._trace.append({
                    "type": "answer",
                    "tag": "answer",
                    "content": answer_content,
                })
                return last_response_truncated

        # Otherwise, use FORCE_ANSWER_PROMPT for one more round (true multi-turn format)
        if messages:
            messages.append({"role": "user", "content": FORCE_ANSWER_PROMPT})
            self._trace.append({
                "type": "prompt",
                "tag": "prompt",
                "content": FORCE_ANSWER_PROMPT,
            })

            final_response = self.base_client.generate_from_messages(messages, **gen_kwargs)
            think_content = self._extract_think(final_response)
            if think_content:
                self._trace.append({
                    "type": "think",
                    "tag": "think",
                    "content": think_content,
                })

            answer_content = self._extract_answer_tag(final_response)
            answer_match_force = re.search(r'<answer>((?:(?!<answer>).)*?)</answer>', final_response, re.DOTALL)
            final_response_truncated = final_response[:answer_match_force.end()] if answer_match_force else final_response
            messages.append({"role": "assistant", "content": final_response_truncated})
            
            if answer_content:
                self._trace.append({
                    "type": "answer",
                    "tag": "answer",
                    "content": answer_content,
                })
            else:
                self._trace.append({
                    "type": "final_response",
                    "content": final_response_truncated,
                })

            return final_response_truncated

        return "[No response]"


    def _seed_registry_from_topic_entities(self, topic_entities: Dict[str, str]) -> None:
        """Seed registry with provided mapping of id->name or name->id."""
        if not topic_entities:
            return
        for k, v in topic_entities.items():
            if self._looks_like_mid(k):
                mid, name = k, v
            elif self._looks_like_mid(v):
                mid, name = v, k
            else:
                # Unknown direction; skip explicit mapping
                mid, name = None, None
            if mid and name:
                self._entity_registry[name.lower()] = mid
                self._entity_registry[mid] = name

    def _extract_initial_entity_names(self, topic_entities: Dict[str, str]) -> List[str]:
        names: List[str] = []
        for k, v in (topic_entities or {}).items():
            if self._looks_like_mid(k) and isinstance(v, str) and v:
                names.append(v.strip())
            elif self._looks_like_mid(v) and isinstance(k, str) and k:
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
        if self._looks_like_mid(entity):
            return entity
        return self._entity_registry.get(entity.lower())
    
    def _resolve_and_register_entity(self, entity: str) -> Optional[str]:
        """Resolve entity and register to entity registry. Resolve via kg_client if not in registry."""
        entity_resolved = self._resolve_entity_for_query(entity)
        if not entity_resolved and self.kg_client:
            entity_resolved = self.kg_client._resolve_entity(entity)
            if entity_resolved:
                self._entity_registry[entity.lower()] = entity_resolved
                self._entity_registry[entity_resolved] = entity
        return entity_resolved
    
    def generate(self, prompt: str, *, system: Optional[str] = None, question: str = "", topic_entities: Optional[Dict[str, str]] = None, **gen_kwargs) -> str:
        """Generate response with optional KG query support.
        
        Args:
            prompt: User prompt
            system: System prompt
            question: Original question (for BM25 ranking)
            **gen_kwargs: Generation kwargs
        
        Returns:
            Model output
        """
        if self.kg_client:
            return self._interactive_generate(prompt, system=system, question=question, topic_entities=topic_entities, **gen_kwargs)
        else:
            return self.base_client.generate(prompt, system=system, **gen_kwargs)
    
    def extract_answer(self, response: str) -> str:
        """Extract final answer from model response.
        
        Args:
            response: Full model response
        
        Returns:
            Extracted answer (cleaned of reasoning tags)
        """
        # Try to extract from <answer>...</answer> tags using robust internal method
        extracted = self._extract_answer_tag(response)
        if extracted:
            return extracted
        
        cleaned = response
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<kg-query>.*?</kg-query>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<information>.*?</information>', '', cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()
        return cleaned if cleaned else response.strip()

    def get_trace_and_reset(self) -> List[Dict[str, Any]]:
        trace = list(self._trace)
        self._trace = []
        # Clear pending flatten relations when question is complete
        if self.kg_client:
            self.kg_client.clear_pending_flatten_relations()
        return trace
    
    def get_filter_stats_and_reset(self) -> Dict[str, int]:
        """Get filter statistics and reset counters.
        
        Returns:
            Dictionary with filter statistics:
            - total_calls: Total number of filter calls
            - fallback_count: Number of times fallback was used
            - parse_fail_count: Number of times parsing failed
            - success_count: Number of successful filter calls
        """
        stats = {
            "total_calls": self._filter_total_calls,
            "fallback_count": self._filter_fallback_count,
            "parse_fail_count": self._filter_parse_fail_count,
            "success_count": self._filter_total_calls - self._filter_fallback_count
        }
        # Reset counters
        self._filter_total_calls = 0
        self._filter_fallback_count = 0
        self._filter_parse_fail_count = 0
        return stats

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
