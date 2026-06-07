from __future__ import annotations

from typing import Any, Dict, List, Tuple, Set
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..utils.client import LLMClient
from kgqa_agent.prompts.traj_gen_prompt import TRAJECTORY_GENERATION_PROMPT
from kgqa_agent.prompts.conj_traj_gen_prompt import CONJUNCTION_TRAJECTORY_GENERATION_PROMPT
from kgqa_agent.prompts.prompts import build_search_prompt
from tqdm import tqdm


def _build_action_for_step(steps: List[Dict[str, Any]], step: Dict[str, Any]) -> str:
    """Construct a deterministic <kg-query> string using entity NAMES and relation only.

    Rules:
    - get_*_relations(entity_name)
    - get_triples(entity_name, [relation1, relation2, ...])
    No ids, no extra args (e.g., limit) should appear.
    """
    qtype = str(step.get("query_type") or "")
    cur = step.get("current")
    args = step.get("args") or {}
    idx = int(step.get("step_index") or 0)

    def quote(s: str) -> str:
        s2 = s.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'"{s2}"'

    if "relations" in qtype:
        # current is expected to be the entity name
        entity_name = str(cur or "")
        return f"<kg-query>{qtype}({quote(entity_name)})</kg-query>"

    if "triples" in qtype:
        # get_triples(entity_name, [relation1, relation2, ...])
        entity_name = str(cur or "")
        relations = args.get("relations") or []
        if isinstance(relations, list) and len(relations) > 0:
            relations_str = ", ".join([quote(str(r)) for r in relations])
            return f"<kg-query>{qtype}({quote(entity_name)}, [{relations_str}])</kg-query>"
        else:
            # No relations available - return empty list (do not fallback to avoid data pollution)
            return f"<kg-query>{qtype}({quote(entity_name)}, [])</kg-query>"

    if "entities" in qtype:
        # Need entity_name from the immediately previous step (relations step)
        entity_name = ""
        # Only use the immediately previous step (step_index == idx-1)
        # Do not fallback to nearest earlier step to avoid data pollution
        prev = None
        for s in steps:
            if int(s.get("step_index") or -10) == idx - 1:
                prev = s
                break
        if prev is not None:
            entity_name = str(prev.get("current") or "")
        # relation name: only use args["relation"], do not fallback to current
        relation_name = str(args.get("relation") or "")
        return f"<kg-query>{qtype}({quote(entity_name)}, {quote(relation_name)})</kg-query>"

    # Unknown type: fall back to compact args (should be rare)
    return f"<kg-query>{qtype}()</kg-query>"


def _obs_values(step: Dict[str, Any]) -> List[str]:
    """Return observation results as strings without truncation.

    Upstream data guarantees at most 5 items per step; no need to slice here.
    """
    vals = step.get("results") or []
    return [str(v) for v in vals]


def _path_steps_block(path: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    steps = path.get("steps", [])
    for s in steps:
        results = s.get("results") or []
        if not results:
            continue
        blocks.append({
            "step_index": int(s.get("step_index") or 0),
            "query_type": s.get("query_type") or "",
            "action": _build_action_for_step(steps, s),
            "observation": _obs_values(s),
        })
    # Ensure sorted by step_index
    blocks.sort(key=lambda x: x["step_index"]) 
    return blocks


def _is_conjunction_path(path: Dict[str, Any]) -> bool:
    """Check if a path is a conjunction type (has name_path1 and name_path2)."""
    return 'name_path1' in path and 'name_path2' in path


def _build_path_input(path: Dict[str, Any]) -> Dict[str, Any] | None:
    """Build input for trajectory generation prompt.
    
    Includes question, name_path (or name_path1/name_path2 for conjunction), 
    topic_entity/topic_entities, answers, and steps.
    """
    is_conj = _is_conjunction_path(path)
    
    # Get question: prefer new_question, fallback to question, skip if neither exists
    question = path.get("question")
    if not question:
        # Return None to indicate this path should be skipped
        return None
    
    result = {
        "question": question,
        "steps": _path_steps_block(path),
    }
    
    if is_conj:
        # For conjunction paths, include name_path1 and name_path2
        result["name_path1"] = path.get("name_path1") or ""
        result["name_path2"] = path.get("name_path2") or ""
        # For conjunction paths, use topic_entities (list of entities)
        topic_entities = path.get("topic_entities")
        if topic_entities:
            # Convert to dict format if needed
            if isinstance(topic_entities, list):
                topic_entities_dict = {}
                for te in topic_entities:
                    if isinstance(te, dict):
                        # Format: {"uri": "...", "name": "..."} or {"entity_id": "entity_name"}
                        if "uri" in te and "name" in te:
                            topic_entities_dict[te["uri"]] = te["name"]
                        elif len(te) == 1:
                            topic_entities_dict.update(te)
                if topic_entities_dict:
                    result["topic_entities"] = topic_entities_dict
            elif isinstance(topic_entities, dict):
                result["topic_entities"] = topic_entities
    else:
        # For regular paths, include name_path
        result["name_path"] = path.get("name_path") or ""
        # Add topic_entity if available
        topic_entity = path.get("topic_entity")
        if topic_entity:
            result["topic_entity"] = topic_entity
    
    # Add answers if available
    answers = path.get("answers")
    if answers:
        if isinstance(answers, list):
            # Extract entity names if answers are dicts with "name" field
            answer_names = []
            for a in answers:
                if isinstance(a, dict) and "name" in a:
                    answer_names.append(a["name"])
                elif isinstance(a, str):
                    answer_names.append(a)
                else:
                    answer_names.append(str(a))
            result["answers"] = answer_names if answer_names else answers
        else:
            result["answers"] = [answers]
    
    return result


def _extract_single_thought(text: str) -> str | None:
    """Extract a single thought from <think>...</think> tag.
    
    Returns the thought content or None if not found.
    """
    # Match <think>...</think> tag (case insensitive, with optional whitespace)
    # Also try <think>...</think> as fallback for compatibility
    m = re.search(r"<\s*redacted_reasoning\s*>(.*?)<\s*/\s*redacted_reasoning\s*>", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    # Fallback to <think>...</think> for backward compatibility
    m = re.search(r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return None


def _inject_think_first(step: Dict[str, Any], think: str) -> None:
    """Insert a 'think' field at the beginning of the step dict (preserving order for other keys)."""
    if not think:
        return
    new_step: Dict[str, Any] = {"think": think}
    for k, v in step.items():
        if k == "think":
            continue
        new_step[k] = v
    step.clear()
    step.update(new_step)


def _strip_surface(value: str) -> str:
    """Strip list numbering from an observation item, keep surface entity/predicate text."""
    v = value.strip()
    v = re.sub(r"^\s*\d+\.\s*", "", v)  # leading numbering like "1. "
    return v.strip()


def generate_trajectories_file(
    *,
    in_file: str,
    traj_out: str,
    limit: int | None = None,
    client: LLMClient | None = None,
    checkpoint_every: int = 10,
    resume: bool = True,
    workers: int = 1,
    max_retries_per_path: int = 3,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> None:
    """Read an info_syn-like JSON and generate clean trajectory records only (pure mode).

    Writes an array of records with fields: raw_question, name_path, prompt, steps[{step_index, kg_query, information, think}],
    optionally final_thought, and answer. The answer is preferred from the last element of `name_path` (assumed unique entity),
    wrapped in <answer>...</answer>. If `name_path` is missing or malformed, we fallback to the last step's first observed entity.
    """
    # Note: a fresh LLMClient is created per task to avoid potential thread-safety issues.
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data  # we'll apply resume/limit filtering below

    # Load existing output for resume (if any)
    import os
    existing_records: List[Dict[str, Any]] = []
    processed_ids: Set[str] = set()
    if resume and os.path.exists(traj_out):
        try:
            with open(traj_out, "r", encoding="utf-8") as rf:
                existing_records = json.load(rf) or []
            for rec in existing_records:
                pid = rec.get("path_id")
                if pid is not None:
                    processed_ids.add(str(pid))
        except Exception:
            # If file is corrupted or not a JSON array, ignore and start fresh
            existing_records = []
            processed_ids = set()

    # Always collect clean trajectory records (newly generated this run)
    traj_records: List[Dict[str, Any]] = []
    
    # Thread-safe counter for discarded paths (due to missing results in steps)
    discarded_count_lock = threading.Lock()
    discarded_count = [0]  # Use list to allow modification in nested function

    def _write_checkpoint() -> None:
        """Atomically write combined records to traj_out."""
        combined = existing_records + traj_records
        tmp_path = f"{traj_out}.tmp"
        output_dir = os.path.dirname(traj_out)
        if output_dir:  # Only create directory if path contains a directory
            os.makedirs(output_dir, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as wf:
            json.dump(combined, wf, ensure_ascii=False, indent=2)
            wf.flush()
            os.fsync(wf.fileno())
        os.replace(tmp_path, traj_out)

    def _answer_from_name_path(name_path_val: Any) -> str | None:
        # name_path can be a string like "A -> r1 -> B -> r2 -> C" or a list
        if isinstance(name_path_val, list):
            if not name_path_val:
                return None
            cand = str(name_path_val[-1]).strip()
            return cand or None
        if isinstance(name_path_val, str):
            parts = [p.strip() for p in name_path_val.split("->")]
            parts = [p for p in parts if p]
            if not parts:
                return None
            return parts[-1]
        return None

    # Build candidate list honoring resume and limit
    candidates: List[Dict[str, Any]] = []
    for path in items:
        pid = path.get("path_id")
        pid_str = str(pid) if pid is not None else None
        if resume and pid_str is not None and pid_str in processed_ids:
            continue
        candidates.append(path)
    if isinstance(limit, int) and limit > 0:
        candidates = candidates[:limit]

    # Worker to process a single path and return a trajectory record or None on failure
    def _process_one(path: Dict[str, Any]) -> Dict[str, Any] | None:
        # Create a fresh client per task to avoid potential thread-safety issues
        # Use provided model config if available
        local_cli = LLMClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        steps: List[Dict[str, Any]] = path.get("steps", [])
        # Get question: prefer new_question, fallback to question, skip if neither exists
        raw_q = path.get("question")
        if not raw_q:
            # Skip this path if no question is available
            return None
        
        # Check if this is a conjunction path
        is_conj = _is_conjunction_path(path)
        
        # Build the prompt that KG Agent saw during rollout (for saving in output)
        if is_conj:
            # For conjunction paths, use topic_entities
            topic_entities = path.get("topic_entities")
            topic_entities_list = None
            if topic_entities:
                if isinstance(topic_entities, dict):
                    # Extract entity names from topic_entities dict (e.g., {"m.123": "Entity Name", "m.456": "Another Entity"})
                    topic_entities_list = list(topic_entities.values())
                elif isinstance(topic_entities, list):
                    # Extract entity names from list of dicts
                    topic_entities_list = []
                    for te in topic_entities:
                        if isinstance(te, dict):
                            if "name" in te:
                                topic_entities_list.append(te["name"])
                            elif len(te) == 1:
                                topic_entities_list.append(list(te.values())[0])
            # Fallback to name_path1 and name_path2 if topic_entities not available
            if not topic_entities_list:
                name_path1 = path.get("name_path1") or ""
                name_path2 = path.get("name_path2") or ""
                entity1 = (name_path1.split("->")[0].strip() if name_path1 else "")
                entity2 = (name_path2.split("->")[0].strip() if name_path2 else "")
                topic_entities_list = [e for e in [entity1, entity2] if e]
        else:
            # For regular paths, use topic_entity
            name_path = path.get("name_path") or ""
            topic_entity = path.get("topic_entity")
            topic_entities_list = None
            if topic_entity:
                if isinstance(topic_entity, dict):
                    # Extract entity name from topic_entity dict (e.g., {"m.123": "Entity Name"})
                    entity_name = list(topic_entity.values())[0] if topic_entity else None
                    if entity_name:
                        topic_entities_list = [entity_name]
                elif isinstance(topic_entity, str):
                    topic_entities_list = [topic_entity]
            # Fallback to name_path if topic_entity not available
            if not topic_entities_list:
                init_entity = (name_path.split("->")[0].strip() if name_path else "")
                topic_entities_list = [init_entity] if init_entity else None
        
        kg_agent_prompt = build_search_prompt(
            question=raw_q,
            max_calls=10,
            topic_entities=topic_entities_list,
        )
        
        # Select prompt based on path type
        # Call TRAJECTORY_GENERATION_PROMPT as a function to get a random example each time
        system_prompt = CONJUNCTION_TRAJECTORY_GENERATION_PROMPT() if is_conj else TRAJECTORY_GENERATION_PROMPT()
        
        # Get topic_entity/topic_entities for input
        if is_conj:
            # For conjunction paths, use topic_entities
            topic_entities = path.get("topic_entities")
            topic_entity_dict = None
            if isinstance(topic_entities, dict):
                topic_entity_dict = topic_entities
            elif isinstance(topic_entities, list) and topic_entities:
                # Convert list to dict format
                topic_entity_dict = {}
                for te in topic_entities:
                    if isinstance(te, dict):
                        if "uri" in te and "name" in te:
                            topic_entity_dict[te["uri"]] = te["name"]
                        elif len(te) == 1:
                            topic_entity_dict.update(te)
        else:
            # For regular paths, use topic_entity
            topic_entity_dict = path.get("topic_entity")
        
        # Step-by-step generation: for each step, generate the thought explaining why the next action is chosen
        per_step_records: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []  # Accumulate history as we go
        
        # Sort steps by step_index to preserve input order (do not filter, check all steps)
        sorted_steps = sorted(steps, key=lambda x: int(x.get("step_index", 0)))
        
        # Check if all steps have results - if any step lacks results, discard this path
        for step in sorted_steps:
            if not step.get("results"):
                # This path has a step without results, discard it and increment counter
                with discarded_count_lock:
                    discarded_count[0] += 1
                return None
        
        # All steps have results, proceed with processing
        valid_steps = sorted_steps
        
        # Get gold answers for final step
        # CRITICAL: Input file MUST have "answers" field - do NOT extract from name_path
        gold_answers = path.get("answers")
        if isinstance(gold_answers, list):
            gold_answers = [str(a).strip() for a in gold_answers if a]
        elif gold_answers:
            gold_answers = [str(gold_answers).strip()]
        else:
            gold_answers = []
        
        # If no answers, skip this path (input file must have "answers" field)
        if not gold_answers:
            return None
        
        # Process each step
        for step_idx, current_step in enumerate(valid_steps):
            step_index = int(current_step.get("step_index", 0))
            current_action = _build_action_for_step(valid_steps, current_step)
            current_observation = _obs_values(current_step)
            
            # Build input for this step
            # History contains all previous steps (think, action, observation) - does NOT include current step
            # next_action is the current step's action (the golden action we want to explain)
            # Field order: question, topic_entity/topic_entities, history, next_action
            step_input: Dict[str, Any] = {
                "question": raw_q,
            }
            # Add topic_entity or topic_entities based on path type (after question)
            if is_conj:
                step_input["topic_entities"] = topic_entity_dict if topic_entity_dict else []
            else:
                step_input["topic_entity"] = topic_entity_dict if topic_entity_dict else {}
            # Add history and next_action
            step_input["history"] = history.copy()  # Copy history up to this point (doesn't include current step)
            step_input["next_action"] = current_action  # The action for this step that we want to explain why it's chosen
            
            user_content = f"INPUT:\n{json.dumps(step_input, ensure_ascii=False, indent=2)}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            
            # Generate thought for this step
            thought: str | None = None
            for _ in range(max_retries_per_path):
                try:
                    text = local_cli._chat(messages)
                    thought = _extract_single_thought(text)
                    if thought:
                        break
                except Exception:
                    thought = None
            
            if thought is None:
                # Failed to generate thought, skip this step
                continue
            
            # Add this step to records
            per_step_records.append({
                "step_index": step_index,
                "think": thought,
                "kg_query": current_action,
                "information": current_observation,
            })
            
            # Add to history for next step (include the current step's think, action, and observation)
            history.append({
                "step_index": step_index,
                "think": thought,
                "action": current_action,
                "observation": current_observation,
            })
        
        # After processing all KG steps, generate a final thought for the answer action
        # This step is ESSENTIAL: it explains why the model can derive the answer from the gathered information
        # Note: gold_answers is guaranteed to be non-empty at this point (we return None earlier if empty)
        if valid_steps:
            # Build input for final answer step
            # Field order: question, topic_entity/topic_entities, history, next_action
            final_answer_action = f"<answer>{json.dumps(gold_answers, ensure_ascii=False)}</answer>"
            step_input: Dict[str, Any] = {
                "question": raw_q,
            }
            # Add topic_entity or topic_entities based on path type (after question)
            if is_conj:
                step_input["topic_entities"] = topic_entity_dict if topic_entity_dict else []
            else:
                step_input["topic_entity"] = topic_entity_dict if topic_entity_dict else {}
            # Add history and next_action
            step_input["history"] = history.copy()  # All previous steps including the last KG step
            step_input["next_action"] = final_answer_action
            
            user_content = f"INPUT:\n{json.dumps(step_input, ensure_ascii=False, indent=2)}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            
            # Generate final thought for answer - this is ESSENTIAL and must be generated
            # Use more retries for final step to ensure we get a valid thought
            final_thought: str | None = None
            max_retries_final = max_retries_per_path * 2  # Double retries for final step
            for retry in range(max_retries_final):
                try:
                    text = local_cli._chat(messages)
                    final_thought = _extract_single_thought(text)
                    if final_thought:
                        break
                except Exception:
                    # Continue retrying on error
                    final_thought = None
            
            # CRITICAL: Always add the final answer step
            # If thought generation failed after all retries, set think to None (null)
            last_step_index = int(valid_steps[-1].get("step_index", 0))
            
            # Add final answer step to records - this step is ALWAYS added
            per_step_records.append({
                "step_index": last_step_index + 1,
                "think": final_thought,  # Will be None if generation failed
                "kg_query": final_answer_action,
                "information": [],  # Empty as per requirement
            })
        
        # Extract final thought if we have a last step with answer
        final_thought_for_rec: str | None = None
        if per_step_records:
            # The last step's thought can serve as final_thought if it's about the answer
            last_record = per_step_records[-1]
            if "<answer>" in last_record.get("kg_query", ""):
                final_thought_for_rec = last_record.get("think", "")

        # Build output record based on path type
        rec: Dict[str, Any] = {
            "path_id": path.get("path_id"),
            "raw_question": raw_q,
            "prompt": kg_agent_prompt,  # This is the prompt that KG Agent saw during rollout
            "steps": per_step_records,
        }
        
        # For conjunction paths, save name_path1 and name_path2; for regular paths, save name_path
        if is_conj:
            rec["name_path1"] = path.get("name_path1") or ""
            rec["name_path2"] = path.get("name_path2") or ""
        else:
            rec["name_path"] = path.get("name_path") or ""
        if final_thought_for_rec:
            rec["final_thought"] = final_thought_for_rec
        
        # Set answer from gold_answers (already extracted above)
        if gold_answers:
            rec["answer"] = f"<answer>{json.dumps(gold_answers, ensure_ascii=False)}</answer>"
        else:
            # Fallback: try to extract from name_path
            name_path_val = path.get("name_path") if not is_conj else None
            if name_path_val:
                answer_candidate = _answer_from_name_path(name_path_val)
                if answer_candidate:
                    rec["answer"] = f"<answer>{json.dumps([answer_candidate], ensure_ascii=False)}</answer>"
                else:
                    rec["answer"] = "<answer>[]</answer>"
            else:
                rec["answer"] = "<answer>[]</answer>"
        
        return rec

    processed_in_this_run = 0
    if workers and workers > 1:
        # Parallel processing with shared progress bar
        try:  # pragma: no cover
            pbar = tqdm(total=len(candidates), desc="Generating trajectories", unit="path")  # type: ignore
        except Exception:  # pragma: no cover
            pbar = None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one, p) for p in candidates]
            for fut in as_completed(futures):
                rec = fut.result()
                if pbar is not None:
                    pbar.update(1)  # type: ignore[attr-defined]
                if rec is None:
                    continue  # skip failed path
                traj_records.append(rec)
                processed_in_this_run += 1
                if checkpoint_every > 0 and (processed_in_this_run % checkpoint_every == 0):
                    _write_checkpoint()
        if pbar is not None:
            pbar.close()  # type: ignore[attr-defined]
    else:
        # Sequential processing; fall back to no-op tqdm if not available
        try:  # pragma: no cover
            iterator = tqdm(candidates, desc="Generating trajectories", unit="path")  # type: ignore
        except Exception:  # pragma: no cover
            iterator = candidates

        for path in iterator:
            rec = _process_one(path)
            if rec is None:
                continue
            traj_records.append(rec)
            processed_in_this_run += 1
            if checkpoint_every > 0 and (processed_in_this_run % checkpoint_every == 0):
                _write_checkpoint()

    _write_checkpoint()
    
    # Print statistics about discarded paths
    if discarded_count[0] > 0:
        print(f"\nNote: Total discarded {discarded_count[0]} paths (reason: missing results in some steps)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate clean per-step reasoning trajectories (pure mode only).")
    parser.add_argument("--in-file", required=True)
    parser.add_argument("--traj-out", required=True, help="Write clean trajectory records to this file")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of paths to process")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Write checkpoint to output every N processed paths")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume-from-output and start fresh")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for batch processing")
    parser.add_argument("--max-retries-per-path", type=int, default=3, help="Max retries for a single path on transient failures")
    
    # Model configuration arguments
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g., gpt-4o-mini)")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for the API")
    parser.add_argument("--api-key", type=str, default=None, help="API key (overrides environment variable)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for generation (e.g., 0.6)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens for generation")
    
    args = parser.parse_args()

    generate_trajectories_file(
        in_file=args.in_file,
        traj_out=args.traj_out,
        limit=args.limit,
        client=None,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        workers=args.workers,
        max_retries_per_path=args.max_retries_per_path,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
