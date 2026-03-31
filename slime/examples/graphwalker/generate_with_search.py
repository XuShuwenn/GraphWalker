# GraphWalker's KG-based multi-turn interaction for SLIME training
# Reuses KGAugmentedModelClient for KG logic, implements training-specific logic
import asyncio
import json
import re
import random
import sys
import os
import time
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from prompts.prompts import build_search_prompt, build_continuation_prompt, FORCE_ANSWER_PROMPT
from src.eval.kg_augmented_client import KGAugmentedModelClient
from graphwalker_reward import compute_graphwalker_reward, extract_answer
from trajectory_utils import save_trajectory

# Compiled regex patterns for performance
_PATTERN_KG_QUERY = re.compile(r'<kg-query>((?:(?!<kg-query>).)*?)</kg-query>', re.DOTALL)

GRAPHWALKER_CONFIGS = {
    "max_calls": 10,
    "kg_top_k": 10,
    "search_concurrency": 128,
    "kg_server_url": "http://localhost:18890",
    "kg_timeout": 10,
    "return_logprob": True,
    "stop_strings": ["</kg-query>", "</answer>"],  # Stop generation at these strings
    "turn_reward_weights": {
        "w_fmt": float(os.getenv("REWARD_W_FMT", "0.5")),
        "w_kg": float(os.getenv("REWARD_W_KG", "0.5")),
    },
    "turn_reward_combination": {
        "w_kg_turns": float(os.getenv("REWARD_W_KG_TURNS", "0.0")),
        "w_answer_turn": float(os.getenv("REWARD_W_ANSWER_TURN", "0.0")),
    },
    "global_reward_weights": {
        "w_answer": float(os.getenv("REWARD_W_ANSWER", "1.0")),
        "w_ret": float(os.getenv("REWARD_W_RET", "0.0")),
    },
    "global_answer_metric": os.getenv("REWARD_ANSWER_METRIC", "em"),  # "f1" or "em"
    # Temporarily disable trajectory saving to speed up rollouts
    "save_trajectories": False,
    "trajectory_save_dir": None,
    "trajectory_save_frequency": 1,
}

SEMAPHORE = asyncio.Semaphore(GRAPHWALKER_CONFIGS["search_concurrency"])

_perf_logger = logging.getLogger("graphwalker.performance")
_perf_logger.setLevel(logging.INFO)
if not _perf_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[PERF] %(message)s'))
    _perf_logger.addHandler(handler)

_shared_filter_client = None
_shared_filter_client_initialized = False


def _get_shared_filter_client():
    """Get or create shared filter client for LLM relation filtering (singleton)."""
    global _shared_filter_client, _shared_filter_client_initialized
    if _shared_filter_client_initialized:
        return _shared_filter_client
    _shared_filter_client_initialized = True
    
    from src.eval.model_client import BaseModelClient, ModelConfig
    
    filter_api_key = os.getenv("FILTER_API_KEY")
    filter_api_url = os.getenv("FILTER_API_URL")
    filter_model = os.getenv("FILTER_MODEL", "deepseek-chat")
        
    cfg = {
        "model": filter_model,
        "api_key": filter_api_key,
        "base_url": filter_api_url,
        "temperature": 0.0,
        "max_tokens": 1024,
        "stop": ["\n\n"],
        "timeout": 10.0,
    }
    return BaseModelClient(ModelConfig(**cfg))


def _create_kg_client(topic_entity: dict) -> KGAugmentedModelClient:
    """Create per-sample KGAugmentedModelClient (reuses shared filter_client)."""
    client = KGAugmentedModelClient(
        base_client=None,
        kg_server_url=GRAPHWALKER_CONFIGS["kg_server_url"],
        max_calls=GRAPHWALKER_CONFIGS["max_calls"],
        kg_top_k=GRAPHWALKER_CONFIGS["kg_top_k"],
        kg_timeout=GRAPHWALKER_CONFIGS.get("kg_timeout", 10),
        filter_client=_get_shared_filter_client(),
        dataset_type="cwq",
    )
    client._seed_registry_from_topic_entities(topic_entity)
    client._initial_entity_names = client._extract_initial_entity_names(topic_entity)
    return client


# ============== Helper Functions ==============

def _apply_user_message_template(state: GenerateState, content: str) -> str:
    """Apply chat template for user message without system prompt (for subsequent turns).
    
    Simple approach: 
    1. Extract assistant closing tag from tokenizer
    2. Generate user message template with empty system
    3. Remove empty system block
    4. Prepend assistant closing tag
    
    This ensures proper format: </assistant> <user>content</user> <assistant>
    """
    # Step 1: Extract assistant closing tag
    # Generate a minimal assistant message to identify the closing tag
    dummy_assistant = state.tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "assistant", "content": "x"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    # The closing tag is everything after "x"
    close_tag_pos = dummy_assistant.rfind("x")
    if close_tag_pos != -1:
        assistant_close = dummy_assistant[close_tag_pos + 1:]
    else:
        assistant_close = "<|im_end|>\n"  # Fallback for Qwen
    
    # Step 2: Generate user message template with empty system
    full_template = state.tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Step 3: Remove empty system block
    only_system = state.tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}],
        tokenize=False,
        add_generation_prompt=False,
    )
    
    user_part = full_template[len(only_system):] if full_template.startswith(only_system) else full_template
    
    # Step 4: Prepend assistant closing tag
    return assistant_close + user_part


def _append_observation(
    state: GenerateState,
    obs_text: str,
    response: str,
    response_token_ids: list,
    loss_mask: list,
    rollout_log_probs: list | None,
) -> tuple[str, list, list, list | None]:
    """Append observation (loss_mask=0) to response."""
    obs_token_ids = state.tokenizer(obs_text, add_special_tokens=False)["input_ids"]
    response += obs_text
    response_token_ids += obs_token_ids
    loss_mask += [0] * len(obs_token_ids)
    if GRAPHWALKER_CONFIGS["return_logprob"] and rollout_log_probs is not None:
        rollout_log_probs += [0.0] * len(obs_token_ids)
    return response, response_token_ids, loss_mask, rollout_log_probs


def _fill_sample_fields(
    sample: Sample,
    prompt_tokens_ids: list,
    response_token_ids: list,
    response: str,
    model_generated_response: str,
    prompt_text: str,
    instruction_prompt: str,
    turns_history: list,
    loss_mask: list,
    rollout_log_probs: list | None = None,
):
    """Fill sample with generated content."""
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample._model_generated_response = model_generated_response
    sample.loss_mask = loss_mask
    sample._full_prompt = prompt_text
    sample._instruction_prompt = instruction_prompt
    sample._turns_history = turns_history
    sample.prompt = prompt_text
    if GRAPHWALKER_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs


# ============== Main Generate Function ==============

async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Main generate function for GraphWalker multi-turn KG interaction.
    
    Supports partial rollout: if sample.response_length > 0, continues from existing state.
    """
    # 断点: generate 入口，查看 sample、sampling_params (调试时取消注释)
    # breakpoint()
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # ============== Partial Rollout: Check if continuing from buffer ==============
    # NOTE: Partial rollout code temporarily commented out - DO NOT DELETE
    # is_continuing = sample.response_length > 0 if hasattr(sample, 'response_length') else False
    is_continuing = False  # Temporarily disabled
    
    # if is_continuing and args.partial_rollout:
    #     # Mask previous off-policy generation for partial rollout (following sglang_rollout.py)
    #     if args.mask_offpolicy_in_partial_rollout and sample.loss_mask is not None:
    #         sample.loss_mask = [0] * sample.response_length
    #     
    #     # Normalize metadata to dict (cwq_train_prepared.jsonl stores metadata as JSON str)
    #     if not isinstance(sample.metadata, dict):
    #         if sample.metadata is None:
    #             sample.metadata = {}
    #         elif isinstance(sample.metadata, str):
    #             try:
    #                 sample.metadata = json.loads(sample.metadata)
    #             except (json.JSONDecodeError, TypeError):
    #                 sample.metadata = {"_raw": sample.metadata}
    #         else:
    #             sample.metadata = {"_raw": sample.metadata}
    #     
    #     # Restore state from partial sample
    #     _perf_logger.info(f"[Sample {getattr(sample, 'index', '?')}] Continuing from partial rollout "
    #                      f"(response_length={sample.response_length}, from rollout_id={sample.metadata.get('start_rollout_id', '?')})")
    #     
    #     # Extract saved state from sample
    #     prompt_tokens_ids = sample.tokens[:len(sample.tokens) - sample.response_length] if sample.tokens else []
    #     response_token_ids = sample.tokens[len(prompt_tokens_ids):] if sample.tokens else []
    #     response = sample.response or ""
    #     loss_mask = sample.loss_mask if sample.loss_mask is not None else []
    #     rollout_log_probs = sample.rollout_log_probs if GRAPHWALKER_CONFIGS["return_logprob"] else None
    #     
    #     # Restore metadata
    #     turns_history = getattr(sample, '_turns_history', [])
    #     model_generated_response = getattr(sample, '_model_generated_response', response)
    #     prompt_text = getattr(sample, '_full_prompt', sample.prompt)
    #     instruction_prompt = getattr(sample, '_instruction_prompt', "")
    #     original_question = getattr(sample, '_original_question', sample.prompt)
    #     
    #     # Reconstruct KG client
    #     metadata = json.loads(sample.metadata) if isinstance(sample.metadata, str) else (sample.metadata or {})
    #     topic_entity = metadata.get("topic_entity", {})
    #     kg_client = _create_kg_client(topic_entity)
    #     
    #     # Determine starting turn (number of completed turns in turns_history)
    #     start_turn_idx = len(turns_history)
    #     
    # else:
    if True:  # Always use fresh generation (partial rollout disabled)
        # ============== Fresh Generation: Initialize from scratch ==============
        metadata = json.loads(sample.metadata) if isinstance(sample.metadata, str) else (sample.metadata or {})
        topic_entity = metadata.get("topic_entity", {})
        original_question = sample.prompt
        
        sample._original_question = original_question
        kg_client = _create_kg_client(topic_entity)
        
        topic_names = list(topic_entity.values()) if topic_entity else []
        instruction_prompt = build_search_prompt(
            question=original_question,
            max_calls=GRAPHWALKER_CONFIGS["max_calls"],
            topic_entities=topic_names
        )
        
        # Manual chat_template application commented out; effect = use raw instruction as prompt.
        messages = [{"role": "user", "content": instruction_prompt}]
        prompt_text = state.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        
        prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        
        response = ""
        model_generated_response = ""
        response_token_ids = []
        loss_mask = []
        rollout_log_probs = [] if GRAPHWALKER_CONFIGS["return_logprob"] else None
        turns_history = []
        start_turn_idx = 0
    
    # ============== Common: Initialize timing and limits ==============
    sample_start_time = time.time()
    total_sglang_time = 0.0
    
    timeout_sec = getattr(args, "rollout_sample_timeout_sec", None)
    if timeout_sec is None:
        timeout_sec = GRAPHWALKER_CONFIGS.get("rollout_sample_timeout_sec")


    # ============== Multi-turn Loop: Continue from start_turn_idx ==============
    for turn_idx in range(start_turn_idx, GRAPHWALKER_CONFIGS["max_calls"]):
        turn_start_time = time.time()
        _perf_logger.info(f"[Sample {getattr(sample, 'index', '?')}] Turn {turn_idx + 1} started")
        
        # Check timeout
        if timeout_sec is not None and (time.time() - sample_start_time) >= timeout_sec:
            _perf_logger.info(f"[Sample {getattr(sample, 'index', '?')}] Timeout after {time.time() - sample_start_time:.1f}s")
            sample.status = Sample.Status.ABORTED
            sample.remove_sample = True
            # Ensure metadata is a dict before setting timed_out flag
            if not isinstance(sample.metadata, dict):
                sample.metadata = {} if not sample.metadata else {}
            sample.metadata["timed_out"] = True
            _fill_sample_fields(
                sample, prompt_tokens_ids, response_token_ids, response,
                model_generated_response, prompt_text, instruction_prompt,
                turns_history, loss_mask or [], rollout_log_probs
            )
            sample.reward = 0.0
            return sample
        
        # Prepare sampling params with stop strings
        sampling_params_with_stop = sampling_params.copy()
        if "stop" not in sampling_params_with_stop or sampling_params_with_stop["stop"] is None:
            sampling_params_with_stop["stop"] = []
        # Merge configured stop strings with existing ones (avoid duplicates)
        existing_stops = set(sampling_params_with_stop["stop"]) if sampling_params_with_stop["stop"] else set()
        sampling_params_with_stop["stop"] = list(existing_stops | set(GRAPHWALKER_CONFIGS["stop_strings"]))
        
        # Use existing tokens for partial rollout continuation (following sglang_rollout.py)
        # NOTE: Partial rollout payload code temporarily commented out - DO NOT DELETE
        # if is_continuing and len(sample.tokens) > 0:
        #     payload = {
        #         "input_ids": sample.tokens,
        #         "sampling_params": sampling_params_with_stop,
        #     }
        # else:
        #     payload = {
        #         "text": prompt_text + response,
        #         "sampling_params": sampling_params_with_stop,
        #     }
        
        # Always use text-based payload (partial rollout disabled)
        payload = {
            "text": prompt_text + response,
            "sampling_params": sampling_params_with_stop,
        }
        
        if GRAPHWALKER_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        sglang_start = time.time()
        output = await post(url, payload)
        sglang_time = time.time() - sglang_start
        total_sglang_time += sglang_time
        # 断点: SGLang 返回后，查看 output、cur_response、finish_reason (调试时取消注释)
        # breakpoint()
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]

        if "<kg-query>" in cur_response and "</kg-query>" not in cur_response:
            cur_response += "</kg-query>"

        if GRAPHWALKER_CONFIGS["return_logprob"]:
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError("output_token_logprobs not found")
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
        
        # Log SGLang performance with output token count
        num_output_tokens = len(cur_response_token_ids)
        _perf_logger.info(f"[Sample {getattr(sample, 'index', '?')}] Turn {turn_idx + 1} SGLang: {sglang_time:.2f}s, out_tokens: {num_output_tokens}, throughput: {num_output_tokens/sglang_time:.1f} token/s")

        response += cur_response
        model_generated_response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)
        if GRAPHWALKER_CONFIGS["return_logprob"]:
            rollout_log_probs += cur_response_log_probs
        
        # Update sample.tokens for partial rollout continuation
        # NOTE: Partial rollout token update code temporarily commented out - DO NOT DELETE
        # if is_continuing:
        #     sample.tokens = sample.tokens + cur_response_token_ids
        #     sample.response_length += len(cur_response_token_ids)
        #     if args.mask_offpolicy_in_partial_rollout and sample.loss_mask is not None:
        #         # Append new loss_mask for on-policy tokens (following sglang_rollout.py)
        #         sample.loss_mask += [1] * len(cur_response_token_ids)
        #     is_continuing = False  # After first turn in continuation, treat as normal

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        kg_query = kg_client._extract_kg_query(cur_response)
        answer = kg_client._extract_answer_tag(cur_response)
        
        kg_query_match = re.search(r'<kg-query>', cur_response)
        answer_match = re.search(r'<answer>', cur_response)
        kg_query_pos = kg_query_match.start() if kg_query_match else float('inf')
        answer_pos = answer_match.start() if answer_match else float('inf')
        
        if kg_query and kg_query_pos < answer_pos:
            think_content = kg_client._extract_think(cur_response)
            
            kg_start = time.time()
            success, query_results = await kg_client._parse_and_execute_query_async(
                kg_query, question=original_question
            )
            kg_time = time.time() - kg_start
            _perf_logger.info(f"[Sample {getattr(sample, 'index', '?')}] Turn {turn_idx + 1} KG query (async): {kg_time:.2f}s")

            turns_history.append({
                "turn": turn_idx + 1,
                "raw_response": cur_response,
                "think": think_content,
                "kg_query": kg_query,
                "information": query_results,
                "query_success": success,
            })

            next_obs_text = build_continuation_prompt(query_results)
            next_obs = _apply_user_message_template(state, next_obs_text)
            response, response_token_ids, loss_mask, rollout_log_probs = _append_observation(
                state, next_obs, response, response_token_ids, loss_mask, rollout_log_probs
            )

        elif answer:
            think_content = kg_client._extract_think(cur_response)
            turns_history.append({
                "turn": turn_idx + 1,
                "raw_response": cur_response,
                "think": think_content,
                "kg_query": None,
                "information": None,
                "query_success": None,
                "answer": answer,
            })
            break

        else:
            think_content = kg_client._extract_think(cur_response)
            if turn_idx < GRAPHWALKER_CONFIGS["max_calls"] - 1:
                turns_history.append({
                    "turn": turn_idx + 1,
                    "raw_response": cur_response,
                    "think": think_content,
                    "kg_query": None, "information": None, "query_success": None,
                    "error": "No valid <kg-query> or <answer> tag found",
                })
                
                error_obs_text = (
                    "\n\nYour response did not contain a valid <kg-query> or <answer> tag. "
                    "Please continue by outputting a valid <kg-query> or <answer>.\n\n"
                )
                error_obs = _apply_user_message_template(state, error_obs_text)
                response, response_token_ids, loss_mask, rollout_log_probs = _append_observation(
                    state, error_obs, response, response_token_ids, loss_mask, rollout_log_probs
                )
            else:
                turns_history.append({
                    "turn": turn_idx + 1,
                    "raw_response": cur_response,
                    "think": think_content,
                    "kg_query": None, "information": None, "query_success": None,
                    "error": "Max turns reached without answer",
                })
                break
        # 断点: KG 信息添加到 response 后，查看完整的 response、loss_mask (调试时取消注释)
        # breakpoint()
            
        turn_time = time.time() - turn_start_time
        _perf_logger.info(f"[Sample {getattr(sample, 'index', '?')}] Turn {turn_idx + 1} total: {turn_time:.2f}s")
    
    # ============== Force Answer if max_calls reached without answer ==============
    # Check if we've completed the loop without finding an answer
    final_answer = None
    if turns_history:
        final_answer = turns_history[-1].get("answer")
    
    # If no answer found and we've reached max_calls, send FORCE_ANSWER_PROMPT
    if final_answer is None and len(turns_history) >= GRAPHWALKER_CONFIGS["max_calls"]:
        _perf_logger.info(f"[Sample {getattr(sample, 'index', '?')}] Max calls reached without answer, sending FORCE_ANSWER_PROMPT")
        # Append FORCE_ANSWER_PROMPT as user message
        force_answer_obs = _apply_user_message_template(state, FORCE_ANSWER_PROMPT)
        response, response_token_ids, loss_mask, rollout_log_probs = _append_observation(
            state, force_answer_obs, response, response_token_ids, loss_mask, rollout_log_probs
        )
        
        # Prepare sampling params with stop strings
        sampling_params_with_stop = sampling_params.copy()
        if "stop" not in sampling_params_with_stop or sampling_params_with_stop["stop"] is None:
            sampling_params_with_stop["stop"] = []
        existing_stops = set(sampling_params_with_stop["stop"]) if sampling_params_with_stop["stop"] else set()
        sampling_params_with_stop["stop"] = list(existing_stops | set(GRAPHWALKER_CONFIGS["stop_strings"]))
        
        # Generate one more time to get the forced answer
        payload = {
            "text": prompt_text + response,
            "sampling_params": sampling_params_with_stop,
        }
        if GRAPHWALKER_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True
        
        sglang_start = time.time()
        output = await post(url, payload)
        sglang_time = time.time() - sglang_start
        total_sglang_time += sglang_time
        
        if output["meta_info"]["finish_reason"]["type"] != "abort":
            cur_response = output["text"]
            
            if GRAPHWALKER_CONFIGS["return_logprob"]:
                if "output_token_logprobs" not in output["meta_info"]:
                    raise RuntimeError("output_token_logprobs not found")
                cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            else:
                cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
            
            num_output_tokens = len(cur_response_token_ids)
            _perf_logger.info(f"[Sample {getattr(sample, 'index', '?')}] Force answer SGLang: {sglang_time:.2f}s, out_tokens: {num_output_tokens}")
            
            response += cur_response
            model_generated_response += cur_response
            response_token_ids += cur_response_token_ids
            loss_mask += [1] * len(cur_response_token_ids)
            if GRAPHWALKER_CONFIGS["return_logprob"]:
                rollout_log_probs += cur_response_log_probs
            
            # Extract answer from forced response
            answer = kg_client._extract_answer_tag(cur_response)
            think_content = kg_client._extract_think(cur_response)
            
            turns_history.append({
                "turn": len(turns_history) + 1,
                "raw_response": cur_response,
                "think": think_content,
                "kg_query": None,
                "information": None,
                "query_success": None,
                "answer": answer,
                "forced_answer": True,  # Mark this as a forced answer
            })
    
    sample_total_time = time.time() - sample_start_time
    _perf_logger.info(f"[Sample {getattr(sample, 'index', '?')}] Sample total: {sample_total_time:.2f}s "
                     f"(SGLang: {total_sglang_time:.2f}s)")

    if GRAPHWALKER_CONFIGS["return_logprob"] and rollout_log_probs:
        assert len(response_token_ids) == len(rollout_log_probs), \
            f"Token/logp length mismatch: {len(response_token_ids)} vs {len(rollout_log_probs)}"

    _fill_sample_fields(
        sample, prompt_tokens_ids, response_token_ids, response,
        model_generated_response, prompt_text, instruction_prompt,
        turns_history, loss_mask, rollout_log_probs
    )

    if sample.status == Sample.Status.PENDING:
        finish_type = output["meta_info"]["finish_reason"]["type"]
        if finish_type == "length":
            sample.status = Sample.Status.TRUNCATED
        elif finish_type == "abort":
            sample.status = Sample.Status.ABORTED
        elif finish_type == "stop":
            sample.status = Sample.Status.COMPLETED
    
    if sample.status == Sample.Status.ABORTED:
        sample.remove_sample = True

    return sample


# ============== Reward Function Entry Point ==============

async def reward_func(args, sample, **kwargs) -> float:
    """GraphWalker reward function entry point.
    
    This function serves as the interface for the training framework.
    All reward computation logic is in graphwalker_reward.py.
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    metadata = json.loads(sample.metadata) if isinstance(sample.metadata, str) else (sample.metadata or {})
    ground_truth = metadata.get("answers", [])
    
    if not ground_truth:
        if hasattr(sample, 'label') and sample.label:
            if isinstance(sample.label, dict):
                ground_truth = sample.label.get("ground_truth", []) or sample.label.get("answers", [])
            elif isinstance(sample.label, list):
                ground_truth = sample.label
            elif isinstance(sample.label, str):
                ground_truth = [sample.label]
    
    model_generated_response = getattr(sample, '_model_generated_response', None)
    turns_history = getattr(sample, '_turns_history', [])

    # Call the actual reward computation from graphwalker_reward.py
    score, reward_breakdown = compute_graphwalker_reward(
        response=sample.response,
        ground_truth=ground_truth,
        model_generated_response=model_generated_response,
        turns_history=turns_history,
        config=GRAPHWALKER_CONFIGS,  # Pass config explicitly
    )
    # 断点: 计算完 reward，查看 score、reward_breakdown (调试时取消注释)
    # breakpoint()
    
    # Save trajectory (pass config to trajectory_utils) - only if enabled in config.
    if GRAPHWALKER_CONFIGS.get("save_trajectories", False):
        await save_trajectory(sample, score, reward_breakdown, metadata, ground_truth, GRAPHWALKER_CONFIGS)
    
    
    return score
