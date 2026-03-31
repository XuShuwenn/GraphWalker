"""GraphWalker Reward Computation Module

This module contains all reward calculation logic for GraphWalker training:
- Turn-level rewards (v_fmt, v_kg)
- Global rewards (answer accuracy, retrieval score)
- Reward aggregation and breakdown
"""

import json
import re
from typing import Tuple, Dict, List, Optional

from src.eval.metrics import exact_match, f1_score, qa_normalize_answer


# Compiled regex patterns for performance
_PATTERN_THINK = re.compile(r'<think>.*?</think>', re.DOTALL)
_PATTERN_ANSWER = re.compile(r'<answer>((?:(?!<answer>).)*?)</answer>', re.DOTALL)
_PATTERN_KG_QUERY = re.compile(r'<kg-query>((?:(?!<kg-query>).)*?)</kg-query>', re.DOTALL)


# ============== Helper Functions ==============

def extract_answer(text: str) -> Optional[str]:
    """Extract <answer>...</answer> content (last occurrence)."""
    matches = _PATTERN_ANSWER.findall(text)
    return matches[-1].strip() if matches else None


def extract_turn_model_content(model_generated_response: str, turn: dict) -> str:
    """Extract model-generated content for a specific turn by matching turn's think/kg_query/answer."""
    if not model_generated_response or not turn:
        return ""
    
    kg_query = turn.get("kg_query")
    if kg_query:
        escaped_query = re.escape(kg_query)
        pattern = rf'<kg-query>((?:(?!<kg-query>).)*?{escaped_query}(?:(?!<kg-query>).)*?)</kg-query>'
        match = re.search(pattern, model_generated_response, re.DOTALL)
        if match:
            start_pos = match.start()
            think_match = _PATTERN_THINK.search(model_generated_response[:start_pos])
            if think_match:
                return model_generated_response[think_match.start():match.end()]
            return match.group(0)
    
    answer = turn.get("answer")
    if answer:
        # Use finditer to get match objects (which include full <answer>...</answer> tags)
        answer_matches = list(_PATTERN_ANSWER.finditer(model_generated_response))
        if answer_matches:
            last_match = answer_matches[-1]
            answer_start = last_match.start()  # Start of <answer> tag
            answer_end = last_match.end()      # End of </answer> tag
            
            think_match = _PATTERN_THINK.search(model_generated_response[:answer_start])
            if think_match:
                # Return from <think> to </answer> (complete tags included)
                return model_generated_response[think_match.start():answer_end]
            # Return complete <answer>...</answer> (tags included)
            return model_generated_response[answer_start:answer_end]
    
    think = turn.get("think")
    if think:
        think_normalized = think.strip().lower()
        for match in _PATTERN_THINK.finditer(model_generated_response):
            think_content = match.group(0).lower()
            if think_normalized[:50] in think_content:
                start_pos = match.start()
                remaining = model_generated_response[start_pos:]
                action_match = re.search(r'<(kg-query|answer)>((?:(?!<\1>).)*?)</\1>', remaining, re.DOTALL)
                if action_match:
                    return remaining[:action_match.end()]
                return match.group(0)
    
    return ""


# ============== Turn-Level Reward Functions ==============

def v_fmt(turn_model_content: str, is_final_turn: bool) -> float:
    """Format validity reward. Checks if content follows required format:
    - Final turn: <answer>...</answer> with valid JSON list
    - Other turns: <think>...</think><kg-query>...</kg-query>
    """
    if not turn_model_content:
        return 0.0
    
    if is_final_turn:
        has_answer = bool(_PATTERN_ANSWER.search(turn_model_content))
        if has_answer:
            answer_matches = _PATTERN_ANSWER.findall(turn_model_content)
            if answer_matches:
                answer_content = answer_matches[-1].strip()
                try:
                    parsed = json.loads(answer_content)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return 1.0
                except (json.JSONDecodeError, ValueError):
                    pass
        return 0.0
    else:
        has_think = bool(_PATTERN_THINK.search(turn_model_content))
        has_kg_query = bool(_PATTERN_KG_QUERY.search(turn_model_content))
        return 1.0 if (has_think and has_kg_query) else 0.0


def v_kg(turn: dict) -> float:
    """KG query reward. Checks if query produces valid results."""
    query_success = turn.get("query_success", False)
    information = turn.get("information", "")
    
    if not query_success or not information or not information.strip():
        return 0.0
    
    error_indicators = [
        "Invalid entity",
        "not in the latest predicate list",
        "Could not parse query",
        "KG query service not available",
        "No triples found",
    ]
    
    if any(err in information for err in error_indicators):
        return 0.0
    
    return 1.0


# ============== Global Reward Functions ==============

def compute_retrieval_score(turns_history: list, ground_truth: list) -> float:
    """Retrieval score. Checks if gold answer entities appear in retrieved information."""
    if not turns_history or not ground_truth:
        return 0.0
    
    all_information = [
        (turn.get("information") or "")
        for turn in turns_history
        if (turn.get("information") or "").strip()
    ]
    
    if not all_information:
        return 0.0
    
    combined_info = " ".join(all_information).lower()
    
    for gold_answer in ground_truth:
        if not gold_answer:
            continue
        normalized_gold = qa_normalize_answer(str(gold_answer))
        if normalized_gold and normalized_gold in combined_info:
            return 1.0
    
    return 0.0


# ============== Main Reward Computation ==============

def compute_graphwalker_reward(
    response: str, 
    ground_truth: list, 
    model_generated_response: str = None,
    turns_history: list = None,
    config: dict = None,
) -> Tuple[float, dict]:
    """Compute GraphWalker reward: turn-level (v_fmt + v_kg) + global (answer + retrieval).
    
    Args:
        response: Complete response string
        ground_truth: List of ground truth answers
        model_generated_response: Model-generated portion (without observations)
        turns_history: List of turn information dicts
        config: Optional config dict (if None, uses default from GRAPHWALKER_CONFIGS)
    
    Returns:
        Tuple of (total_reward, reward_breakdown_dict)
    """
    # Load config (allow override for testing)
    if config is None:
        from generate_with_search import GRAPHWALKER_CONFIGS
        config = GRAPHWALKER_CONFIGS
    
    w_fmt = config["turn_reward_weights"]["w_fmt"]
    w_kg = config["turn_reward_weights"]["w_kg"]
    w_kg_turns = config["turn_reward_combination"]["w_kg_turns"]
    w_answer_turn = config["turn_reward_combination"]["w_answer_turn"]
    w_answer = config["global_reward_weights"]["w_answer"]
    w_ret = config["global_reward_weights"]["w_ret"]
    global_answer_metric = config.get("global_answer_metric", "f1")
    if global_answer_metric not in ("f1", "em"):
        global_answer_metric = "f1"
    
    if turns_history is None:
        turns_history = []
    if model_generated_response is None:
        model_generated_response = response
    
    # Compute turn-level rewards
    kg_turn_rewards = []
    answer_turn_reward = 0.0
    turn_rewards_detail = []
    
    for turn_idx, turn in enumerate(turns_history):
        turn_model_content = extract_turn_model_content(model_generated_response, turn)
        is_final_turn = (turn_idx == len(turns_history) - 1) and ("answer" in turn and turn.get("answer") is not None)
        
        fmt_score = v_fmt(turn_model_content, is_final_turn)
        
        if is_final_turn:
            kg_score = None
            r_turn = w_fmt * fmt_score
            answer_turn_reward = r_turn
        else:
            kg_score = v_kg(turn)
            r_turn = w_fmt * fmt_score + w_kg * kg_score
            kg_turn_rewards.append(r_turn)
        
        turn_rewards_detail.append({
            "turn": turn_idx + 1,
            "v_fmt": fmt_score,
            "v_kg": kg_score,
            "r_turn": r_turn,
            "is_final_turn": is_final_turn,
        })
    
    mean_kg_turn_reward = sum(kg_turn_rewards) / len(kg_turn_rewards) if kg_turn_rewards else 0.0
    total_turn_reward = w_kg_turns * mean_kg_turn_reward + w_answer_turn * answer_turn_reward
    
    # Compute global rewards
    answer_content = extract_answer(response)
    f1 = f1_score(answer_content, ground_truth) if answer_content and ground_truth else 0.0
    em = exact_match(answer_content, ground_truth) if answer_content and ground_truth else 0.0
    v_ret_score = compute_retrieval_score(turns_history, ground_truth)
    
    answer_score = f1 if global_answer_metric == "f1" else em
    global_reward = w_answer * answer_score + w_ret * v_ret_score
    total_reward = global_reward + total_turn_reward
    
    reward_breakdown = {
        "weights": {
            "w_fmt": w_fmt,
            "w_kg": w_kg,
            "w_kg_turns": w_kg_turns,
            "w_answer_turn": w_answer_turn,
            "w_answer": w_answer,
            "w_ret": w_ret,
            "global_answer_metric": global_answer_metric,
        },
        "turn_rewards": turn_rewards_detail,
        "mean_kg_turn_reward": mean_kg_turn_reward,
        "answer_turn_reward": answer_turn_reward,
        "total_turn_reward": total_turn_reward,
        "turn_reward_breakdown": {
            "kg_contribution": w_kg_turns * mean_kg_turn_reward,
            "answer_contribution": w_answer_turn * answer_turn_reward,
        },
        "metrics": {
            "f1": f1,
            "em": em,
            "v_ret": v_ret_score,
        },
        "global_reward": global_reward,
        "global_reward_breakdown": {
            "answer_contribution": w_answer * answer_score,
            "answer_metric": global_answer_metric,
            "retrieval_contribution": w_ret * v_ret_score,
        },
        "total_reward": total_reward,
    }
    
    return total_reward, reward_breakdown
