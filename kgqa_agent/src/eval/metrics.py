"""Evaluation metrics for KGQA tasks.

Includes:
- Token-based F1 (normalized)
- Relaxed EM (normalized exact + bidirectional substring)
"""
from __future__ import annotations
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import string

def qa_normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an)\b", " ", s)
    s = " ".join(s.split())
    return s

def parse_prediction(pred: str) -> List[str]:
    if not pred:
        return []
    clean = pred.strip()
    
    # Try parsing as JSON list
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return [str(p).strip() for p in parsed if p]
        # If it's a single string in JSON format (unlikely but possible if model outputs "Answer")
        return [str(parsed).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback for legacy formats or malformed JSON
    # If prediction uses pipe separators: "A | B | C"
    if "|" in clean:
        return [p.strip() for p in clean.split("|") if p.strip()]
    
    # Fallback: return the whole prediction as single candidate
    return [clean]

def _single_exact_match(pred: str, golds: List[str]) -> float:
    """Compute EM for a single prediction string against golds."""
    if not pred:
        return 0.0

    if isinstance(golds, str):
        gold_list = [golds]
    else:
        gold_list = golds or []

    npred = qa_normalize_answer(pred)
    for g in gold_list:
        if qa_normalize_answer(str(g)) == npred:
            return 1.0
    for g in gold_list:
        if qa_normalize_answer(str(g)) in npred:
            return 1.0
    for g in gold_list:
        if npred and (npred in qa_normalize_answer(str(g))):
            return 1.0
    return 0.0

def exact_match(pred: str, golds: List[str]) -> float:
    """Compute best EM across all parsed predictions."""
    preds = parse_prediction(pred)
    if not preds:
        return 0.0
    return max(_single_exact_match(p, golds) for p in preds)


def token_f1_score(pred: str, golds: List[str]) -> float:
    """Compute SQuAD-style F1 score (Max F1 over gold aliases).
    
    Unlike the previous implementation which merged all golds into a single token set
    (penalizing cases with many aliases), this computes the F1 score for each 
    (prediction_candidate, gold_alias) pair and returns the maximum score.
    
    Args:
        pred: Prediction string (may contain multiple candidates)
        golds: List of gold answer strings
        
    Returns:
        The best F1 score found between any prediction candidate and any gold alias.
    """
    # Parse prediction into multiple candidates
    preds = parse_prediction(pred)
    if not preds:
        return 0.0
    
    def _compute_single_pair_f1(p_str: str, g_str: str) -> float:
        p_tokens = set(qa_normalize_answer(p_str).split())
        g_tokens = set(qa_normalize_answer(str(g_str)).split())
        
        if not p_tokens or not g_tokens:
            return 0.0
            
        common = len(p_tokens & g_tokens)
        if common == 0:
            return 0.0
            
        precision = common / len(p_tokens)
        recall = common / len(g_tokens)
        
        return 2 * precision * recall / (precision + recall)

    # Calculate global max F1
    best_f1 = 0.0
    for p in preds:
        for g in golds or []:
            f1 = _compute_single_pair_f1(p, g)
            if f1 > best_f1:
                best_f1 = f1
                
    return best_f1

f1_score = token_f1_score
