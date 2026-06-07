"""ComplexWebQuestions (CWQ) dataset loader.

Loads data from rmanluo/RoG-cwq format with support for composition answers.
"""
from __future__ import annotations
import json
from typing import List, Dict, Any, Optional


def load_cwq(path: str, split: str = "dev") -> List[Dict[str, Any]]:
    """Load CWQ dataset.
    
    Args:
        path: Path to CWQ JSON file
        split: Dataset split (train/dev/test)
    
    Returns:
        List of examples with standardized format containing only fields
        that are used by the evaluation pipeline:
        - id: Question ID
        - question: Question text
        - answers: List of answer strings
        - composition_answer: The composition answer (if available)
        - topic_entity: optional mapping of topic entities
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    out = []
    for i, ex in enumerate(data):
        # Extract question
        q = ex.get("question") or ex.get("webqsp_question") or ex.get("machine_question") or ""
        
        # Extract question ID
        qid = ex.get("ID") or ex.get("id") or f"cwq_{i}"
        # Extract answers (support both 'answers' list and legacy single 'answer')
        answers_raw = ex.get("answers")
        if answers_raw is None:
            # fall back to singular 'answer' field used in some exports
            answers_raw = ex.get("answer")

        golds: List[str] = []

        if isinstance(answers_raw, list):
            for a in answers_raw:
                if isinstance(a, dict):
                    val = a.get("answer") or a.get("answer_id") or a.get("kb_id")
                    if val:
                        golds.append(str(val))
                else:
                    golds.append(str(a))
        elif isinstance(answers_raw, dict):
            # single dict with answer info
            val = answers_raw.get("answer") or answers_raw.get("answer_id") or answers_raw.get("kb_id")
            if val:
                golds = [str(val)]
        elif isinstance(answers_raw, str):
            golds = [answers_raw]
        
        # Add composition answer if present
        comp = ex.get("composition_answer")
        if comp:
            golds.append(str(comp))
        
        golds = [g for g in golds if g]
        
        # Optional topic entities mapping: {mid: name} or {name: mid}
        topic_entity = ex.get("topic_entity") if isinstance(ex.get("topic_entity"), dict) else None

        out.append({
            "id": qid,
            "question": q,
            "answers": golds,
            "composition_answer": comp,
            "topic_entity": topic_entity,
        })
    
    return out

