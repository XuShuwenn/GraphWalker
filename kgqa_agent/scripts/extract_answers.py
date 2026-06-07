#!/usr/bin/env python3
"""Extract answers from the last step's results based on the last triple in name_path."""
from __future__ import annotations

import json
import re
import argparse
from typing import Any, Dict, List, Tuple
from tqdm import tqdm


def parse_triple_string(triple_str: str) -> Tuple[str, str, str] | None:
    """Parse a triple string like '[head, relation, tail]' into (head, relation, tail).
    
    Deprecated: Prefer regex-based validation in find_matching_triples for better robustness.
    """
    # Remove brackets and strip
    triple_str = triple_str.strip()
    if not (triple_str.startswith('[') and triple_str.endswith(']')):
        return None
    
    # Remove outer brackets
    content = triple_str[1:-1].strip()
    
    # Split by comma, but be careful with commas inside entity names
    # Simple approach: split by ', ' (comma followed by space)
    parts = [p.strip() for p in content.split(', ')]
    if len(parts) != 3:
        return None
    
    return tuple(parts)


def find_matching_triples(
    results: List[str],
    target_entity: str,
    target_relation: str
) -> List[str]:
    """Find all triples in results that match the golden triple direction.
    
    The golden triple is [target_entity, target_relation, answer].
    Uses regex matching to strictly validate the prefix [target_entity, target_relation, ...]
    to unambiguously handle cases where entities contain commas.
    """
    answers = []
    
    # Construct regex pattern to match: [target_entity, target_relation, (capture_tail)]
    # We escape target_entity and target_relation to handle special chars.
    # We use \\s* after commas to allow flexible spacing (typically ", ").
    pattern_str = (
        r"^\[" 
        + re.escape(target_entity.strip()) 
        + r",\s*" 
        + re.escape(target_relation.strip()) 
        + r",\s*(.*)\]$"
    )
    pattern = re.compile(pattern_str)
    
    for result_str in results:
        result_str = result_str.strip()
        match = pattern.match(result_str)
        if match:
            # Extract the tail (answer) from the capturing group
            tail = match.group(1).strip()
            answers.append(tail)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_answers = []
    for ans in answers:
        if ans not in seen:
            seen.add(ans)
            unique_answers.append(ans)
    
    return unique_answers


def extract_last_triple_from_name_path(name_path: str) -> Tuple[str, str, str] | None:
    """Extract the last triple (entity, relation, entity) from name_path.
    
    Example: "A -> r1 -> B -> r2 -> C" -> (B, r2, C)
    """
    if not name_path or not isinstance(name_path, str):
        return None
    
    parts = [p.strip() for p in name_path.split('->')]
    if len(parts) < 3:
        return None
    
    # Last triple: [second-to-last entity, last relation, last entity]
    if len(parts) >= 3:
        head = parts[-3]
        relation = parts[-2]
        tail = parts[-1]
        return (head, relation, tail)
    
    return None


def extract_first_entity_from_name_path(name_path: str) -> str | None:
    """Extract the first entity from name_path.
    
    Example: "A -> r1 -> B -> r2 -> C" -> "A"
    """
    if not name_path or not isinstance(name_path, str):
        return None
    
    parts = [p.strip() for p in name_path.split('->')]
    if not parts:
        return None
    
    return parts[0]


def extract_topic_entity(item: Dict[str, Any], step_index: int = 0) -> Dict[str, str] | None:
    """Extract topic_entity from the specified step (default index=0).
    
    Returns a dictionary with entity_id as key and entity_name as value.
    Format: {"m.0d_rw": "The Twilight Zone"}
    """
    steps = item.get("steps", [])
    if not steps:
        return None
    
    # Find the target step
    target_step = None
    for step in steps:
        if int(step.get("step_index", -1)) == step_index:
            target_step = step
            break
    
    if target_step is None:
        return None
    
    # Extract entity_id from args and entity_name from current
    args = target_step.get("args", {})
    entity_id = args.get("entity")
    entity_name = target_step.get("current")
    
    if entity_id and entity_name:
        return {str(entity_id): str(entity_name)}
    
    return None


def check_topic_entity_consistency(topic_entity: Dict[str, str] | None, name_path: str) -> bool:
    """Check if topic_entity name matches the first entity in name_path.
    
    Returns True if consistent, False otherwise.
    """
    if not topic_entity or not name_path:
        return False
    
    # Extract entity name from topic_entity (prefer value, fallback to key if no value)
    topic_entity_name = list(topic_entity.values())[0] if topic_entity else None
    if not topic_entity_name:
        return False
    
    # Extract first entity from name_path
    first_entity = extract_first_entity_from_name_path(name_path)
    if not first_entity:
        return False
    
    # Compare (case-sensitive exact match)
    return str(topic_entity_name).strip() == str(first_entity).strip()


def find_step_results(steps: List[Dict[str, Any]], target_entity: str) -> List[str]:
    """Find results from the last step query for a target entity (head of last triple).
    
    We look for a 'get_triples' step whose 'current' entity matches our target_entity.
    We prefer the one with the highest step_index if multiple exist.
    """
    last_results: List[str] = []
    max_idx = -1
    
    normalized_target = target_entity.strip()
    
    for step in steps:
        if step.get("query_type") != "get_triples":
            continue
        
        current = str(step.get("current", "")).strip()
        # Loose match: entity name match. But ideally, we should rely on step flow.
        # Since we don't have explicit linking, we match by entity name.
        if current == normalized_target:
            idx = int(step.get("step_index", -1))
            res = step.get("results", [])
            if res and idx > max_idx:
                max_idx = idx
                last_results = res
                
    return last_results


def process_item_comp(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single item (composition type) and add answers and topic_entity fields."""
    name_path = item.get("name_path", "")
    steps = item.get("steps", [])
    
    # Extract topic_entity from first step (index 0)
    topic_entity = extract_topic_entity(item, step_index=0)
    if topic_entity:
        item["topic_entity"] = topic_entity
    
    # Extract last triple from name_path
    last_triple = extract_last_triple_from_name_path(name_path)
    if last_triple is None:
        item["answers"] = []
        return item
    
    head_entity, relation, tail_entity = last_triple
    
    # Find results for the step corresponding to head_entity
    # Since steps are linear in comp, we can usually just take the LAST step results if it matches
    # But explicitly searching by head_entity is safer
    last_step = None
    max_step_index = -1
    for step in steps:
        step_idx = int(step.get("step_index", -1))
        results = step.get("results", [])
        if results and step_idx > max_step_index:
            max_step_index = step_idx
            last_step = step
    
    if last_step is None:
        item["answers"] = []
        return item
        
    results = last_step.get("results", [])
    
    # Find matching triples with direction consistency
    answers = find_matching_triples(results, head_entity, relation)
    
    item["answers"] = answers
    return item


def process_item_conj(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single item (conjunction type) and add answers and topic_entities fields."""
    name_path1 = item.get("name_path1", "")
    name_path2 = item.get("name_path2", "")
    steps = item.get("steps", [])
    
    if not name_path1 or not name_path2:
        item["answers"] = []
        return item

    # 1. Extract Last Triples for both paths
    triple1 = extract_last_triple_from_name_path(name_path1)
    triple2 = extract_last_triple_from_name_path(name_path2)
    
    if not triple1 or not triple2:
        item["answers"] = []
        return item
    
    head1, rel1, ans1 = triple1
    head2, rel2, ans2 = triple2
    
    # 2. Extract Answers for Path 1
    # We need to find the step that queried triples for head1
    results1 = find_step_results(steps, head1)
    extracted1 = find_matching_triples(results1, head1, rel1)
    
    # 3. Extract Answers for Path 2
    # We need to find the step that queried triples for head2
    results2 = find_step_results(steps, head2)
    extracted2 = find_matching_triples(results2, head2, rel2)
    
    # 4. Conjunction Logic: Intersection
    # We keep an answer only if it appears in BOTH lists
    # Note: entity names must match exactly (case-sensitive strip)
    set1 = set(extracted1)
    set2 = set(extracted2)
    common_answers = list(set1.intersection(set2))
    
    item["answers"] = common_answers
    
    # 5. Extract Topic Entities (start of each path)
    # Conjunction paths start from TWO different entities.
    # In synthesize logic, usually step 0 corresponds to one path head, and step 2 (or 1) to the other.
    # We just scan steps to find start entities matching name_path1 start and name_path2 start.
    
    start1 = extract_first_entity_from_name_path(name_path1) or ""
    start2 = extract_first_entity_from_name_path(name_path2) or ""
    
    topic_entities = {}
    
    # Scan all steps to find topic entities
    # We look for steps where 'current' matches start entity
    for step in steps:
        current = str(step.get("current", "")).strip()
        args = step.get("args", {})
        eid = str(args.get("entity", ""))
        
        if not eid: 
            continue
            
        if current == start1.strip():
            topic_entities[eid] = current
        elif current == start2.strip():
            topic_entities[eid] = current
            
    if topic_entities:
        item["topic_entities"] = topic_entities
        
    return item


def check_consistency_comp(item: Dict[str, Any]) -> bool:
    """Check consistency for composition items (single topic_entity match)."""
    return check_topic_entity_consistency(item.get("topic_entity"), item.get("name_path"))


def check_consistency_conj(item: Dict[str, Any]) -> bool:
    """Check consistency for conjunction items (two topic_entities match starts)."""
    name_path1 = item.get("name_path1", "")
    name_path2 = item.get("name_path2", "")
    topic_entities = item.get("topic_entities", {})
    
    start1 = extract_first_entity_from_name_path(name_path1) or ""
    start2 = extract_first_entity_from_name_path(name_path2) or ""
    
    if not start1 or not start2 or not topic_entities:
        return False
    
    # Check coverage: do we have topic entities for both starts?
    # Values in topic_entities are names
    found_names = set(v.strip() for v in topic_entities.values())
    
    # Strict check: both start entities must be present in topic_entities values
    return (start1.strip() in found_names) and (start2.strip() in found_names)


def main():
    parser = argparse.ArgumentParser(
        description="Extract answers from tool trace steps based on golden path topology."
    )
    parser.add_argument("--in-file", required=True, help="Input JSON file path")
    parser.add_argument("--out-file", required=True, help="Output JSON file path")
    parser.add_argument("--data-type", choices=["comp", "conj"], default="comp", help="Data type: 'comp' (composition) or 'conj' (conjunction)")
    parser.add_argument("--backup", action="store_true", help="Create .backup of input file before run")
    args = parser.parse_args()
    
    print(f"Loading {args.in_file} ...")
    with open(args.in_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list")

    if args.backup:
        bak = f"{args.in_file}.backup"
        print(f"Backing up to {bak}")
        with open(bak, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    print(f"Processing {len(data)} items as type='{args.data_type}'...")
    processed_data = []
    stats = {
        "total": 0,
        "kept": 0,
        "filtered_no_answer": 0,
        "filtered_inconsistent": 0,
    }
    
    for item in tqdm(data):
        stats["total"] += 1
        
        # Dispatch processing
        if args.data_type == "comp":
            p_item = process_item_comp(item)
            is_consistent = check_consistency_comp(p_item)
        else: # conj
            p_item = process_item_conj(item)
            is_consistent = check_consistency_conj(p_item)
            
        answers = p_item.get("answers", [])
        has_answers = len(answers) > 0
        
        if has_answers and is_consistent:
            processed_data.append(p_item)
            stats["kept"] += 1
        else:
            if not has_answers:
                stats["filtered_no_answer"] += 1
            if not is_consistent:
                stats["filtered_inconsistent"] += 1
                
    # Atomic write
    import os
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    tmp = f"{args.out_file}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, args.out_file)
    
    print("\nProcessing Complete.")
    print(f"  Total: {stats['total']}")
    print(f"  Kept:  {stats['kept']}")
    print(f"  Filtered (No Answer): {stats['filtered_no_answer']}")
    print(f"  Filtered (Inconsistent Topic Entity): {stats['filtered_inconsistent']}")
    print(f"  Output: {args.out_file}")

if __name__ == "__main__":
    main()
