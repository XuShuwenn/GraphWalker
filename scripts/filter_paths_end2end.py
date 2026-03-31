#!/usr/bin/env python3
"""End-to-end path filtering.

This script integrates the filtering logic from:
- scripts/filter_paths_by_name_path.py
- scripts/filter_by_pred_and_entity_blacklist.py

Supported input formats:
- JSON array (stream-parsed; suitable for large files)
- JSONL / ndjson (one JSON object per line)

Filters (configurable):
1) name_path structure filters
   - hop count in a given range (or exact hop count)
   - no duplicate entity names
   - no duplicate relation names
   - entity components cannot start with "m." or "g."
2) predicate/entity blacklist filters
   - any predicate URI in blacklist appears in path relations or step relation_uri/relation_name
   - any entity name (normalized lowercase) appears in entity blacklist
   - any step node_uri starting with "m." or "g." (defensive)

Typical usage (3-5 hop paths):
  python3 scripts/filter_paths_end2end.py \
    --input kgqa_agent/data/3-5hop/test_500k/paths/st128.json \
    --min-hops 3 --max-hops 5 \
    --pred datasets/cwq/cwq_predicates/black_list.json \
    --entity-script scripts/check_blacklist_entities.py
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple


SEPARATORS = ["->", "→", "-->"]


def split_name_path_from_string(s: str) -> List[str]:
    for sep in SEPARATORS:
        s = s.replace(sep, "->")
    return [p.strip() for p in s.split("->") if p.strip()]


def extract_name_path_components(item: Any) -> Optional[List[str]]:
    candidates = [
        "name_path",
        "namePath",
        "path",
        "name_path_str",
        "names",
        "namepath",
    ]
    if isinstance(item, dict):
        for k in candidates:
            if k in item and item[k] is not None:
                v = item[k]
                if isinstance(v, list):
                    return [str(x).strip() for x in v if str(x).strip()]
                if isinstance(v, str):
                    return split_name_path_from_string(v)
        for v in item.values():
            if isinstance(v, str) and ("->" in v or "→" in v):
                return split_name_path_from_string(v)
    elif isinstance(item, str):
        return split_name_path_from_string(item)
    return None



def extract_name_paths(item: Any) -> List[List[str]]:
    paths = []
    # Check for conjunctions first
    if isinstance(item, dict):
        if "name_path1" in item and "name_path2" in item:
            p1 = split_name_path_from_string(str(item["name_path1"]))
            p2 = split_name_path_from_string(str(item["name_path2"]))
            return [p1, p2]

    # Fallback to single path search
    single = extract_name_path_components(item)
    if single:
        return [single]
    return []

def compute_hops_from_components(components: List[str]) -> int:
    if not components:
        return 0
    return (len(components) - 1) // 2


def normalize(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def load_predicate_blacklist(path: str) -> Set[str]:
    p = Path(path)
    if not p.exists():
        print(f"Predicate blacklist not found: {path}", file=sys.stderr)
        sys.exit(2)
    text = p.read_text(encoding="utf-8")
    if text.strip().startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
    try:
        data = json.loads(text)
    except Exception as e:
        print(f"Failed to parse predicate blacklist JSON: {e}", file=sys.stderr)
        sys.exit(3)
    if not isinstance(data, list):
        print("Predicate blacklist should be a JSON array", file=sys.stderr)
        sys.exit(4)
    return set(map(str, data))


def load_entity_blacklist_from_script(script_path: str) -> Set[str]:
    p = Path(script_path)
    if not p.exists():
        print(f"Entity blacklist script not found: {script_path}", file=sys.stderr)
        sys.exit(2)
    try:
        mod = runpy.run_path(str(p))
    except Exception as e:
        print(f"Failed to run {script_path}: {e}", file=sys.stderr)
        sys.exit(3)

    if "BLACKLIST_LOWER" in mod:
        bl = mod["BLACKLIST_LOWER"]
    elif "BLACKLIST" in mod:
        bl = {str(s).lower() for s in mod["BLACKLIST"]}
    else:
        print("No BLACKLIST or BLACKLIST_LOWER found in script", file=sys.stderr)
        sys.exit(4)

    return {str(x).strip().lower() for x in bl}


def iter_jsonl(path: str) -> Iterator[Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def iter_json_array_stream(path: str, chunk_size: int = 1 << 20) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        idx = 0

        def refill() -> bool:
            nonlocal buf
            more = f.read(chunk_size)
            if not more:
                return False
            buf += more
            return True

        # Find '['
        while True:
            if idx >= len(buf) and not refill():
                raise ValueError("Unexpected EOF while looking for '['")
            while idx < len(buf) and buf[idx].isspace():
                idx += 1
            if idx < len(buf):
                if buf[idx] != "[":
                    raise ValueError("Input is not a JSON array")
                idx += 1
                break

        while True:
            # Skip whitespace and commas
            while True:
                if idx >= len(buf):
                    if not refill():
                        raise ValueError("Unexpected EOF in JSON array")
                while idx < len(buf) and buf[idx].isspace():
                    idx += 1
                if idx < len(buf) and buf[idx] == ",":
                    idx += 1
                    continue
                break

            if idx >= len(buf):
                continue

            if buf[idx] == "]":
                return

            while True:
                try:
                    obj, end = decoder.raw_decode(buf, idx)
                    idx = end
                    yield obj
                    if idx > chunk_size:
                        buf = buf[idx:]
                        idx = 0
                    break
                except json.JSONDecodeError:
                    if not refill():
                        raise


def iter_items(path: str) -> Iterator[Any]:
    with open(path, "r", encoding="utf-8") as f:
        first = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if ch.isspace():
                continue
            first = ch
            break

    if first == "[":
        return iter_json_array_stream(path)
    if first == "{":
        def _one() -> Iterator[Any]:
            with open(path, "r", encoding="utf-8") as f2:
                yield json.load(f2)
        return _one()

    return iter_jsonl(path)


def detect_input_format(path: str) -> str:
    """Return 'json' for JSON array/object, 'jsonl' for ndjson."""
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            ch = f.read(1)
            if not ch:
                break
            if ch.isspace():
                continue
            if ch == '[' or ch == '{':
                return 'json'
            return 'jsonl'
    return 'json'


def name_path_filters(
    components: List[str],
    require_hops: Optional[int],
    min_hops: Optional[int],
    max_hops: Optional[int],
) -> Tuple[bool, Optional[str]]:
    hops = compute_hops_from_components(components)
    if require_hops is not None and hops != require_hops:
        return False, "hops_not_equal"
    if min_hops is not None and hops < min_hops:
        return False, "hops_too_short"
    if max_hops is not None and hops > max_hops:
        return False, "hops_too_long"

    entities = [components[i] for i in range(0, len(components), 2)]
    relations = [components[i] for i in range(1, len(components), 2)]
    norm_entities = [normalize(e) for e in entities]
    if len(norm_entities) != len(set(norm_entities)):
        return False, "duplicate_entities"
    norm_relations = [normalize(r) for r in relations]
    if len(norm_relations) != len(set(norm_relations)):
        return False, "duplicate_relations"

    for e in entities:
        if isinstance(e, str) and (e.startswith("m.") or e.startswith("g.")):
            return False, "entity_mid_prefix_in_name_path"

    return True, None


def blacklist_filters(
    item: Dict[str, Any],
    components: Optional[List[str]],
    pred_bl: Optional[Set[str]],
    ent_bl: Optional[Set[str]],
) -> Tuple[bool, Optional[str]]:
    if components:
        if pred_bl:
            relations = [components[i] for i in range(1, len(components), 2)]
            for r in relations:
                if isinstance(r, str) and r in pred_bl:
                    return False, "predicate_blacklisted_in_name_path"
        if ent_bl:
            entities = [components[i] for i in range(0, len(components), 2)]
            for e in entities:
                en = normalize(e)
                if en and en in ent_bl:
                    return False, "entity_blacklisted_in_name_path"

    lists_to_check: List[List[Any]] = []
    if isinstance(item.get("steps"), list):
        lists_to_check.append(item["steps"])
    if isinstance(item.get("query_steps"), list):
        lists_to_check.append(item["query_steps"])
    if isinstance(item.get("path_nodes"), list):
        lists_to_check.append(item["path_nodes"])

    for lst in lists_to_check:
        for step in lst:
            if not isinstance(step, dict):
                continue

            if pred_bl:
                rel_uri = step.get("relation_uri") or step.get("relation")
                if isinstance(rel_uri, str) and rel_uri in pred_bl:
                    return False, "predicate_blacklisted_in_steps"

                rel_name = normalize(step.get("relation_name"))
                if rel_name and rel_name in pred_bl:
                    return False, "predicate_blacklisted_in_steps"

            if ent_bl:
                node_name = normalize(step.get("node_name") or step.get("node"))
                if node_name and node_name in ent_bl:
                    return False, "entity_blacklisted_in_steps"

            node_uri = step.get("node_uri")
            if isinstance(node_uri, str) and (node_uri.startswith("m.") or node_uri.startswith("g.")):
                return False, "mid_node_uri_in_steps"

    return True, None


def write_output(out_path: str, output_format: str, items: Iterable[Any], pretty_json: bool = False) -> int:
    count = 0
    if output_format == "jsonl":
        with open(out_path, "w", encoding="utf-8") as out_f:
            for item in items:
                out_f.write(json.dumps(item, ensure_ascii=False))
                out_f.write("\n")
                count += 1
        return count

    # output_format == 'json'
    if pretty_json:
        # collect items and write a pretty-printed JSON array (indent=2)
        list_items = list(items)
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(list_items, out_f, ensure_ascii=False, indent=2)
            out_f.write("\n")
        return len(list_items)

    # fallback: streaming compact array (preserves previous behavior)
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("[\n")
        first = True
        for item in items:
            if not first:
                out_f.write(",\n")
            json.dump(item, out_f, ensure_ascii=False)
            first = False
            count += 1
        out_f.write("\n]\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview", action="store_true", help="only print stats")

    parser.add_argument("--require-hops", type=int, default=None)
    parser.add_argument("--min-hops", type=int, default=None)
    parser.add_argument("--max-hops", type=int, default=None)

    parser.add_argument("--pred", default=None, help="predicate blacklist JSON file")
    parser.add_argument("--entity-script", default=None, help="entity blacklist script path")

    parser.add_argument("--output-format", choices=["json", "jsonl"], default="json")
    parser.add_argument("--report", default=None, help="write stats JSON to this path")

    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    out_path = args.output
    if not out_path:
        suffix = ".filtered.jsonl" if args.output_format == "jsonl" else ".filtered.json"
        out_path = str(inp) + suffix

    pred_bl: Optional[Set[str]] = None
    ent_bl: Optional[Set[str]] = None
    if args.pred is not None:
        pred_bl = load_predicate_blacklist(args.pred)
    if args.entity_script is not None:
        ent_bl = load_entity_blacklist_from_script(args.entity_script)

    print("Config:")
    print(f"  input: {inp}")
    print(f"  output: {out_path}")
    print(f"  hops: require={args.require_hops}, min={args.min_hops}, max={args.max_hops}")
    print(f"  pred blacklist: {len(pred_bl) if pred_bl else 0}")
    print(f"  entity blacklist: {len(ent_bl) if ent_bl else 0}")
    print(f"  output format: {args.output_format}")

    stats = Counter()
    drop_reasons = Counter()

    def kept_items() -> Iterator[Any]:
        for item in iter_items(str(inp)):
            stats["total"] += 1
            if not isinstance(item, dict):
                drop_reasons["non_dict_item"] += 1
                continue

            name_paths = extract_name_paths(item)
            if not name_paths:
                drop_reasons["missing_name_path"] += 1
                continue

            rejected = False
            all_path_entities = []
            
            for components in name_paths:
                ok, reason = name_path_filters(
                    components,
                    require_hops=args.require_hops,
                    min_hops=args.min_hops,
                    max_hops=args.max_hops,
                )
                if not ok:
                    drop_reasons[reason or "name_path_rejected"] += 1
                    rejected = True
                    break
                # Collect entities for cross-path check
                p_ents = [normalize(components[i]) for i in range(0, len(components), 2)]
                all_path_entities.append(set(p_ents))

            if rejected:
                continue

            # Cross-path duplicate entity check (for conjunctions)
            # If we have multiple paths (e.g. 2 for conjunction), check intersection of entities
            # EXCEPT for the topic entity (usually start) and answer entity (usually end)? 
            # Wait, usually standard conjunctions share the ANSWER entity (target).
            # And they start from DIFFERENT topic entities. 
            # If they share an intermediate entity, is that bad?
            # 
            # User request: "两条子路径共享实体而应当拒绝"
            # However, for QA conjunctions: p1: Topic1 -> ... -> Answer, p2: Topic2 -> ... -> Answer
            # They definitely share the Answer node. We should NOT reject if they share ONLY the Answer node.
            # 
            # Typical structure for len=2 paths:
            #   Path1: [Topic1, Rel1, Answer]
            #   Path2: [Topic2, Rel2, Answer]
            # They share 'Answer'.
            # 
            # If paths are longer, e.g. 2 hops:
            #   Path1: [Topic1, R1, Mid1, R2, Answer]
            #   Path2: [Topic2, R3, Mid2, R4, Answer]
            # They share Answer. If Mid1 == Mid2, that might be a problem (diamond shape? or basically same path?)
            # 
            # Let's count occurrences across all paths. 
            # Actually, standard valid conjunctions SHOULD share the answer entity.
            # If the user says "检测跨 path1/path2 的重复实体...而应当拒绝", usually implies unintended overlap.
            # But the answer node IS an overlap.
            # 
            # Heuristic: Check intersection. If intersection size > 1, imply sharing more than just the answer?
            # Or if sharing occurs at non-answer positions?
            # 
            # Let's implement a strict check: "Duplicate entites across paths IMPLIES rejection"
            # BUT we must exempt the ANSWER entity if it's a conjunction.
            # 
            # Assumption for "name_path":
            #   The LAST entity in the component list is the Answer/Tail.
            #   The FIRST entity is the Topic/Head.
            #
            # Let's get the set of all entities across all paths.
            # If we simply check len(union) vs sum(len), we catch all duplicates.
            # 
            # If item has 'name_path1' and 'name_path2', it is a conjunction.
            # They intersect at the answer.
            # Implementation:
            #   Gather all entities from all paths.
            #   Count frequencies of each normalized entity.
            #   If any entity appears in > 1 path:
            #      If it is the "answer" (last node of both paths), it is allowed (count == #paths).
            #      Any other sharing is disallowed?
            #
            # Let's try:
            #   For each path, get list of entities.
            #   Verify that the LAST entity of all paths is the same (the answer).
            #   Verify that no OTHER entity is shared between paths.
            #   Also verify no internal duplicates (already done by name_path_filters).
            
            if len(name_paths) > 1:
                # Assuming all paths end at the same answer for valid conjunction
                # And we want to ban other overlaps.
                
                # Get entities per path
                # components list: [E1, R1, E2, R2, E3 ...]
                path_ent_lists = [[normalize(comp[i]) for i in range(0, len(comp), 2)] for comp in name_paths]
                
                # Check if they share the answer (last entity)
                last_ents = [p[-1] for p in path_ent_lists if p]
                if len(set(last_ents)) > 1:
                    # Paths end at different entities? For CoG/Conjunction, usually they end at same.
                    # If they differ, it's not a standard conjunction answer path set. 
                    # But we only want to filter duplicates. 
                    pass
                
                # Check for non-answer overlaps
                # We collect all entities EXCEPT the last one from each path
                non_answer_entities = []
                for p_ents in path_ent_lists:
                    if len(p_ents) > 1:
                        non_answer_entities.extend(p_ents[:-1])
                    else:
                        # Path has only 1 entity? (0 hops)
                        non_answer_entities.extend(p_ents)
                
                # If any entity in non_answer_entities appears in 'other' paths...
                # Actually, simpler: 
                #   Union of all entities across paths.
                #   Total count of entities.
                #   Allowed overlap = (num_paths - 1) if they all share 1 answer node.
                # 
                # Let's go with: No shared entities allowed EXCEPT the last entity of each path IF they are equal.
                
                # Algorithm:
                # 1. Identify Answer Node = Last entity of Path 1.
                # 2. If valid conjunction, Last entity of Path 2...N should match Answer Node.
                # 3. If they match, we treat Answer Node as "allowed shared".
                # 4. Any OTHER entity shared between any pair of paths -> Reject.
                
                ref_answer = path_ent_lists[0][-1] if path_ent_lists[0] else None
                
                # Check if all paths end with this ref_answer
                all_end_same = all((p and p[-1] == ref_answer) for p in path_ent_lists)
                
                ignored_common = set()
                if all_end_same and ref_answer:
                    ignored_common.add(ref_answer)
                
                # Now check for intersections excluding ignored_common
                seen_so_far = set()
                conflict = False
                for p_ents in path_ent_lists:
                    current_set = set(p_ents)
                    # intersection with previous
                    common = current_set & seen_so_far
                    # remove allowed
                    common = common - ignored_common
                    if common:
                        conflict = True
                        break
                    seen_so_far.update(current_set)
                
                if conflict:
                    drop_reasons["cross_path_duplicate_entity"] += 1
                    continue

                # Cross-path relation overlap check
                path_rel_lists = [[normalize(comp[i]) for i in range(1, len(comp), 2)] for comp in name_paths]
                seen_rels = set()
                rel_conflict = False
                for rels in path_rel_lists:
                    current_set = set(rels)
                    common = current_set & seen_rels
                    if common:
                        rel_conflict = True
                        break
                    seen_rels.update(current_set)
                if rel_conflict:
                    drop_reasons["cross_path_duplicate_relation"] += 1
                    continue


            # Check item steps/nodes
            ok, reason = blacklist_filters(item, None, pred_bl=pred_bl, ent_bl=ent_bl)
            if not ok:
                 drop_reasons[reason or "blacklist_rejected"] += 1
                 continue
            
            # Check components of all paths against blacklist
            for components in name_paths:
                ok, reason = blacklist_filters({}, components, pred_bl=pred_bl, ent_bl=ent_bl)
                if not ok:
                    drop_reasons[reason or "blacklist_rejected"] += 1
                    rejected = True
                    break
            if rejected:
                continue

            stats["kept"] += 1
            yield item

    if args.preview:
        for _ in kept_items():
            pass
        print(f"Total: {stats['total']}, Kept: {stats['kept']}, Dropped: {stats['total'] - stats['kept']}")
        print("Dropped by reason:")
        for k, v in drop_reasons.most_common():
            print(f"  {k}: {v}")
        return

    if args.overwrite:
        bak = str(inp) + ".bak"
        if not Path(bak).exists():
            os.replace(str(inp), bak)
        out_path = str(inp)

    input_fmt = detect_input_format(str(inp))
    pretty_json = (args.output_format == 'json' and input_fmt == 'json')
    wrote = write_output(out_path, args.output_format, kept_items(), pretty_json=pretty_json)

    print(f"Total: {stats['total']}, Kept: {stats['kept']}, Dropped: {stats['total'] - stats['kept']}")
    print(f"Wrote {wrote} items to {out_path}")

    print("Dropped by reason:")
    for k, v in drop_reasons.most_common():
        print(f"  {k}: {v}")

    if args.report:
        report_obj = {
            "input": str(inp),
            "output": out_path,
            "total": int(stats["total"]),
            "kept": int(stats["kept"]),
            "dropped": int(stats["total"] - stats["kept"]),
            "drop_reasons": dict(drop_reasons),
            "config": {
                "require_hops": args.require_hops,
                "min_hops": args.min_hops,
                "max_hops": args.max_hops,
                "pred": args.pred,
                "entity_script": args.entity_script,
                "output_format": args.output_format,
                "overwrite": bool(args.overwrite),
            },
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("Wrote report to", args.report)


if __name__ == "__main__":
    main()
