#!/usr/bin/env python3
"""
Extract indices for entries whose average score >= threshold.

Usage:
  python extract_highscore_indices.py /path/to/input.json [--threshold 9.66]

Writes: <input_stem>_indices_avg_ge_<threshold>.json next to input file.
"""
import sys
import json
from pathlib import Path
import statistics


def detect_average(obj):
    for k in ("average_score", "avg_score", "average", "avg", "mean_score"):
        if k in obj and isinstance(obj[k], (int, float)):
            return float(obj[k])
    for k in ("scores", "scores_list", "ratings"):
        if k in obj and isinstance(obj[k], (list, tuple)) and obj[k]:
            try:
                nums = [float(x) for x in obj[k]]
                return statistics.mean(nums)
            except Exception:
                pass
    if "evaluated" in obj and isinstance(obj["evaluated"], (list, tuple)) and obj["evaluated"]:
        try:
            nums = []
            for e in obj["evaluated"]:
                if isinstance(e, dict):
                    for k2 in ("score", "avg_score", "rating"):
                        if k2 in e and isinstance(e[k2], (int, float)):
                            nums.append(float(e[k2])); break
                elif isinstance(e, (int, float)):
                    nums.append(float(e))
            if nums:
                return statistics.mean(nums)
        except Exception:
            pass
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_highscore_indices.py <input.json> [--threshold N]")
        sys.exit(1)
    input_path = Path(sys.argv[1])
    threshold = 9.66
    if "--threshold" in sys.argv:
        try:
            i = sys.argv.index("--threshold")
            threshold = float(sys.argv[i+1])
        except Exception:
            pass
    if not input_path.exists():
        print("Input file not found:", input_path)
        sys.exit(2)
    data = json.loads(input_path.read_text())
    if not isinstance(data, list):
        print("Expected top-level JSON array; got", type(data))
        sys.exit(3)
    out = []
    for pos, item in enumerate(data):
        avg = None
        if isinstance(item, dict):
            avg = detect_average(item)
        if avg is None and isinstance(item, (int, float)):
            avg = float(item)
        if avg is None:
            continue
        if avg >= threshold:
            idx = item.get("index", pos) if isinstance(item, dict) else pos
            out.append(idx)
    out_name = input_path.with_name(f"{input_path.stem}_indices_avg_ge_{threshold}.json")
    out_name.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote {len(out)} indices to {out_name}")


if __name__ == '__main__':
    main()
