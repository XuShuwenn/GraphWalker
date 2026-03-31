#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper: extract predicates per-dataset (CWQ, GrailQA) and filter them using
`scripts/filter_cvt_predicates.py` logic. Produces per-dataset JSON outputs.

Usage:
  python scripts/extract_and_filter_predicates.py

This script calls the existing scripts via import (SourceFileLoader) so we
reuse their parsing and filtering logic without duplicating code.
"""
from importlib.machinery import SourceFileLoader
from pathlib import Path
import os
import json

# Load extract module
extract_mod = SourceFileLoader(
    "extract_cwq_predicates",
    os.path.join("scripts", "extract_cwq_predicates.py")
).load_module()
# Load filter module
filter_mod = SourceFileLoader(
    "filter_cvt_predicates",
    os.path.join("scripts", "filter_cvt_predicates.py")
).load_module()

REPO_ROOT = Path(__file__).resolve().parents[1]

# Configurations for datasets to process
TASKS = [
    {
        "name": "cwq",
        "input_dir": "datasets/cwq/cwq_downloads",
        "output_dir": "datasets/cwq/cwq_predicates",
        "merged_white_list": "cwq_white_list.json",
    },
    {
        "name": "grailqa",
        "input_dir": "datasets/GrailQA_v1.0",
        "output_dir": "datasets/GrailQA_v1.0/predicates",
        "merged_white_list": "grailqa_white_list.json",
    },
]


def run_task(cfg: dict):
    print(f"\n=== Processing {cfg['name']} ===")
    # Prepare paths
    input_dir = os.path.join(str(REPO_ROOT), cfg["input_dir"])
    output_dir = os.path.join(str(REPO_ROOT), cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Build override config for extract module
    override = {
        "input_dir": cfg["input_dir"],
        "input_glob": "*.json",
        "output_dir": cfg["output_dir"],
        "merged_white_list": cfg.get("merged_white_list", "white_list.json"),
    }

    # Call extract main (it writes per-file and merged outputs)
    try:
        extract_mod.main(override)
    except Exception as e:
        print(f"Error during extraction for {cfg['name']}: {e}")
        raise

    merged_white = os.path.join(output_dir, override["merged_white_list"])
    filtered_out = os.path.join(output_dir, f"filtered_{Path(override['merged_white_list']).stem}.json")
    filtered_final = os.path.join(output_dir, f"filtered_{cfg['name']}_white_list.json")

    if not os.path.exists(merged_white):
        print(f"Merged white-list not found: {merged_white} (skipping filter)")
        return

    print(f"Filtering predicates file: {merged_white}")
    # Use filter_predicates to produce filtered file (it expects a list input and writes output)
    try:
        stats = filter_mod.filter_predicates(merged_white, filtered_final, verbose=True)
    except Exception as e:
        print(f"Error during filtering for {cfg['name']}: {e}")
        raise

    print(f"Finished {cfg['name']}: filtered -> {filtered_final}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    for t in TASKS:
        run_task(t)
    print("\nAll tasks completed.")
