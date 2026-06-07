#!/usr/bin/env python3
"""CLI entry point for conjunction-type path generation.

Usage:
    python kgqa_agent/scripts/conjunction_walk.py --config <path-to-yaml>

Generates conjunction-type paths from a knowledge graph based on YAML configuration,
where two different entities connect to the same target entity via different relations:
  entity1 -> relationA -> entity2
  entity3 -> relationB -> entity2

Saves results as JSON files with full path details and name-only versions.
"""
from __future__ import annotations
import argparse
import os
import sys
import pathlib

# Ensure repo root is on sys.path so `import kgqa_agent...` works when running the script directly
repo_root = str(pathlib.Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from kgqa_agent.src.data_gen.conjunction_walk import run_conjunction_walk_from_config


def main():
    """Parse config, run conjunction walk generation, save results to single JSON file."""
    parser = argparse.ArgumentParser(
        description="Generate conjunction-type paths from knowledge graph"
    )
    parser.add_argument('--config', required=True, help='YAML config file')
    parser.add_argument('--out-dir', default=None, help='Override directory to write outputs (optional)')
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    output_cfg = cfg.get('output') or {}
    cfg_out_dir = output_cfg.get('dir')
    out_dir = args.out_dir or cfg_out_dir or 'kgqa_agent/data/paths'
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(repo_root, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg_file_name = output_cfg.get('file_name')
    base_name = os.path.splitext(str(cfg_file_name))[0] if cfg_file_name else 'conjunction_walk_paths'
    out_file = os.path.join(out_dir, f'{base_name}.json')
    out_names = os.path.join(out_dir, f'{base_name}_names.json')

    paths = run_conjunction_walk_from_config(
        cfg,
        cfg_path=args.config,
        save_every=20,
        resume=True,
        checkpoint_out_file=out_file,
        checkpoint_out_names=out_names,
    )
    # paths already saved incrementally with resume; still print a summary
    print(f"wrote {len(paths)} paths (incremental, every 20) -> {out_file}\nname-only -> {out_names}")


if __name__ == '__main__':
    main()

