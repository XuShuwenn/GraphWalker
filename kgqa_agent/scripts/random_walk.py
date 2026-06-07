#!/usr/bin/env python3
"""CLI entry point for random walk generation.

Usage:
    python kgqa_agent/scripts/random_walk.py --config <path-to-yaml>

Generates random walks from a knowledge graph based on YAML configuration,
saves results as JSON files with full path details and name-only versions.
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

from kgqa_agent.src.data_gen.random_walk import run_random_walk_from_config, save_paths


def main():
    """Parse config, run random walk generation, save results to single JSON file."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='YAML config in the same schema as original random_walk.py')
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
    base_name = os.path.splitext(str(cfg_file_name))[0] if cfg_file_name else 'random_walk_paths'
    out_file = os.path.join(out_dir, f'{base_name}.json')
    out_names = os.path.join(out_dir, f'{base_name}_names.json')

    paths = run_random_walk_from_config(
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
