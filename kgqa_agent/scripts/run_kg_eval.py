"""Standardized KG evaluation runner (API/Local) with optional KG client self-check.

Aligns CLI with src/eval/run_eval.py, supports CWQ/WebQSP/GrailQA datasets,
API or Local model backends, plus optional KG self-check and summary file.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import argparse
from datetime import datetime

from kgqa_agent.src.eval.evaluator import run_evaluation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_kg_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KG Evaluation Runner (standardized)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--dataset-type", type=str, required=True,
                        choices=["cwq", "webqsp", "grailqa", "simpleqa"], help="Dataset type")
    parser.add_argument("--model-config", type=str, required=True, help="Model config JSON string (must include 'model' and optionally 'base_url')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker threads (each thread processes one example)")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--output-path", type=str, default=None, help="Explicit output json path (enables resume)")
    parser.add_argument("--kg-server-url", type=str, default="http://localhost:18890",
                        help="Base URL of the KG server (Virtuoso host/port; SPARQL suffix added if needed)")
    parser.add_argument("--max-calls", type=int, default=3, help="Max KG calls per question")
    parser.add_argument("--kg-top-k", type=int, default=10, help="Top-K KG results to return after ranking")
    parser.add_argument("--full-list", action="store_true", default=False,
                        help="Use merged full whitelist instead of dataset-specific whitelist")

    return parser.parse_args()


def save_human_summary(out_path: Path, summary_dir: Path) -> Optional[Path]:
    """Write a brief, human-readable summary next to the JSON result."""
    if not out_path.exists():
        return None
    with open(out_path, 'r', encoding='utf-8') as f:
        saved = json.load(f)
    metrics = saved.get('metrics', {})
    preds = saved.get('predictions', [])
    kg_stats = saved.get('kg_query_stats', {}) or {}
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"summary_{ts}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("KG Evaluation Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Out file: {out_path}\n\n")
        f.write("Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
    return summary_path


def main():
    args = parse_args()

    # parse model_config
    try:
        model_cfg = json.loads(args.model_config)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in --model-config: {args.model_config}")
        logger.error(f"Error: {e}")
        return

    # 
    results = run_evaluation(
        dataset_path=args.dataset,
        dataset_type=args.dataset_type,
        model_cfg=model_cfg,
        limit=args.limit,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        output_path=args.output_path,
        kg_server_url=args.kg_server_url,
        max_calls=args.max_calls,
        kg_top_k=args.kg_top_k,
        full_list=args.full_list,
    )

    out_path = Path(results.get("out_path")) if isinstance(results, dict) and results.get("out_path") else None
    if out_path:
        summary_dir = Path(args.output_dir)
        try:
            summary_file = save_human_summary(out_path, summary_dir)
            if summary_file:
                print(f"Summary saved to: {summary_file}")
        except Exception as e:
            logger.warning(f"Failed to generate summary file: {e}")


if __name__ == "__main__":
    main()
