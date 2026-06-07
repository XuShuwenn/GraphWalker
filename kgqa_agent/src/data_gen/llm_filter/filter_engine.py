"""Minimal engine for LLM-based plan + filtering over info_syn files."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import random

from ..utils.client import LLMClient
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
from threading import Lock

def candidate_key(item: Dict[str, Any]) -> str:
    """Preferred key for a candidate: id if present, else name."""
    vid = (item.get("id") or "").strip()
    vname = (item.get("name") or "").strip()
    return vid if vid else vname


def plan_and_filter_step(
    *,
    client: LLMClient,
    raw_question: str,
    name_path: str,
    step: Dict[str, Any],
    k: int,
    golden_key: str,
    prompt_template: str,
) -> Dict[str, Any]:
    """Return {"plan", "keep_keys"} for one step via the LLM client."""
    results: List[str] = step.get("results") or []
    qtype = step.get("query_type") or ""
    is_relation_step = "relations" in qtype

    if is_relation_step:
        candidates = [{"id": str(x)} for x in results]
        golden_obj = {"id": golden_key}
    else:
        candidates = [{"name": str(x)} for x in results]
        golden_obj = {"name": golden_key}

    out = client.plan_and_filter(
        raw_question=raw_question,
        name_path=name_path,
        step={key: step[key] for key in ("step_index", "query_type", "current", "args") if key in step},
        candidates=candidates,
        golden=golden_obj,
        k=k,
        prompt_template=prompt_template,
    )
    return {"plan": out.get("plan", ""), "keep_keys": out.get("keep_keys", [])}


def apply_plan_and_truncate_step(
    *,
    step: Dict[str, Any],
    plan: str,
    keep_keys: List[str],
) -> Dict[str, Any]:
    """Attach plan; keep_keys follow candidate order; results are kept keys shuffled (seed=42)."""
    orig = [str(v) for v in (step.get("results") or [])]
    keyset = set(keep_keys)
    # keep_keys: reorder to follow the original candidate order
    keep_in_candidate_order = [v for v in orig if v in keyset]
    # results: same kept values but randomly shuffled with a fixed seed
    shuffled = list(keep_in_candidate_order)
    random.Random(42).shuffle(shuffled)

    new_step = dict(step)
    new_step["plan"] = plan
    new_step["keep_keys"] = keep_in_candidate_order
    new_step["results"] = shuffled
    return new_step


def process_file(
    *,
    client: LLMClient,
    in_file: str,
    out_file: str,
    k: int = 5,
    prompt_template: str | None = None,
    workers: int = 4,
    save_every: int = 10,
    resume: bool = False,
) -> None:
    """Process an info_syn JSON file and write a filtered copy with:
    - Parallel per-path processing (workers threads)
    - Periodic checkpoint writes every `save_every` processed paths
    - Resume by path_id (skip those already in out_file)
    """
    from kgqa_agent.prompts.info_filter_prompt import (
        TRAJECTORY_PLAN_AND_FILTER_PROMPT,
    )

    tmpl = prompt_template or TRAJECTORY_PLAN_AND_FILTER_PROMPT

    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    def parse_tokens(path_str: str) -> List[str]:
        return [t.strip() for t in (path_str or "").split("->") if t.strip()] if path_str else []

    # Atomic write helper
    def _atomic_write(target: str, payload: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        tmp = f"{target}.tmp"
        with open(tmp, "w", encoding="utf-8") as wf:
            json.dump(payload, wf, ensure_ascii=False, indent=2)
            wf.flush(); os.fsync(wf.fileno())
        os.replace(tmp, target)

    # Resume: load existing output and build processed path_id set
    results: List[Dict[str, Any]] = []
    processed_ids: set[int] = set()

    if resume and os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as rf:
            existing = json.load(rf) or []
        if isinstance(existing, list):
            results.extend(existing)
            for it in existing:
                processed_ids.add(int(it.get("path_id")))
            print(f"[llm_filter] Resuming: loaded {len(existing)} records; skip their path_id(s)")

    # Build candidates to process
    def _get_pid(p: Dict[str, Any]) -> int:
        try:
            return int(p.get("path_id") or 0)
        except Exception:
            return 0

    candidates: List[Tuple[int, Dict[str, Any]]] = []
    total_paths = len(data)
    for p in data:
        pid = _get_pid(p)
        if not pid:
            continue
        if resume and pid in processed_ids:
            continue
        candidates.append((pid, p))

    print(f"[llm_filter] Will process {len(candidates)}/{total_paths} paths (after resume filtering)")

    # Per-thread worker that processes one path dict and returns (pid, processed_path)
    def _worker(path_record: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        pid, path = path_record
        # Construct a per-thread client to avoid session contention
        local_client = LLMClient(config=client.config)
        name_path = path.get("name_path") or ""
        tokens = parse_tokens(name_path)
        raw_q = path.get("raw_question") or path.get("question") or ""
        steps = path.get("steps", [])

        for step in steps:
            results_step = step.get("results") or []
            if not results_step:
                continue
            qtype = step.get("query_type") or ""
            idx = int(step.get("step_index") or 0)
            hop = idx // 2
            rel_i = 1 + 2 * hop
            ent_i = 2 + 2 * hop
            golden_key = tokens[rel_i] if "relations" in qtype and rel_i < len(tokens) else (
                tokens[ent_i] if ent_i < len(tokens) else ""
            )

            out = plan_and_filter_step(
                client=local_client,
                raw_question=raw_q,
                name_path=name_path,
                step=step,
                k=k,
                golden_key=golden_key,
                prompt_template=tmpl,
            )
            new_step = apply_plan_and_truncate_step(step=step, plan=out["plan"], keep_keys=out["keep_keys"])
            step.update(new_step)
        return pid, path

    lock = Lock()
    new_since_save = 0

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(_worker, rec) for rec in candidates]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="Filtering paths", unit="path"):
            pid, processed = _.result()
            with lock:
                results.append(processed)
                processed_ids.add(pid)
                new_since_save += 1
                if save_every and new_since_save >= int(save_every):
                    _atomic_write(out_file, results)
                    print(f"[llm_filter] checkpoint wrote {len(results)} records -> {out_file}")
                    new_since_save = 0

    
    _atomic_write(out_file, results)
    print(f"[llm_filter] final write {len(results)} records -> {out_file}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply LLM plan+filter to info_syn JSON.")
    parser.add_argument("--in-file", required=True, help="Input info_syn JSON file")
    parser.add_argument("--out-file", required=True, help="Output JSON file")
    parser.add_argument("--k", type=int, default=5, help="Max items to keep per step")
    parser.add_argument("--model", type=str, default=None, help="Optional model name override")
    parser.add_argument("--base-url", type=str, default=None, help="Optional API base URL override")
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key override")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker threads")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint write every N processed paths")
    parser.add_argument("--resume", action="store_true", help="Resume by path_id from existing out-file")
    args = parser.parse_args()

    client = LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key)
    process_file(
        client=client,
        in_file=args.in_file,
        out_file=args.out_file,
        k=args.k,
        workers=args.workers,
        save_every=args.save_every,
        resume=args.resume,
    )
