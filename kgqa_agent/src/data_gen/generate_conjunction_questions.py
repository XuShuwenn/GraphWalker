from __future__ import annotations
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List, Optional
import random

import openai  

from dotenv import load_dotenv  
load_dotenv()

from tqdm import tqdm
# Prompt template
import sys
from pathlib import Path

# Ensure repo root is on sys.path for absolute imports when running by file path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3] if len(_THIS_FILE.parents) >= 4 else _THIS_FILE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kgqa_agent.prompts.qa_conj_prompt import GENERATE_CONJUNCTION_QUESTION

# -------- API call logging (file-based, append-only) --------
LOG_FILE = os.path.join(str(_REPO_ROOT), "model_api_calls.log")
_LOG_LOCK = threading.Lock()

def _append_api_log(rec: Dict[str, Any]) -> None:
    with _LOG_LOCK:
        with open(LOG_FILE, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_client() -> Optional[openai.OpenAI]:
    """Create an OpenAI client using OPENAI_* env vars.

    Required:
      - OPENAI_API_KEY
      - OPENAI_API_URL (base URL for API endpoint)
    """
    if openai is None:
        print("[WARN] openai package not installed; LLM calls will be skipped.")
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        print("[WARN] OPENAI_API_KEY not found; LLM calls will be skipped.")
        return None
    if not base_url:
        print("[WARN] OPENAI_API_URL/OPENAI_BASE_URL not found; LLM calls will be skipped.")
        return None
    return openai.OpenAI(api_key=api_key, base_url=base_url)  # type: ignore


def extract_question(text: str) -> Optional[str]:
    """Extract question from model output.
    
    Only extracts questions from the model's actual output, not from the prompt.
    Only supports the format: <question>...</question>
    Only matches content after the last "Output:" marker to avoid matching prompt examples.
    """
    if not isinstance(text, str):
        return None
    
    # Find the last "Output:" marker to locate model's actual output
    # The prompt contains an example with "Output:", so we need the last one
    output_marker = "Output:"
    last_output_idx = text.rfind(output_marker)
    
    if last_output_idx == -1:
        # If no "Output:" found, fallback to searching entire text
        # but this should not happen with proper prompt format
        model_output_text = text
    else:
        # Only search in the content after the last "Output:" marker
        model_output_text = text[last_output_idx + len(output_marker):]
    
    # Match format: <question>...</question> only in model output section
    pattern = r"<question>(.*?)</question>"
    matches = list(re.finditer(pattern, model_output_text, re.DOTALL | re.IGNORECASE))
    
    if not matches:
        return None
    
    # Use the first match in the model output section (should be the only one)
    # If multiple matches exist, use the first one (closest to "Output:")
    match = matches[0]
    question_text = match.group(1).strip()
    
    return question_text


def render_prompt_from_item(item: Dict[str, Any]) -> str:
    """Render prompt from conjunction path item."""
    path1 = item.get("name_path1", "")
    path2 = item.get("name_path2", "")
    if not isinstance(path1, str) or not isinstance(path2, str):
        raise ValueError(f"name_path1 and name_path2 must be strings, got {type(path1)}, {type(path2)}")
    return GENERATE_CONJUNCTION_QUESTION % {"path1": path1, "path2": path2}


def call_llm(
    client: Optional[openai.OpenAI], 
    model: str, 
    prompt: str, 
    retries: int, 
    backoff_sec: float,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not client:
        return None
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            create_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            if max_tokens is not None:
                create_kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**create_kwargs)
            # Extract token usage if available
            usage = getattr(resp, "usage", None)
            usage_dict = None
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            text = resp.choices[0].message.content  # type: ignore
            # Append API call record
            _append_api_log({
                "ts": time.time(),
                "model": model,
                "usage": usage_dict,
                "len_prompt": len(prompt) if isinstance(prompt, str) else None,
                "len_output": len(text) if isinstance(text, str) else None,
            })
            return {"text": text, "usage": usage_dict}
        except Exception as e:  # pragma: no cover
            last_err = e
            # exponential backoff with jitter: base * 2^(attempt-1) plus [0, base]
            exp = backoff_sec * (2 ** (attempt - 1))
            jitter = random.uniform(0, backoff_sec)
            sleep_for = min(exp + jitter, 60.0)  # cap to 60s per attempt
            print(f"[LLM] error (attempt {attempt}/{retries}): {e}. sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
    print(f"[LLM] failed after {retries} attempts: {last_err}")
    return None


def parse_json_ids(path: str, id_key: str) -> set[str]:
    """Parse already processed IDs from JSON file."""
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict) and id_key in obj:
                    done.add(str(obj[id_key]))
    except Exception:
        pass
    return done


def load_input_items(path: str) -> List[Dict[str, Any]]:
    """Load input items from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def process_item(
    item: Dict[str, Any],
    client: Optional[openai.OpenAI],
    model: str,
    id_key: str,
    retries: int,
    backoff_sec: float,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single item to generate question."""
    prompt = render_prompt_from_item(item)
    llm_res = call_llm(
        client, 
        model, 
        prompt, 
        retries=retries, 
        backoff_sec=backoff_sec,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not llm_res:
        return None
    
    question = extract_question(llm_res.get("text", ""))
    
    # Build output with required fields
    out: Dict[str, Any] = {}
    if id_key in item:
        out[id_key] = item[id_key]
    else:
        out[id_key] = None
    
    out["question"] = question
    out["name_path1"] = item.get("name_path1")
    out["name_path2"] = item.get("name_path2")
    out["topic_entities"] = item.get("topic_entities", {})
    out["answers"] = item.get("answers", {})
    
    # Add debug info: full model output and token usage
    if llm_res.get("text"):
        out["_debug_full_output"] = llm_res.get("text")
    if llm_res.get("usage"):
        out["_debug_token_usage"] = llm_res.get("usage")
    
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_conjunction_questions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=str, help="Input JSON file of conjunction paths")
    p.add_argument("--id-key", default="path_id", type=str, help="Key name used as unique id in input")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gemini-2.5-pro"), type=str, help="Model name sent to the OpenAI-compatible endpoint")
    
    # LLM generation parameters with environment variable support
    temp_env = os.getenv("OPENAI_TEMPERATURE")
    p.add_argument(
        "--temperature", 
        type=float, 
        default=float(temp_env) if temp_env else 0.7,
        help="Temperature for generation (0.0-2.0). Can also be set via OPENAI_TEMPERATURE env var. Default: 0.7"
    )
    max_tokens_env = os.getenv("OPENAI_MAX_TOKENS")
    p.add_argument(
        "--max-tokens", 
        type=int, 
        default=int(max_tokens_env) if max_tokens_env else None,
        help="Maximum tokens to generate. Can also be set via OPENAI_MAX_TOKENS env var. Default: None (model default)"
    )
    
    p.add_argument("--max-workers", default=4, type=int)
    p.add_argument("--future-timeout-sec", default=180, type=float)
    p.add_argument("--retries", default=5, type=int)
    p.add_argument("--backoff-sec", default=2.0, type=float)
    p.add_argument("--limit", default=0, type=int, help="Optional cap on number of items to process")
    p.add_argument("--output", default="", type=str, help="Optional explicit output file (.json). If not set, will write to <input_dir>/../qa/<input_basename>.json")
    return p


def derive_output_path(input_path: str, explicit: str) -> str:
    """Derive output path from input path."""
    if explicit:
        os.makedirs(os.path.dirname(explicit) if os.path.dirname(explicit) else ".", exist_ok=True)
        return explicit
    # Default: place in kgqa_agent/data/conjunction/qa/
    input_path_abs = os.path.abspath(input_path)
    input_dir = os.path.dirname(input_path_abs)
    
    # Navigate to conjunction/qa directory
    # Input is typically: kgqa_agent/data/conjunction/paths/xxx.json
    # Output should be: kgqa_agent/data/conjunction/qa/xxx.json
    if "conjunction" in input_dir and "paths" in input_dir:
        # Replace "paths" with "qa"
        qa_dir = input_dir.replace("paths", "qa")
    elif "conjunction" in input_dir:
        # If already in conjunction directory, add qa subdirectory
        qa_dir = os.path.join(input_dir, "qa")
    else:
        # Fallback: use parent directory
        parent_dir = os.path.dirname(input_dir)
        qa_dir = os.path.join(parent_dir, "qa")
    
    os.makedirs(qa_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(qa_dir, f"{base}.json")


def main():
    args = build_argparser().parse_args()

    client = build_client()

    items = load_input_items(args.input)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    out_path = derive_output_path(args.input, args.output)
    processed_ids = parse_json_ids(out_path, args.id_key)
    if processed_ids:
        print(f"[RESUME] already processed ids: {len(processed_ids)}")

    # Filter out already processed by id_key
    work_items: List[Dict[str, Any]] = []
    for it in items:
        if args.id_key in it and str(it[args.id_key]) in processed_ids:
            continue
        work_items.append(it)

    total = len(work_items)
    if total == 0:
        print("[DONE] nothing to process")
        print(f"[OUT] {out_path}")
        return

    # Load existing results if resuming
    existing_results: List[Dict[str, Any]] = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = []
        except Exception:
            existing_results = []
    
    existing_ids = {str(r.get(args.id_key)) for r in existing_results if args.id_key in r}
    results_dict = {str(r.get(args.id_key)): r for r in existing_results if args.id_key in r}

    write_lock = threading.Lock()
    fn = partial(
        process_item,
        client=client,
        model=args.model,
        id_key=args.id_key,
        retries=args.retries,
        backoff_sec=args.backoff_sec,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    prog = tqdm(total=total, desc="Generating", ncols=80) if tqdm else None
    new_results: List[Dict[str, Any]] = []
    
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {ex.submit(fn, it): it for it in work_items}
        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=args.future_timeout_sec)
            except Exception as e:
                print(f"[FUTURE] error/timeout: {e}")
                res = None
            if res:
                with write_lock:
                    new_results.append(res)
                    # Periodically save
                    if len(new_results) >= 10:
                        # Merge with existing results
                        all_results = list(results_dict.values())
                        for nr in new_results:
                            if args.id_key in nr:
                                results_dict[str(nr[args.id_key])] = nr
                        all_results = list(results_dict.values())
                        # Sort by path_id if possible
                        try:
                            all_results.sort(key=lambda x: x.get(args.id_key, 0) if isinstance(x.get(args.id_key), (int, float)) else 0)
                        except Exception:
                            pass
                        # Atomic write
                        tmp_path = f"{out_path}.tmp"
                        with open(tmp_path, "w", encoding="utf-8") as fout:
                            json.dump(all_results, fout, ensure_ascii=False, indent=2)
                            fout.flush()
                            os.fsync(fout.fileno())
                        os.replace(tmp_path, out_path)
                        new_results = []
            if prog:
                prog.update(1)
    
    # Final save
    if new_results:
        for nr in new_results:
            if args.id_key in nr:
                results_dict[str(nr[args.id_key])] = nr
    all_results = list(results_dict.values())
    try:
        all_results.sort(key=lambda x: x.get(args.id_key, 0) if isinstance(x.get(args.id_key), (int, float)) else 0)
    except Exception:
        pass
    
    # Atomic write
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fout:
        json.dump(all_results, fout, ensure_ascii=False, indent=2)
        fout.flush()
        os.fsync(fout.fileno())
    os.replace(tmp_path, out_path)
    
    if prog:
        prog.close()

    print(f"[DONE] wrote to {out_path}")


if __name__ == "__main__":
    main()

