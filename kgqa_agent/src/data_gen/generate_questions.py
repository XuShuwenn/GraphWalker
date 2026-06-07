from __future__ import annotations
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
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

from kgqa_agent.prompts.new_qa_prompt import GENERATE_QUESTION_FREEBASE

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


def extract_qa(text: str) -> Dict[str, Optional[str]]:
    if not isinstance(text, str):
        return {"question": None, "answer": None}
    # Match <question>...</question> tag only; answer is derived from name_path.
    qm = re.search(r"<question>(.*?)</question>", text, re.DOTALL | re.IGNORECASE)
    q = qm.group(1).strip() if qm else None
    return {"question": q, "answer": None}


def extract_topic_and_answer_from_name_path(name_path: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(name_path, str) or not name_path.strip():
        return None, None
    parts = [t.strip() for t in re.split(r"\s*->\s*", name_path) if t.strip()]
    if not parts:
        return None, None
    # name_path is expected as: entity -> relation -> entity -> ... -> entity
    entities = parts[0::2]
    topic = entities[0] if entities else None
    answer = entities[-1] if entities else None
    return topic, answer



def render_prompt_from_item(item: Dict[str, Any]) -> Optional[str]:
    # Prefer explicit name_path, otherwise fall back to a best-effort path string
    path_str = item.get("name_path") or item.get("path") or None
    if not isinstance(path_str, str):
        # try to build from names array
        names = item.get("names") or item.get("path_names")
        if isinstance(names, list) and names:
            path_str = " -> ".join(map(str, names))
        else:
            # Removed unsafe fallback that stringifies the whole item (may include
            # large _debug fields). Return None so caller can skip this item.
            return None
    # Replace intermediate entity names with entity1, entity2, etc.
    # Format: entity -> relation -> entity -> relation -> ... -> entity
    # Keep first entity (topic) and last entity (answer), replace middle entities
    if path_str and isinstance(path_str, str):
        parts = [p.strip() for p in path_str.split(" -> ")]
        if len(parts) >= 3:  # At least: entity -> relation -> entity
            # Entities are at even indices (0, 2, 4, ...), relations at odd indices (1, 3, 5, ...)
            topic_entity = parts[0]  # First entity (topic entity)
            answer_entity = parts[-1]  # Last entity (answer entity)
            
            # Build new path with intermediate entities replaced
            new_parts = [topic_entity]
            entity_counter = 1
            
            # Process all parts: relations and intermediate entities
            # The path format is: entity -> relation -> entity -> relation -> ... -> entity
            # So we process from index 1 (first relation) to the end
            for i in range(1, len(parts)):
                if i == len(parts) - 1:
                    # Last part is the answer entity, keep it
                    new_parts.append(answer_entity)
                elif i % 2 == 1:  # Relation
                    new_parts.append(parts[i])
                else:  # Intermediate entity
                    new_parts.append(f"entity{entity_counter}")
                    entity_counter += 1
            
            path_str = " -> ".join(new_parts)
    
    return GENERATE_QUESTION_FREEBASE % {"path": path_str}


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


def parse_jsonl_ids(path: str, id_key: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if id_key in obj and isinstance(obj[id_key], (str, int)):
                    done.add(str(obj[id_key]))
            except Exception:
                continue
    return done


def load_input_items(path: str) -> List[Dict[str, Any]]:
    # Try JSON array first
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except Exception:
        pass

    # Fallback: JSONL
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                continue
    return items


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
    prompt = render_prompt_from_item(item)
    if not prompt:
        return None
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
    qa = extract_qa(llm_res.get("text", ""))
    # Derive topic_entity and answer entity from name_path
    topic_entity: Optional[str] = None
    answer_entity: Optional[str] = None
    name_path = item.get("name_path")
    if isinstance(name_path, str) and name_path.strip():
        topic_entity, answer_entity = extract_topic_and_answer_from_name_path(name_path)

    # Fallbacks if name_path missing
    if not topic_entity:
        names = item.get("names") or item.get("path_names")
        if isinstance(names, list) and names:
            topic_entity = str(names[0])
    if not answer_entity:
        names = item.get("names") or item.get("path_names")
        if isinstance(names, list) and names:
            answer_entity = str(names[-1])

    # If we still cannot determine the answer entity, skip this item
    if not answer_entity:
        return None

    # Build output in the required order: path_id, question, topic_entity, answer
    out: Dict[str, Any] = {}
    if id_key in item:
        out[id_key] = item[id_key]
    else:
        out[id_key] = None
    out["question"] = qa.get("question")
    out["topic_entity"] = topic_entity
    out["answer"] = answer_entity
    # Add debug info: full model output and token usage
    if llm_res.get("text"):
        out["_debug_full_output"] = llm_res.get("text")
    if llm_res.get("usage"):
        out["_debug_token_usage"] = llm_res.get("usage")
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_questions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=str, help="Input JSON/JSONL file of paths")
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
    p.add_argument("--retries", default=2, type=int)
    p.add_argument("--backoff-sec", default=2.0, type=float)
    p.add_argument("--limit", default=0, type=int, help="Optional cap on number of items to process")
    p.add_argument("--output", default="", type=str, help="Optional explicit output file (.jsonl). If not set, will write to <parent_of_input_dir>/qa/<input_basename>.qa.jsonl")
    return p


def derive_output_path(input_path: str, explicit: str) -> str:
    if explicit:
        os.makedirs(os.path.dirname(explicit), exist_ok=True)
        return explicit
    # Default: place under the parent-of-input-directory in a folder named 'qa'
    input_dir = os.path.dirname(os.path.abspath(input_path))
    parent_of_input_dir = os.path.dirname(input_dir)
    qa_dir = os.path.join(parent_of_input_dir, "qa")
    os.makedirs(qa_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(qa_dir, f"{base}.qa.jsonl")


def main():
    args = build_argparser().parse_args()

    client = build_client()

    items = load_input_items(args.input)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    out_path = derive_output_path(args.input, args.output)
    processed_ids = parse_jsonl_ids(out_path, args.id_key)
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
    writes_since_sync = 0
    with open(out_path, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = [ex.submit(fn, it) for it in work_items]
        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=args.future_timeout_sec)
            except Exception as e:
                print(f"[FUTURE] error/timeout: {e}")
                res = None
            if res:
                with write_lock:
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    writes_since_sync += 1
                    if writes_since_sync >= 10:
                        fout.flush()
                        os.fsync(fout.fileno())
                        writes_since_sync = 0
            if prog:
                prog.update(1)
        # Final flush/sync (inside with block, before file closes)
        with write_lock:
            if writes_since_sync > 0:
                fout.flush()
                os.fsync(fout.fileno())
    if prog:
        prog.close()

    print(f"[DONE] wrote to {out_path}")


if __name__ == "__main__":
    main()
