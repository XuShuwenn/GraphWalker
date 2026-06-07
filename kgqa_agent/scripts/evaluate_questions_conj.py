#!/usr/bin/env python3
"""Multi-dimensional quality evaluation and filtering for conjunction question-path data using a strong model."""

# ============================================================================
# Configuration
# ============================================================================
INPUT_FILE = "all_data/question_path_conj_5k.json"  # Input JSON file path
OUTPUT_FILE = "all_data/question_path_conj_5k_evaluated.json"  # Output JSON file path
LIMIT = None  # Max number of items to process (None means process all; e.g., set to 10 for testing)
MODEL = "gpt-4o-mini"  # Model name (None means use LLMConfig.from_env() default)
TEMPERATURE = 0.0  # Generation temperature (recommended 0.0 for deterministic evaluation)
NUM_THREADS = 4  # Number of worker threads (recommended 2-8 based on API limits and machine resources)
# ============================================================================

import json
import re
import sys
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add repository root to Python path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) >= 3 else _THIS_FILE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kgqa_agent.src.data_gen.utils.client import LLMClient, LLMConfig
from kgqa_agent.prompts.question_path_filter_prompt_conj import (
    get_filter_prompt,
    REASONING_PATH_QUALITY,
    QUESTION_PATH_RELEVANCE,
    QUESTION_SEMANTIC_COHERENCE,
)


def parse_score_from_response(response: str) -> Optional[int]:
    """Parse a score from model response text.

    Args:
        response: Model output text.

    Returns:
        Parsed score in [0, 10], or None if parsing fails.
    """
    # Try to match <score><number></score>
    match = re.search(r'<score>\s*(\d+)\s*</score>', response, re.IGNORECASE)
    if match:
        try:
            score = int(match.group(1))
            if 0 <= score <= 10:
                return score
        except ValueError:
            pass

    # Fallback: find any standalone 0-10 number
    numbers = re.findall(r'\b([0-9]|10)\b', response)
    if numbers:
        try:
            score = int(numbers[-1])  # use the last matched number
            if 0 <= score <= 10:
                return score
        except ValueError:
            pass

    return None


def evaluate_single_dimension(
    client: LLMClient,
    item: Dict[str, Any],
    dimension: str,
    prompt: str
) -> Optional[int]:
    """Evaluate a single dimension score (conjunction version).

    Args:
        client: LLM client.
        item: Data item containing name_path1, name_path2, and raw_question.
        dimension: Dimension name.
        prompt: Full evaluation prompt.

    Returns:
        Score in [0, 10], or None if evaluation fails.
    """
    name_path1 = item.get("name_path1", "")
    name_path2 = item.get("name_path2", "")
    question = item.get("raw_question", "")

    # Build model input content
    input_content = f"""name_path1: \"{name_path1}\"\nname_path2: \"{name_path2}\"\nquestion: \"{question}\"\n\nPlease evaluate this question-path pair and output the score."""

    # Combine full prompt
    full_prompt = f"{prompt}\n\n{input_content}"

    # Call LLM
    try:
        messages = [
            {"role": "system", "content": "You are an expert evaluator. Follow the instructions precisely and output only the score in the specified format."},
            {"role": "user", "content": full_prompt}
        ]

        # Use LLMClient private _chat method for direct call in this script
        response = client._chat(messages, temperature=0.0, max_tokens=100)

        # Parse score
        score = parse_score_from_response(response)
        return score

    except Exception as e:
        print(f"⚠️  Error evaluating dimension {dimension} (index={item.get('index', '?')}): {e}", file=sys.stderr)
        return None


def get_unprocessed_indices(output_file: str, input_data: List[Dict[str, Any]]) -> List[int]:
    """Get unprocessed input array indices by checking existing output.

    Args:
        output_file: Output file path.
        input_data: Input data list.

    Returns:
        List of unprocessed indices in input_data (0-based array indices).
    """
    output_path = Path(output_file)
    processed_indices = set()

    # Read processed item indices
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    for item in existing_data:
                        idx = item.get("index")
                        if idx is not None:
                            processed_indices.add(int(idx))
        except Exception as e:
            print(f"⚠️  Error reading output file: {e}", file=sys.stderr)

    # Find unprocessed input array positions based on item['index']
    unprocessed_indices = []
    for array_idx, item in enumerate(input_data):
        item_index = item.get("index")
        if item_index is not None and int(item_index) not in processed_indices:
            unprocessed_indices.append(array_idx)

    return unprocessed_indices


def filter_questions(
    input_file: str,
    output_file: str,
    limit: Optional[int] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    num_threads: int = 4,
) -> None:
    """Run multi-dimensional evaluation for conjunction question-path data.

    Args:
        input_file: Input JSON file path.
        output_file: Output JSON file path.
        limit: Max number of items to process (None means process all).
        model: Model name (None means use default configuration).
        temperature: Generation temperature.
        num_threads: Number of worker threads.
    """
    print(f"Reading input file: {input_file}")

    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("❌ Error: input file must be a JSON array")
        return

    total_items = len(data)
    print(f"Total items: {total_items}")

    # Resume support: get unprocessed item indices
    unprocessed_indices = get_unprocessed_indices(output_file, data)

    if unprocessed_indices:
        processed_count = total_items - len(unprocessed_indices)
        print(f"📌 Existing output detected. Processed: {processed_count}/{total_items} ({processed_count/total_items*100:.1f}%)")
        print(f"   Remaining: {len(unprocessed_indices)}")

        # If limit is set, only process the first `limit` unprocessed items
        if limit:
            unprocessed_indices = unprocessed_indices[:limit]
            print(f"   This run will process: {len(unprocessed_indices)} (limited by limit)")

        items_to_process = [data[idx] for idx in unprocessed_indices]
        index_mapping = {i: unprocessed_indices[i] for i in range(len(items_to_process))}
    else:
        print("📌 Start from index 0 (new run)")
        end_index = limit if limit else total_items
        items_to_process = data[:end_index]
        index_mapping = {i: i for i in range(len(items_to_process))}

    print(f"Will process {len(items_to_process)} items")

    # Initialize LLM client
    print("\nInitializing LLM client...")
    import os
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    if not base_url or not api_key:
        print("❌ Error: please set OPENAI_BASE_URL and OPENAI_API_KEY in .env")
        return

    # Use specified model or default model
    if model:
        config = LLMConfig(
            model=model,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            temperature=temperature,
            max_tokens=128,
        )
        client = LLMClient(config=config)
    else:
        default_config = LLMConfig.from_env()
        config = LLMConfig(
            model=default_config.model,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            temperature=temperature,
            max_tokens=128,
        )
        client = LLMClient(config=config)

    print("✅ LLM client initialized")
    print(f"   Model: {client.config.model}")
    print(f"   API URL: {client.config.base_url}")

    # Build prompts for three dimensions (conjunction version)
    prompts = {
        REASONING_PATH_QUALITY: get_filter_prompt(REASONING_PATH_QUALITY),
        QUESTION_PATH_RELEVANCE: get_filter_prompt(QUESTION_PATH_RELEVANCE),
        QUESTION_SEMANTIC_COHERENCE: get_filter_prompt(QUESTION_SEMANTIC_COHERENCE),
    }

    print(f"\nStart evaluation (using {num_threads} threads)...")

    # Lock for file writes
    file_lock = threading.Lock()

    # Progress bar
    pbar = tqdm(total=len(items_to_process), desc="Evaluation progress")

    # Shared statistics (thread-safe)
    stats_lock = threading.Lock()
    all_results = []
    failed_count = [0]  # list wrapper for shared mutability across threads

    def process_single_item(item_data: tuple) -> Dict[str, Any]:
        """Process one item (worker function).

        Args:
            item_data: (idx, item, original_array_idx, actual_index)
                - idx: index in items_to_process
                - item: data item
                - original_array_idx: index in original data array
                - actual_index: item['index'] value

        Returns:
            Processed result dictionary.
        """
        idx, item, original_array_idx, actual_index = item_data

        # Each thread uses its own LLM client
        if model:
            thread_client = LLMClient(config=LLMConfig(
                model=model,
                base_url=base_url.rstrip("/"),
                api_key=api_key,
                temperature=temperature,
                max_tokens=128,
            ))
        else:
            default_config = LLMConfig.from_env()
            thread_client = LLMClient(config=LLMConfig(
                model=default_config.model,
                base_url=base_url.rstrip("/"),
                api_key=api_key,
                temperature=temperature,
                max_tokens=128,
            ))

        item_result = {
            "index": item.get("index", actual_index),
            "name_path1": item.get("name_path1", ""),
            "name_path2": item.get("name_path2", ""),
            "raw_question": item.get("raw_question", ""),
            "scores": {},
            "evaluation_status": "success"
        }

        # Evaluate all three dimensions
        for dimension in [REASONING_PATH_QUALITY, QUESTION_PATH_RELEVANCE, QUESTION_SEMANTIC_COHERENCE]:
            prompt = prompts[dimension]
            score = evaluate_single_dimension(thread_client, item, dimension, prompt)

            if score is not None:
                item_result["scores"][dimension] = score
            else:
                item_result["scores"][dimension] = None
                item_result["evaluation_status"] = "partial_failure"
                with stats_lock:
                    failed_count[0] += 1

        # If all dimensions failed
        if all(v is None for v in item_result["scores"].values()):
            item_result["evaluation_status"] = "failed"

        pbar.update(1)
        return item_result

    # Attach index metadata for each item
    items_with_index = []
    for idx, item in enumerate(items_to_process):
        original_array_idx = index_mapping[idx]
        actual_index = item.get("index", original_array_idx + 1)
        items_with_index.append((idx, item, original_array_idx, actual_index))

    # Thread pool execution
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_single_item, item_data) for item_data in items_with_index]

        completed_count = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                with stats_lock:
                    all_results.append(result)
                    completed_count += 1

                    # Save every 10 completed items
                    if completed_count % 10 == 0:
                        _save_results(all_results, output_file, 0, lock=file_lock)
            except Exception as e:
                print(f"⚠️  Error while processing item: {e}", file=sys.stderr)

    pbar.close()

    # Final save
    _save_results(all_results, output_file, 0, lock=file_lock)

    results = all_results

    # Summary statistics
    print("\nEvaluation complete!")
    print(f"  Processed: {len(results)}")
    print(f"  Success: {len([r for r in results if r['evaluation_status'] == 'success'])}")
    print(f"  Partial failure: {len([r for r in results if r['evaluation_status'] == 'partial_failure'])}")
    print(f"  Failed: {len([r for r in results if r['evaluation_status'] == 'failed'])}")
    print(f"  Failed evaluations: {failed_count[0]}")

    # Average score per dimension
    for dimension in [REASONING_PATH_QUALITY, QUESTION_PATH_RELEVANCE, QUESTION_SEMANTIC_COHERENCE]:
        scores = [r["scores"].get(dimension) for r in results if r["scores"].get(dimension) is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"  {dimension} average: {avg_score:.2f} (valid: {len(scores)}/{len(results)})")

    print(f"\nResults saved to: {output_file}")


def _save_results(results: List[Dict[str, Any]], output_file: str, start_index: int = 0, lock: Optional[threading.Lock] = None) -> None:
    """Save results to file (thread-safe).

    Args:
        results: Result list to save.
        output_file: Output file path.
        start_index: Start index (deprecated; kept for backward compatibility).
        lock: Optional lock. If None, caller is responsible for synchronization.
    """
    def _do_save():
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing results if file exists
        existing_results = []
        if output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        existing_results = existing_data
            except Exception:
                pass

        # Merge by index (new results overwrite old ones)
        result_dict = {r["index"]: r for r in existing_results}
        for r in results:
            result_dict[r["index"]] = r

        # Sort by index
        final_results = sorted(result_dict.values(), key=lambda x: x.get("index", 0))

        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

    if lock is not None:
        with lock:
            _do_save()
    else:
        _do_save()


def main():
    """Entry point: run evaluation with configuration above."""
    filter_questions(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        limit=LIMIT,
        model=MODEL,
        temperature=TEMPERATURE,
        num_threads=NUM_THREADS,
    )


if __name__ == '__main__':
    main()
