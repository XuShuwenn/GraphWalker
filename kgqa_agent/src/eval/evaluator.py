"""Unified evaluation framework for KGQA tasks.

Supports multiple datasets with standardized interfaces:
- CWQ (ComplexWebQuestions)
- WebQSP (WebQuestionsSP)
- GrailQA

Provides flexible evaluation with answer-based and logical form-based metrics.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
import time
from typing import Dict, Any, List, Optional

from kgqa_agent.src.eval.model_client import build_model_client
from kgqa_agent.src.eval.metrics import exact_match, f1_score

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from kgqa_agent.prompts.prompts import build_search_prompt
from tqdm import tqdm

def load_dataset(dataset_type: str, dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset using unified cwq_loader (all datasets now follow cwq format).
    Args:
        dataset_type: Dataset type identifier (for metadata/logging only)
        dataset_path: Path to dataset file
    Returns:
        List of standardized examples
    """
    from .datasets.cwq_loader import load_cwq
    return load_cwq(dataset_path)
    
def run_evaluation(dataset_path: str, *,
                   dataset_type: str,
                   model_cfg: Dict[str, Any],
                   prompt_system: Optional[str] = None,
                   prompt_user_template: Optional[str] = None,
                   limit: Optional[int] = None,
                   num_workers: int = 10,
                   log_path: Optional[str] = None,
                   kg_server_url: str = "http://localhost:18890",
                   max_calls: int = 10,
                   kg_top_k: int = 10,
                   output_dir: str = "eval_results",
                   output_path: Optional[str] = None,
                   full_list: bool = False) -> Dict[str, Any]:
    """Run unified evaluation on a KGQA dataset with batch-level persistence and resume."""
    data = load_dataset(dataset_type, dataset_path)
    
    if limit is not None:
        data = data[:limit]

    # Create shared model clients (all models now use API-based clients)
    # This avoids creating multiple API clients and allows reuse across threads
    shared_base_model = build_model_client(model_cfg)
    
    # Create shared filter client for relation filtering
    # This client is reused across all KGAugmentedModelClient instances to avoid
    # creating multiple API clients and reduce overhead
    # Support separate API configuration for filter client via FILTER_API_KEY and FILTER_API_URL
    from kgqa_agent.src.eval.model_client import BaseModelClient, ModelConfig
    import os
    
    # Get filter API configuration from environment variables (with fallback to main API config)
    # Priority: FILTER_API_KEY/FILTER_API_URL > OPENAI_API_KEY/OPENAI_BASE_URL > default
    filter_api_key = os.getenv("FILTER_API_KEY")
    filter_api_url = os.getenv("FILTER_API_URL")
    filter_model = os.getenv("FILTER_MODEL", "gpt-4o-mini")  # Default filter model
    
    # Build filter client config
    filter_cfg = {
        "model": filter_model,
        "temperature": 0.0,  # Deterministic filtering (no randomness)
        "max_tokens": 1024,  # Sufficient for relation filtering responses
        "stop": ["\n\n"],
        "timeout": 30.0,  # 30 second timeout (shorter than main model to fail fast)
    }
    
    # Set API key: use FILTER_API_KEY if set, otherwise fallback to OPENAI_API_KEY
    if filter_api_key:
        filter_cfg["api_key"] = filter_api_key
        # Don't set api_key_env if api_key is provided directly
    else:
        filter_cfg["api_key_env"] = "OPENAI_API_KEY"
    
    # Set base URL: use FILTER_API_URL if set, otherwise fallback to OPENAI_BASE_URL
    if filter_api_url:
        filter_cfg["base_url"] = filter_api_url
        # Don't set base_url_env if base_url is provided directly
    else:
        filter_cfg["base_url_env"] = "OPENAI_BASE_URL"
    
    shared_filter_client = BaseModelClient(ModelConfig(**filter_cfg))

    def create_model():
        """Create model client for a thread. Reuses shared_base_model and shared_filter_client.
        """
        from kgqa_agent.src.eval.kg_augmented_client import KGAugmentedModelClient
        mdl = KGAugmentedModelClient(
            base_client=shared_base_model,
            filter_client=shared_filter_client,  # Pass shared filter client
            kg_server_url=kg_server_url,
            max_calls=max_calls,
            kg_top_k=kg_top_k,
            dataset_type=dataset_type,
            full_list=full_list,
        )
        return mdl

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    def _id_to_key(value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    model_name = model_cfg.get("model") or model_cfg.get("model_path") or "model"
    model_name = model_name.replace("/", "_")
    os.makedirs(output_dir, exist_ok=True)
    if output_path:
        out_path = output_path
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"{model_name}_{dataset_type}_{ts}.json")

    def _load_existing_output(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    existing_output = _load_existing_output(out_path)
    created_at_value = None
    if isinstance(existing_output, dict):
        created_at_value = existing_output.get("created_at")
    metadata = {
        "dataset_type": dataset_type,
        "dataset_path": dataset_path,
        "model_cfg": model_cfg,
        "created_at": created_at_value or datetime.utcnow().isoformat() + "Z",
    }
    existing_preds = existing_output.get("predictions", []) if isinstance(existing_output, dict) else []
    if not isinstance(existing_preds, list):
        existing_preds = []
    existing_output = existing_output if isinstance(existing_output, dict) else {}
    existing_output.update(metadata)
    existing_output["predictions"] = existing_preds
    existing_output.pop("metrics", None)

    def _persist(data: Dict[str, Any]) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    _persist(existing_output)

    processed_ids = set()
    stored_count = len(existing_preds)
    em_sum = 0.0
    f1_sum = 0.0
    for pred in existing_preds:
        key = _id_to_key(pred.get("id"))
        if key:
            processed_ids.add(key)
        em_sum += float(pred.get("em", 0.0) or 0.0)
        f1_sum += float(pred.get("f1", 0.0) or 0.0)

    # free memory from existing predictions after capturing stats
    existing_output = {}
    existing_preds = []

    # Thread lock for file writing
    _append_lock = threading.Lock()
    # Separate lock for file I/O to prevent concurrent writes
    _io_lock = threading.Lock()
    
    # Buffer for batch writing (write every 10 records to reduce I/O)
    _write_buffer: List[Dict[str, Any]] = []
    _buffer_size = 10
    
    def _flush_buffer() -> None:
        """Flush buffered predictions to file (thread-safe, batch write)."""
        # Extract buffer contents while holding append lock (minimize lock time)
        records_to_write = []
        with _append_lock:
            if not _write_buffer:
                return
            # Copy buffer contents and clear it (all within lock)
            records_to_write = list(_write_buffer)
            _write_buffer.clear()
        
        # Perform I/O with separate lock to prevent concurrent file operations
        # This ensures only one thread writes at a time, preventing data corruption
        with _io_lock:
            # Check if file exists, if not create it with initial structure
            if not os.path.exists(out_path):
                file_content = dict(metadata)
                file_content["predictions"] = []
            else:
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        file_content = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning("Failed to read %s while appending; recreating header. Error: %s", out_path, e)
                    file_content = dict(metadata)
                    file_content["predictions"] = []
            
            if "predictions" not in file_content or not isinstance(file_content["predictions"], list):
                file_content["predictions"] = []
            file_content.update(metadata)
            file_content.pop("metrics", None)
            
            # Append all buffered records at once
            file_content["predictions"].extend(records_to_write)
            
            # Atomic write using temporary file
            buffered_count = len(records_to_write)
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(file_content, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, out_path)
            
            if logger.handlers:
                logger.debug("Flushed %d buffered predictions to %s", buffered_count, out_path)
    
    def _append_prediction(record: Dict[str, Any]) -> None:
        """Append a single prediction record to buffer, flush when buffer is full (thread-safe)."""
        should_flush = False
        with _append_lock:
            _write_buffer.append(record)
            # Check if we need to flush (while holding the lock)
            if len(_write_buffer) >= _buffer_size:
                should_flush = True
        
        # Flush outside the lock to avoid holding it during I/O
        if should_flush:
            _flush_buffer()

    def process_single_example(ex: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single example (one complete multi-turn conversation)."""
        model = create_model()
        q = ex.get("question") or ""
        _id = ex.get("id")
        golds = ex.get("answers", [])
        topic_entity = ex.get("topic_entity")
        
        te_map = topic_entity or {}
        te_names = list(te_map.values()) if isinstance(te_map, dict) else None
        user_prompt = build_search_prompt(question=q, max_calls=max_calls, topic_entities=te_names)
        
        # Generate prediction (handles multi-turn internally)
        pred_text = model.generate(
            user_prompt,
            system=None,
            question=q,
            topic_entities=topic_entity,  # topic_entity is Dict[str, str] (mid->name mapping)
        )
        
        # Extract answer from KG-augmented model response
        p_clean = model.extract_answer(pred_text) if hasattr(model, 'extract_answer') else pred_text
        
        em = exact_match(p_clean, golds)
        f1 = f1_score(p_clean, golds)
        
        rec = {
            "id": _id,
            "raw_question": q,
            "user_prompt": user_prompt,
            "question": q,
            "answers": golds,
            "prediction": p_clean,
            "raw_prediction": pred_text,
            "em": em,
            "f1": f1
        }
        
        # Attach trace (always available since KG is always enabled)
        if hasattr(model, 'get_trace_and_reset'):
            trace = model.get_trace_and_reset()
            if trace:
                rec["trace"] = trace
        
        # Collect filter statistics
        if hasattr(model, 'get_filter_stats_and_reset'):
            filter_stats = model.get_filter_stats_and_reset()
            # Aggregate statistics (thread-safe)
            with _stats_lock:
                _filter_stats_global["total_calls"] += filter_stats.get("total_calls", 0)
                _filter_stats_global["fallback_count"] += filter_stats.get("fallback_count", 0)
                _filter_stats_global["parse_fail_count"] += filter_stats.get("parse_fail_count", 0)
                _filter_stats_global["success_count"] += filter_stats.get("success_count", 0)
        
        return rec
    
    to_eval: List[Dict[str, Any]] = []
    skipped = 0
    for ex in data:
        key = _id_to_key(ex.get("id"))
        if key and key in processed_ids:
            skipped += 1
            continue
        to_eval.append(ex)

    new_predictions: List[Dict[str, Any]] = []
    
    # Thread lock for statistics updates
    _stats_lock = threading.Lock()
    
    # Global filter statistics (aggregated across all threads)
    _filter_stats_global = {
        "total_calls": 0,
        "fallback_count": 0,
        "parse_fail_count": 0,
        "success_count": 0
    }

    def _handle_single_result(rec: Dict[str, Any]):
        """Handle a single result: save to file and update statistics (thread-safe)."""
        nonlocal stored_count, em_sum, f1_sum
        if not rec:
            return
        
        # Save to file (thread-safe)
        _append_prediction(rec)
        
        # Update statistics (thread-safe)
        with _stats_lock:
            new_predictions.append(rec)
            stored_count += 1
            em_sum += float(rec.get("em", 0.0) or 0.0)
            f1_sum += float(rec.get("f1", 0.0) or 0.0)
            key = _id_to_key(rec.get("id"))
            if key:
                processed_ids.add(key)
        
        if logger.handlers:
            logger.info("Saved prediction for id=%s (total=%d, em=%.4f, f1=%.4f)", 
                       rec.get("id"), stored_count, rec.get("em", 0.0), rec.get("f1", 0.0))

    start_ts = time.time()
    total_remaining = len(to_eval)
    if skipped:
        logger.info("Resume mode: %d examples already present in %s", skipped, out_path)

    # Use ThreadPoolExecutor for parallel processing (each thread handles one example)
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = {executor.submit(process_single_example, ex): ex for ex in to_eval}
        with tqdm(total=total_remaining, desc="Evaluating", unit="ex") as pbar:
            for fut in as_completed(futures):
                try:
                    res = fut.result(timeout=600)  # 10 minute timeout per example
                    _handle_single_result(res)
                    pbar.update(1)
                except Exception as e:
                    ex = futures[fut]
                    logger.error("Error processing example id=%s: %s", ex.get("id"), str(e))
                    # Create error record
                    error_rec = {
                        "id": ex.get("id"),
                        "raw_question": ex.get("question", ""),
                        "question": ex.get("question", ""),
                        "answers": ex.get("answers", []),
                        "prediction": f"[ERROR: {str(e)}]",
                        "raw_prediction": f"[ERROR: {str(e)}]",
                        "em": 0.0,
                        "f1": 0.0,
                        "error": str(e)
                    }
                    _handle_single_result(error_rec)
                    pbar.update(1)

    # Flush any remaining buffered predictions before final metrics update
    _flush_buffer()
    
    total_predictions = stored_count
    metrics = {
        "em": em_sum / max(1, total_predictions),
        "f1": f1_sum / max(1, total_predictions),
        "num_examples": total_predictions,
    }

    with open(out_path, "r+", encoding="utf-8") as f:
        final_data = json.load(f)
        final_data.update(metadata)
        final_data["metrics"] = metrics
        if "predictions" not in final_data or not isinstance(final_data["predictions"], list):
            final_data["predictions"] = []
        f.seek(0)
        json.dump(final_data, f, ensure_ascii=False, indent=2)
        f.truncate()

    print(f"\n{'='*60}")
    print(f"Evaluation Results - {dataset_type}")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Examples (total): {metrics.get('num_examples', total_predictions)}")
    total_time = max(1e-6, time.time() - start_ts)
    overall_rate = len(new_predictions) / total_time if new_predictions else 0.0
    print(f"Processed this run: {len(new_predictions)}")
    print(f"Speed: {overall_rate:.2f} it/s | Total time: {total_time:.1f}s")
    print(f"\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and k != "num_examples":
            print(f"  {k}: {v:.4f}")
    
    # Print filter statistics
    if _filter_stats_global["total_calls"] > 0:
        total = _filter_stats_global["total_calls"]
        fallback = _filter_stats_global["fallback_count"]
        parse_fail = _filter_stats_global["parse_fail_count"]
        success = _filter_stats_global["success_count"]
        fallback_rate = (fallback / total * 100) if total > 0 else 0.0
        parse_fail_rate = (parse_fail / total * 100) if total > 0 else 0.0
        success_rate = (success / total * 100) if total > 0 else 0.0
        
        print(f"\nFilter Statistics (LLM Relation Filtering):")
        print(f"  Total filter calls: {total}")
        print(f"  Successful: {success} ({success_rate:.2f}%)")
        print(f"  Fallback (error/timeout): {fallback} ({fallback_rate:.2f}%)")
        print(f"  Parse failures: {parse_fail} ({parse_fail_rate:.2f}%)")
        
        # Log to file as well
        logger.info("Filter Statistics (LLM Relation Filtering):")
        logger.info("  Total filter calls: %d", total)
        logger.info("  Successful: %d (%.2f%%)", success, success_rate)
        logger.info("  Fallback (error/timeout): %d (%.2f%%)", fallback, fallback_rate)
        logger.info("  Parse failures: %d (%.2f%%)", parse_fail, parse_fail_rate)
        
        if fallback_rate > 10.0:
            warning_msg = f"High filter fallback rate: {fallback_rate:.2f}% - consider investigating filter_client performance"
            print(f"  ⚠️  WARNING: {warning_msg}")
            logger.warning(warning_msg)
        if parse_fail_rate > 5.0:
            warning_msg = f"High filter parse failure rate: {parse_fail_rate:.2f}% - consider improving prompt or response parsing"
            print(f"  ⚠️  WARNING: {warning_msg}")
            logger.warning(warning_msg)
    
    print(f"\nResults saved to: {out_path}")
    print(f"{'='*60}\n")

    return {
        "dataset_type": dataset_type,
        "dataset_path": dataset_path,
        "model_cfg": model_cfg,
        "metrics": metrics,
        "predictions": new_predictions,
        "out_path": out_path,
        "summary": {k: v for k, v in metrics.items() if k != "results"},
        "count": len(new_predictions),
    }
