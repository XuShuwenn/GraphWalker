"""
Trajectory saving utilities for GraphWalker training.

This module handles:
- Experiment directory management
- Trajectory file naming and locking
- Async trajectory saving to JSONL files
"""

import asyncio
import json
import os
import re
import hashlib
from datetime import datetime

from slime.utils.types import Sample
from graphwalker_reward import extract_answer


# ============== Directory Management ==============

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJECTORY_BASE_DIR = os.path.join(SCRIPT_DIR, "trajectories")
os.makedirs(TRAJECTORY_BASE_DIR, exist_ok=True)

_experiment_trajectory_dir = None


def get_experiment_trajectory_dir() -> str:
    """Get or create per-experiment trajectory directory (trajectories/experiment_YYYYMMDD_HHMMSS/)."""
    global _experiment_trajectory_dir
    
    if _experiment_trajectory_dir is not None:
        return _experiment_trajectory_dir
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _experiment_trajectory_dir = os.path.join(TRAJECTORY_BASE_DIR, f"experiment_{timestamp}")
    os.makedirs(_experiment_trajectory_dir, exist_ok=True)
    
    return _experiment_trajectory_dir


# ============== Trajectory Saving State ==============

_trajectory_counter = 0
_trajectory_lock = asyncio.Lock()
_file_locks = {}
_file_locks_lock = asyncio.Lock()


# ============== Utility Functions ==============

def get_trajectory_filename(metadata: dict, original_question: str) -> str:
    """Generate filename based on prompt ID or question hash."""
    if metadata and "id" in metadata:
        safe_id = re.sub(r'[^\w\-_\.]', '_', str(metadata["id"]))
        return f"trajectory_{safe_id}.jsonl"
    question_hash = hashlib.md5(original_question.encode('utf-8')).hexdigest()[:12]
    return f"trajectory_{question_hash}.jsonl"


async def get_file_lock(filename: str) -> asyncio.Lock:
    """Get or create lock for a specific file."""
    async with _file_locks_lock:
        if filename not in _file_locks:
            _file_locks[filename] = asyncio.Lock()
        return _file_locks[filename]


# ============== Main Trajectory Saving Function ==============

async def save_trajectory(
    sample: Sample,
    reward: float,
    reward_breakdown: dict,
    metadata: dict,
    ground_truth: list,
    config: dict,
) -> None:
    """Save trajectory to JSONL file (grouped by prompt).
    
    Args:
        sample: The Sample object containing generation results
        reward: Final reward score
        reward_breakdown: Detailed reward breakdown dict
        metadata: Sample metadata (e.g., question ID, topic entities)
        ground_truth: Ground truth answers
        config: Configuration dict (expects 'save_trajectories' and 'trajectory_save_frequency' keys)
    """
    global _trajectory_counter
    
    if not config.get("save_trajectories", True):
        return
    
    async with _trajectory_lock:
        _trajectory_counter += 1
        trajectory_id = _trajectory_counter
        save_frequency = config.get("trajectory_save_frequency", 1)
        if _trajectory_counter % save_frequency != 0:
            return
    
    extracted_answer = extract_answer(sample.response) if sample.response else None
    turns_history = getattr(sample, '_turns_history', [])
    original_question = getattr(sample, '_original_question', None) or sample.prompt
    
    trajectory = {
        "trajectory_id": trajectory_id,
        "sample_index": getattr(sample, 'index', None),
        "group_index": getattr(sample, 'group_index', None),
        "original_question": original_question,
        "instruction_prompt": getattr(sample, '_instruction_prompt', None),
        "turns": turns_history,
        "extracted_answer": extracted_answer,
        "ground_truth": ground_truth,
        "reward": reward,
        "reward_breakdown": reward_breakdown,
        "metadata": metadata,
        "status": sample.status.name if hasattr(sample.status, 'name') else str(sample.status),
        "response_length": len(sample.response) if sample.response else 0,
        "token_count": len(sample.tokens) if hasattr(sample, 'tokens') and sample.tokens else 0,
        "num_turns": len(turns_history),
    }
    
    filename = get_trajectory_filename(metadata, original_question)
    experiment_dir = get_experiment_trajectory_dir()
    filepath = os.path.join(experiment_dir, filename)
    file_lock = await get_file_lock(filename)
    
    async with file_lock:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trajectory, indent=2, ensure_ascii=False) + '\n')
