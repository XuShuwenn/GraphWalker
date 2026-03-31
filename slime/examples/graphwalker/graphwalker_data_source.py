"""
GraphWalker data source: ensures sample.metadata is always a dict.

cwq_train_prepared.jsonl stores "metadata" as a JSON string. Slime's partial-rollout
abort() does sample.metadata["start_rollout_id"] = rollout_id, which crashes if
metadata is a str. This wrapper normalizes metadata to dict in get_samples() so
all downstream code (including slime's abort()) sees a dict without modifying slime.
"""

import json

from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.types import Sample


def _ensure_metadata_dict(sample: Sample) -> None:
    """In-place: ensure sample.metadata is a dict (for JSONL string metadata)."""
    if isinstance(sample.metadata, dict):
        return
    if sample.metadata is None:
        sample.metadata = {}
    elif isinstance(sample.metadata, str):
        try:
            sample.metadata = json.loads(sample.metadata)
        except (json.JSONDecodeError, TypeError):
            sample.metadata = {"_raw": sample.metadata}
    else:
        sample.metadata = {"_raw": sample.metadata}


class GraphWalkerDataSource(RolloutDataSourceWithBuffer):
    """Data source that normalizes metadata to dict for GraphWalker (no slime changes)."""

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples = super().get_samples(num_samples)
        for group in samples:
            for sample in group:
                _ensure_metadata_dict(sample)
        return samples
