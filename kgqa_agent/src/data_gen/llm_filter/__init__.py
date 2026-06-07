"""LLM-based plan + filtering (package facade).

We avoid importing heavy submodules at package import time. Only expose
LLMClient eagerly; other helpers are loaded lazily on first access.
"""

from typing import Any
from ..utils.client import LLMClient

# Only export the light-weight client by default
__all__ = ["LLMClient"]

_LAZY_EXPORTS = {
    "candidate_key", 
    "plan_and_filter_step", 
    "apply_plan_and_truncate_step", 
    "process_file"
    }


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        from . import filter_engine as _fe  # local import to keep import-time light
        return getattr(_fe, name)
    raise AttributeError(name)
