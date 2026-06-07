#!/usr/bin/env python3
"""
CLI entry point for generating questions from conjunction paths.
"""
import sys
from pathlib import Path

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kgqa_agent.src.data_gen.generate_conjunction_questions import main

if __name__ == "__main__":
    main()

