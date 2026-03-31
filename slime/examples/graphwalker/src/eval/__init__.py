"""
Evaluation utilities for KGQA-Agent.

This package contains:
- model_client: Local/API model wrappers
- evaluator: Orchestrates evaluation and saves results
- datasets: Dataset loaders for CWQ/WebQSP/GrailQA
- kg_augmented_client: LLM client with interactive KG querying

Note: SPARQL tools have been moved to kgqa_agent.src.tools:
- tools.direct_sparql_client: Direct Virtuoso SPARQL client
- tools.relation_normalizer: Predicate normalization helpers
- tools.entity_resolver: Optional entity resolver
"""

__all__ = [
    "model_client",
    "datasets",
    "metrics",
    "evaluator",
    "kg_augmented_client",
]
