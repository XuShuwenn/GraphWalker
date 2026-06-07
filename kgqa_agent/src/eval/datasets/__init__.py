"""Dataset loader for KGQA evaluation.

Provides unified loader for all KGQA datasets (all datasets now follow cwq format):
- CWQ (ComplexWebQuestions)
- WebQSP (WebQuestionsSP)  
- GrailQA
- SimpleQA

All datasets return standardized format with id, question, answers, and topic_entity.
"""
from .cwq_loader import load_cwq

__all__ = ["load_cwq"]

