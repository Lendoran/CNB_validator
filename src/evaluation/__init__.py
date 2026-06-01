"""Evaluation package for CNB OAM document classification.

Provides evaluation metrics, confusion matrix plots, per-class F1 bar charts,
and side-by-side model comparison utilities.
"""

from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.comparison import MethodComparison

__all__ = [
    "ClassificationMetrics",
    "MethodComparison",
]
