"""Evaluation package."""

from automl.evaluation.metrics import (
    ClassificationMetrics,
    evaluate_classifier,
    select_best_model,
)

__all__ = [
    "ClassificationMetrics",
    "evaluate_classifier",
    "select_best_model",
]