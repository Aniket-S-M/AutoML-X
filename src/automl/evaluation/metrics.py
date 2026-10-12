"""Evaluation utilities for classification models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from automl.exceptions import DataValidationError
from automl.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Classification metrics calculated for a model."""

    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        """Return metrics as a serializable dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def get(self, metric_name: str) -> float:
        """Return a metric by name."""
        normalized_name = metric_name.strip().lower()

        metrics = self.as_dict()

        if normalized_name not in metrics:
            supported = ", ".join(metrics.keys())
            raise DataValidationError(
                f"Unsupported evaluation metric '{metric_name}'. "
                f"Supported metrics: {supported}"
            )

        return metrics[normalized_name]


def evaluate_classifier(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
) -> ClassificationMetrics:
    """
    Evaluate a fitted classification model.

    Precision, recall and F1 use weighted averaging so that this
    function supports both binary and multiclass classification.
    """
    if len(X) != len(y):
        raise DataValidationError(
            "Feature and target lengths must match during evaluation"
        )

    if len(y) == 0:
        raise DataValidationError(
            "Cannot evaluate a model on an empty dataset"
        )

    predictions = model.predict(X)

    metrics = ClassificationMetrics(
        accuracy=float(accuracy_score(y, predictions)),
        precision=float(
            precision_score(
                y,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    )

    logger.info(
        "Evaluation completed: accuracy=%.4f precision=%.4f "
        "recall=%.4f f1=%.4f",
        metrics.accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1,
    )

    return metrics


def select_best_model(
    validation_results: Mapping[str, ClassificationMetrics],
    metric: str,
) -> str:
    """
    Select the best model using validation metrics.

    The test set must never be involved in model selection.
    """
    if not validation_results:
        raise DataValidationError(
            "Cannot select a model because no validation results are available"
        )

    normalized_metric = metric.strip().lower()

    supported_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
    }

    if normalized_metric not in supported_metrics:
        supported = ", ".join(sorted(supported_metrics))
        raise DataValidationError(
            f"Unsupported model-selection metric '{metric}'. "
            f"Supported metrics: {supported}"
        )

    best_model_name = max(
        validation_results,
        key=lambda model_name: validation_results[model_name].get(
            normalized_metric
        ),
    )

    logger.info(
        "Selected best model '%s' using validation metric '%s' "
        "with score %.4f",
        best_model_name,
        normalized_metric,
        validation_results[best_model_name].get(normalized_metric),
    )

    return best_model_name