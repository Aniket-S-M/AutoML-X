"""Evaluation utilities for classification models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
    f1_yes: float | None = None
    precision_yes: float | None = None
    recall_yes: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    confusion_matrix: list[list[int]] | None = None

    @property
    def f1_weighted(self) -> float:
        """Return the existing weighted F1 under an explicit report name."""

        return self.f1

    def as_dict(self) -> dict[str, Any]:
        """Return metrics as a serializable dictionary."""
        metrics: dict[str, Any] = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f1_weighted": self.f1_weighted,
        }
        optional_metrics = {
            "precision_yes": self.precision_yes,
            "recall_yes": self.recall_yes,
            "f1_yes": self.f1_yes,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "confusion_matrix": self.confusion_matrix,
        }
        metrics.update(
            {name: value for name, value in optional_metrics.items() if value is not None}
        )
        return metrics

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

    The existing precision, recall, and F1 values use weighted averaging.
    For churn labels, positive-class metrics use ``Yes`` explicitly. ROC-AUC,
    PR-AUC, and the confusion matrix use the same ``No``/``Yes`` label order.
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
    observed_labels = set(np.asarray(y, dtype=object).tolist()) | set(
        np.asarray(predictions, dtype=object).tolist()
    )
    has_yes_label = "Yes" in observed_labels
    y_has_yes = "Yes" in set(np.asarray(y, dtype=object).tolist())
    y_has_no = "No" in set(np.asarray(y, dtype=object).tolist())
    positive_scores = (
        _positive_class_scores(model, X) if has_yes_label else None
    )
    churn_auc_metrics = {}
    if positive_scores is not None:
        binary_y = (np.asarray(y, dtype=object) == "Yes").astype(int)
        if y_has_yes and y_has_no:
            churn_auc_metrics["roc_auc"] = float(
                roc_auc_score(binary_y, positive_scores)
            )
        if y_has_yes:
            churn_auc_metrics["pr_auc"] = float(
                average_precision_score(binary_y, positive_scores)
            )

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
        f1_yes=(
            float(
                f1_score(
                    y,
                    predictions,
                    average="binary",
                    pos_label="Yes",
                    zero_division=0,
                )
            )
            if has_yes_label
            else None
        ),
        precision_yes=(
            float(
                precision_score(
                    y,
                    predictions,
                    average="binary",
                    pos_label="Yes",
                    zero_division=0,
                )
            )
            if has_yes_label
            else None
        ),
        recall_yes=(
            float(
                recall_score(
                    y,
                    predictions,
                    average="binary",
                    pos_label="Yes",
                    zero_division=0,
                )
            )
            if has_yes_label
            else None
        ),
        roc_auc=churn_auc_metrics.get("roc_auc"),
        pr_auc=churn_auc_metrics.get("pr_auc"),
        confusion_matrix=(
            confusion_matrix(y, predictions, labels=["No", "Yes"]).tolist()
            if has_yes_label
            else None
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
        "f1_yes",
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


def _positive_class_scores(
    model: ClassifierMixin,
    X: np.ndarray,
) -> np.ndarray | None:
    """Return probability/decision scores for the literal ``Yes`` class."""

    classes = list(getattr(model, "classes_", []))
    if "Yes" not in classes:
        return None
    yes_index = classes.index("Yes")

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.ndim == 2 and yes_index < probabilities.shape[1]:
            return probabilities[:, yes_index]

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X))
        if decision.ndim == 1 and len(classes) == 2:
            return decision if yes_index == 1 else -decision
        if decision.ndim == 2 and yes_index < decision.shape[1]:
            return decision[:, yes_index]
    return None
