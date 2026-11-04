from __future__ import annotations

import numpy as np

from automl.evaluation.metrics import evaluate_classifier


class FixedClassifier:
    def __init__(self, predictions: list[str], positive_scores=None):
        self.predictions = np.asarray(predictions)
        self.classes_ = np.asarray(["No", "Yes"])
        self.positive_scores = (
            np.asarray(positive_scores, dtype=float)
            if positive_scores is not None
            else np.where(self.predictions == "Yes", 0.8, 0.2)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert len(X) == len(self.predictions)
        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert len(X) == len(self.positive_scores)
        return np.column_stack([1.0 - self.positive_scores, self.positive_scores])


def test_f1_yes_uses_yes_as_positive_label():
    y = np.asarray(["No", "No", "No", "Yes", "Yes"])
    predictions = ["No", "No", "No", "Yes", "No"]

    metrics = evaluate_classifier(FixedClassifier(predictions), np.zeros((5, 1)), y)

    assert metrics.f1_yes == 2 / 3
    assert metrics.as_dict()["f1_yes"] == 2 / 3
    assert metrics.f1 != metrics.f1_yes


def test_f1_yes_remains_class_specific_on_imbalanced_data():
    y = np.asarray(["No"] * 95 + ["Yes"] * 5)
    predictions = ["No"] * 95 + ["Yes"] * 2 + ["No"] * 3

    metrics = evaluate_classifier(
        FixedClassifier(predictions), np.zeros((100, 1)), y
    )

    assert metrics.f1_yes == 4 / 7
    assert metrics.f1 > metrics.f1_yes


def test_f1_yes_is_zero_when_no_positive_predictions_are_made():
    y = np.asarray(["No"] * 10 + ["Yes"] * 2)
    predictions = ["No"] * len(y)

    metrics = evaluate_classifier(
        FixedClassifier(predictions), np.zeros((len(y), 1)), y
    )

    # zero_division=0 makes undefined positive-class precision and F1 safe zeros.
    assert metrics.precision_yes == 0.0
    assert metrics.recall_yes == 0.0
    assert metrics.f1_yes == 0.0


def test_f1_yes_is_absent_for_non_churn_label_schemes():
    y = np.asarray([0, 0, 1, 1])
    metrics = evaluate_classifier(
        FixedClassifier([0, 0, 1, 0]), np.zeros((4, 1)), y
    )

    assert metrics.f1_yes is None
    assert "f1_yes" not in metrics.as_dict()


def test_churn_metrics_include_positive_class_scores_auc_and_confusion_matrix():
    y = np.asarray(["No", "No", "Yes", "Yes"])
    predictions = ["No", "Yes", "Yes", "No"]
    model = FixedClassifier(predictions, positive_scores=[0.1, 0.7, 0.8, 0.4])

    metrics = evaluate_classifier(model, np.zeros((4, 1)), y)

    assert metrics.accuracy == 0.5
    assert metrics.precision_yes == 0.5
    assert metrics.recall_yes == 0.5
    assert metrics.f1_yes == 0.5
    assert metrics.f1_weighted == metrics.f1
    assert metrics.roc_auc == 0.75
    assert np.isclose(metrics.pr_auc, 5 / 6)
    assert metrics.confusion_matrix == [[1, 1], [1, 1]]
