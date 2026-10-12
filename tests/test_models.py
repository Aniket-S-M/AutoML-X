"""Tests for model training and evaluation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from automl.config.settings import Settings
from automl.evaluation.metrics import (
    ClassificationMetrics,
    evaluate_classifier,
    select_best_model,
)
from automl.exceptions import DataValidationError
from automl.models.trainer import build_classification_models, train_models
from automl.preprocessing import preprocess_dataset


@pytest.fixture
def classification_settings() -> Settings:
    """Return deterministic settings for classification tests."""
    return Settings(
        data={
            "target_column": "target",
            "test_size": 0.2,
            "validation_size": 0.2,
        },
        training={
            "random_seed": 42,
            "metric": "accuracy",
        },
    )


@pytest.fixture
def classification_dataframe() -> pd.DataFrame:
    """Create a deterministic binary classification dataset."""
    rng = np.random.default_rng(42)

    rows_per_class = 60

    class_zero = pd.DataFrame(
        {
            "feature_a": rng.normal(0.0, 0.5, rows_per_class),
            "feature_b": rng.normal(0.0, 0.5, rows_per_class),
            "category": ["A"] * rows_per_class,
            "target": [0] * rows_per_class,
        }
    )

    class_one = pd.DataFrame(
        {
            "feature_a": rng.normal(3.0, 0.5, rows_per_class),
            "feature_b": rng.normal(3.0, 0.5, rows_per_class),
            "category": ["B"] * rows_per_class,
            "target": [1] * rows_per_class,
        }
    )

    return pd.concat(
        [class_zero, class_one],
        ignore_index=True,
    )


@pytest.fixture
def preprocessed_classification_data(
    classification_dataframe: pd.DataFrame,
    classification_settings: Settings,
):
    """Return preprocessed classification data."""
    return preprocess_dataset(
        classification_dataframe,
        classification_settings,
    )


def test_build_classification_models_returns_expected_candidates():
    models = build_classification_models(random_seed=42)

    assert set(models) == {
        "logistic_regression",
        "random_forest",
    }


def test_models_receive_configured_random_seed():
    models = build_classification_models(random_seed=123)

    assert models["logistic_regression"].random_state == 123
    assert models["random_forest"].random_state == 123


def test_models_are_returned_unfitted():
    models = build_classification_models(random_seed=42)

    for model in models.values():
        assert not hasattr(model, "classes_")


def test_evaluate_classifier_returns_all_metrics(
    preprocessed_classification_data,
):
    models = build_classification_models(random_seed=42)

    model = models["logistic_regression"]

    model.fit(
        preprocessed_classification_data.X_train,
        preprocessed_classification_data.y_train,
    )

    metrics = evaluate_classifier(
        model,
        preprocessed_classification_data.X_validation,
        preprocessed_classification_data.y_validation,
    )

    assert isinstance(metrics, ClassificationMetrics)

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0


def test_metric_dictionary_contains_expected_values():
    metrics = ClassificationMetrics(
        accuracy=0.90,
        precision=0.88,
        recall=0.87,
        f1=0.875,
    )

    assert metrics.as_dict() == {
        "accuracy": 0.90,
        "precision": 0.88,
        "recall": 0.87,
        "f1": 0.875,
    }


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("accuracy", 0.90),
        ("precision", 0.88),
        ("recall", 0.87),
        ("f1", 0.875),
        (" F1 ", 0.875),
    ],
)
def test_metric_lookup_is_case_and_whitespace_tolerant(
    metric_name,
    expected,
):
    metrics = ClassificationMetrics(
        accuracy=0.90,
        precision=0.88,
        recall=0.87,
        f1=0.875,
    )

    assert metrics.get(metric_name) == expected


def test_invalid_metric_is_rejected():
    metrics = ClassificationMetrics(
        accuracy=0.90,
        precision=0.88,
        recall=0.87,
        f1=0.875,
    )

    with pytest.raises(DataValidationError, match="Unsupported"):
        metrics.get("roc_auc")


def test_best_model_is_selected_using_validation_metric():
    validation_results = {
        "model_a": ClassificationMetrics(
            accuracy=0.80,
            precision=0.82,
            recall=0.79,
            f1=0.80,
        ),
        "model_b": ClassificationMetrics(
            accuracy=0.91,
            precision=0.89,
            recall=0.90,
            f1=0.895,
        ),
    }

    assert select_best_model(
        validation_results,
        "accuracy",
    ) == "model_b"


def test_best_model_selection_can_use_f1():
    validation_results = {
        "model_a": ClassificationMetrics(
            accuracy=0.90,
            precision=0.70,
            recall=0.80,
            f1=0.85,
        ),
        "model_b": ClassificationMetrics(
            accuracy=0.88,
            precision=0.92,
            recall=0.90,
            f1=0.91,
        ),
    }

    assert select_best_model(
        validation_results,
        "f1",
    ) == "model_b"


def test_empty_validation_results_are_rejected():
    with pytest.raises(DataValidationError, match="no validation"):
        select_best_model({}, "accuracy")


def test_invalid_selection_metric_is_rejected():
    validation_results = {
        "model_a": ClassificationMetrics(
            accuracy=0.90,
            precision=0.90,
            recall=0.90,
            f1=0.90,
        )
    }

    with pytest.raises(DataValidationError, match="Unsupported"):
        select_best_model(validation_results, "roc_auc")


def test_full_training_pipeline_returns_best_model(
    preprocessed_classification_data,
    classification_settings,
):
    result = train_models(
        preprocessed_classification_data,
        classification_settings,
    )

    assert result.best_model_name in {
        "logistic_regression",
        "random_forest",
    }

    assert result.best_model is result.trained_models[
        result.best_model_name
    ]

    assert result.best_model_name in result.validation_results

    assert 0.0 <= result.test_metrics.accuracy <= 1.0
    assert 0.0 <= result.test_metrics.precision <= 1.0
    assert 0.0 <= result.test_metrics.recall <= 1.0
    assert 0.0 <= result.test_metrics.f1 <= 1.0


def test_all_successful_models_have_validation_results(
    preprocessed_classification_data,
    classification_settings,
):
    result = train_models(
        preprocessed_classification_data,
        classification_settings,
    )

    assert set(result.trained_models) == set(
        result.validation_results
    )


def test_training_is_deterministic(
    preprocessed_classification_data,
    classification_settings,
):
    first_result = train_models(
        preprocessed_classification_data,
        classification_settings,
    )

    second_result = train_models(
        preprocessed_classification_data,
        classification_settings,
    )

    assert first_result.best_model_name == second_result.best_model_name

    assert first_result.validation_results == (
        second_result.validation_results
    )

    assert first_result.test_metrics == second_result.test_metrics


def test_training_rejects_empty_training_data(
    preprocessed_classification_data,
    classification_settings,
):
    invalid_data = replace(
        preprocessed_classification_data,
        X_train=np.empty(
            (
                0,
                preprocessed_classification_data.X_train.shape[1],
            )
        ),
    )

    with pytest.raises(
        DataValidationError,
        match="Training dataset is empty",
    ):
        train_models(
            invalid_data,
            classification_settings,
        )


def test_training_rejects_empty_validation_data(
    preprocessed_classification_data,
    classification_settings,
):
    invalid_data = replace(
        preprocessed_classification_data,
        X_validation=np.empty(
            (
                0,
                preprocessed_classification_data.X_validation.shape[1],
            )
        ),
    )

    with pytest.raises(
        DataValidationError,
        match="Validation dataset is empty",
    ):
        train_models(
            invalid_data,
            classification_settings,
        )


def test_training_rejects_empty_test_data(
    preprocessed_classification_data,
    classification_settings,
):
    invalid_data = replace(
        preprocessed_classification_data,
        X_test=np.empty(
            (
                0,
                preprocessed_classification_data.X_test.shape[1],
            )
        ),
    )

    with pytest.raises(
        DataValidationError,
        match="Test dataset is empty",
    ):
        train_models(
            invalid_data,
            classification_settings,
        )