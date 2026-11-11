from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from automl.config.settings import Settings
from automl.evaluation.metrics import ClassificationMetrics
from automl.exceptions import DataValidationError
from automl.models import trainer
from automl.models.trainer import TrainingResult, TrainingSelectionResult
from automl.orchestration import (
    build_inference_pipeline,
    ingest_dataset,
    preprocess_validated_dataset,
    train_and_select_artifact,
    validate_ingested_dataset,
)
from automl.preprocessing import PreprocessingResult, preprocess_dataset


def _classification_frame() -> pd.DataFrame:
    rng = np.random.default_rng(29)
    rows_per_class = 40
    class_zero = pd.DataFrame(
        {
            "signal": rng.normal(0.0, 0.4, rows_per_class),
            "category": ["low"] * rows_per_class,
            "target": [0] * rows_per_class,
        }
    )
    class_one = pd.DataFrame(
        {
            "signal": rng.normal(3.0, 0.4, rows_per_class),
            "category": ["high"] * rows_per_class,
            "target": [1] * rows_per_class,
        }
    )
    return pd.concat([class_zero, class_one], ignore_index=True)


def _settings(
    *, metric: str = "accuracy", excluded_feature_columns: list[str] | None = None
) -> Settings:
    return Settings(
        project={"random_seed": 29},
        data={
            "input_path": "unused.csv",
            "target_column": "target",
            "excluded_feature_columns": excluded_feature_columns or [],
            "test_size": 0.2,
            "validation_size": 0.2,
        },
        training={"random_seed": 29, "metric": metric},
    )


def _write_settings(path: Path) -> None:
    import yaml

    path.write_text(
        yaml.safe_dump(_settings().model_dump(mode="json")),
        encoding="utf-8",
    )


def _assert_small_metadata(value: Any) -> None:
    """Assert a task result contains only primitive XCom-safe values."""

    assert not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
    assert not isinstance(value, PreprocessingResult)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_small_metadata(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_small_metadata(nested)
    else:
        assert value is None or isinstance(value, (str, int, float, bool))


def test_task_adapters_use_durable_handoffs_and_preserve_warnings(tmp_path):
    input_path = tmp_path / "synthetic.csv"
    frame = _classification_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(input_path, index=False)
    config_path = tmp_path / "config.yaml"
    _write_settings(config_path)
    artifact_root = tmp_path / "run-artifacts"

    ingested = ingest_dataset(
        artifact_root,
        config_path=config_path,
        data_path=input_path,
        run_id="synthetic-run",
    )
    validated = validate_ingested_dataset(ingested)
    preprocessed = preprocess_validated_dataset(validated)
    selected = train_and_select_artifact(preprocessed)

    assert validated["warnings"]
    assert any("duplicate rows" in warning for warning in validated["warnings"])
    assert validated["fingerprint"]
    assert preprocessed["feature_count"] > 0
    assert selected["best_model_name"] in {
        "logistic_regression",
        "random_forest",
    }
    assert selected["selection_metric"] == "accuracy"
    assert 0.0 <= selected["selection_score"] <= 1.0

    for handoff in (ingested, validated, preprocessed, selected):
        _assert_small_metadata(handoff)


def test_validation_adapter_fails_clearly_for_invalid_dataset(tmp_path):
    input_path = tmp_path / "missing-target.csv"
    pd.DataFrame({"feature": [0, 1, 2, 3], "wrong_target": [0, 1, 0, 1]}).to_csv(
        input_path,
        index=False,
    )
    config_path = tmp_path / "config.yaml"
    _write_settings(config_path)
    ingested = ingest_dataset(
        tmp_path / "run-artifacts",
        config_path=config_path,
        data_path=input_path,
        run_id="invalid-run",
    )

    with pytest.raises(DataValidationError, match="target column is missing"):
        validate_ingested_dataset(ingested)


def test_validation_only_training_does_not_access_test_split(monkeypatch):
    frame = _classification_frame()
    settings = _settings()
    preprocessed = preprocess_dataset(frame, settings)
    no_test_split = replace(
        preprocessed,
        X_test=np.empty((0, preprocessed.X_test.shape[1])),
        y_test=pd.Series(dtype=preprocessed.y_test.dtype),
    )
    evaluated_splits: list[np.ndarray] = []
    original_evaluator = trainer.evaluate_classifier

    def record_validation_evaluation(model, X, y):
        evaluated_splits.append(np.asarray(X))
        return original_evaluator(model, X, y)

    monkeypatch.setattr(trainer, "evaluate_classifier", record_validation_evaluation)

    result = trainer.train_and_select_models(no_test_split, settings)

    assert isinstance(result, TrainingSelectionResult)
    assert not hasattr(result, "test_metrics")
    assert len(evaluated_splits) == len(result.trained_models)
    assert all(
        np.array_equal(split, preprocessed.X_validation)
        for split in evaluated_splits
    )


def test_f1_yes_selects_models_on_the_same_validation_split(monkeypatch):
    frame = _classification_frame()
    frame["customerID"] = [f"customer-{index}" for index in range(len(frame))]
    frame["target"] = np.where(frame["target"] == 1, "Yes", "No")
    settings = _settings(
        metric="f1_yes",
        excluded_feature_columns=["customerID"],
    )
    preprocessed = preprocess_dataset(frame, settings)
    no_test_split = replace(
        preprocessed,
        X_test=np.empty((0, preprocessed.X_test.shape[1])),
        y_test=pd.Series(dtype=preprocessed.y_test.dtype),
    )
    evaluated: list[tuple[str, np.ndarray]] = []

    def controlled_validation_metrics(model, X, y):
        evaluated.append((type(model).__name__, np.asarray(X).copy()))
        if type(model).__name__ == "LogisticRegression":
            return ClassificationMetrics(
                accuracy=0.99, precision=0.99, recall=0.99, f1=0.99, f1_yes=0.2
            )
        return ClassificationMetrics(
            accuracy=0.6, precision=0.6, recall=0.6, f1=0.6, f1_yes=0.8
        )

    monkeypatch.setattr(trainer, "evaluate_classifier", controlled_validation_metrics)

    result = trainer.train_and_select_models(no_test_split, settings)

    assert result.best_model_name == "random_forest"
    assert result.validation_results["logistic_regression"].accuracy == 0.99
    assert result.validation_results["random_forest"].accuracy == 0.6
    assert len(evaluated) == 2
    assert {name for name, _ in evaluated} == {
        "LogisticRegression",
        "RandomForestClassifier",
    }
    assert all(
        np.array_equal(features, preprocessed.X_validation)
        for _, features in evaluated
    )
    inference_pipeline = build_inference_pipeline(
        preprocessed,
        result.best_model,
    )
    assert inference_pipeline.named_steps["preprocessor"] is preprocessed.preprocessor
    assert inference_pipeline.named_steps["estimator"] is result.best_model


def test_train_models_compatibility_wrapper_evaluates_test_once(monkeypatch):
    settings = _settings()
    preprocessed = preprocess_dataset(_classification_frame(), settings)
    original_evaluator = trainer.evaluate_classifier
    evaluated_splits: list[np.ndarray] = []

    def record_evaluation(model, X, y):
        evaluated_splits.append(np.asarray(X))
        return original_evaluator(model, X, y)

    monkeypatch.setattr(trainer, "evaluate_classifier", record_evaluation)

    result = trainer.train_models(preprocessed, settings)

    assert isinstance(result, TrainingResult)
    assert result.best_model is result.trained_models[result.best_model_name]
    assert result.test_metrics.accuracy >= 0.0
    assert sum(
        np.array_equal(split, preprocessed.X_test) for split in evaluated_splits
    ) == 1
    assert len(evaluated_splits) == len(result.trained_models) + 1
