from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import yaml
from sklearn.pipeline import Pipeline

from automl.artifacts import RunArtifacts
from automl.exceptions import TrainingError
from automl.orchestration import run_training_pipeline
import automl.orchestration.training as orchestration_training


def _classification_frame() -> pd.DataFrame:
    rng = np.random.default_rng(812)
    per_class = 60
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "customerID": [
                        f"customer-{index:03d}"
                        for index in range(per_class)
                    ],
                    "signal": rng.normal(-2.5, 0.45, per_class),
                    "group": ["control"] * per_class,
                    "target": [0] * per_class,
                }
            ),
            pd.DataFrame(
                {
                    "customerID": [
                        f"customer-{index + per_class:03d}"
                        for index in range(per_class)
                    ],
                    "signal": rng.normal(2.5, 0.45, per_class),
                    "group": ["treated"] * per_class,
                    "target": [1] * per_class,
                }
            ),
        ],
        ignore_index=True,
    )


def _write_inputs(
    tmp_path: Path,
    *,
    threshold: float,
    gate_metric: str = "accuracy",
    frame: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    input_path = tmp_path / "synthetic.csv"
    (frame if frame is not None else _classification_frame()).to_csv(
        input_path, index=False
    )
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"random_seed": 812},
                "data": {
                    "input_path": str(input_path),
                    "target_column": "target",
                    "excluded_feature_columns": ["customerID"],
                    "test_size": 0.2,
                    "validation_size": 0.2,
                },
                "training": {"random_seed": 812, "metric": "accuracy"},
                "evaluation_gate": {
                    "metric_name": gate_metric,
                    "minimum_validation_score": threshold,
                },
            }
        ),
        encoding="utf-8",
    )
    return input_path, config_path


def test_complete_package_pipeline_publishes_consistent_bundle_and_manifest(
    tmp_path, monkeypatch
):
    from automl.evaluation.metrics import ClassificationMetrics

    # Exercise the publication path without evaluating the held-out test split.
    monkeypatch.setattr(
        orchestration_training,
        "evaluate_classifier",
        lambda *_args, **_kwargs: ClassificationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
        ),
    )
    input_path, config_path = _write_inputs(tmp_path, threshold=0.5)

    result = run_training_pipeline(
        tmp_path / "artifacts",
        config_path=config_path,
        data_path=input_path,
        run_id="accepted-run",
    )

    artifacts = RunArtifacts.open(result["artifact_root"], result["run_id"])
    manifest = artifacts.read_json(result["approved_manifest_ref"])
    bundle = artifacts.read_joblib(result["inference_bundle_ref"])

    assert result["run_id"] == "accepted-run"
    assert manifest["run_id"] == result["run_id"]
    assert manifest["selected_model_name"] == result["selected_model_name"]
    assert manifest["input"]["reference"] == "input.csv"
    assert manifest["input"]["source_path"] == str(input_path.resolve())
    assert len(manifest["input"]["fingerprint"]) == 64
    assert manifest["resolved_configuration"]["data"]["input_path"] == str(
        input_path.resolve()
    )
    assert manifest["resolved_configuration"]["evaluation_gate"] == {
        "metric_name": "accuracy",
        "minimum_validation_score": 0.5,
    }
    assert manifest["gate_decision"]["passed"] is True
    assert manifest["validation_metrics"]
    candidate_metrics = manifest["candidate_validation_metrics"]
    assert set(candidate_metrics) == {"logistic_regression", "random_forest"}
    assert candidate_metrics[manifest["selected_model_name"]] == manifest[
        "validation_metrics"
    ]
    assert all("test_metrics" not in metrics for metrics in candidate_metrics.values())
    assert manifest["final_test_metrics"] == result["final_test_metrics"]
    assert manifest["artifacts"]["inference_bundle_ref"] == result[
        "inference_bundle_ref"
    ]
    assert manifest["artifacts"]["test_evaluation_ref"] == result[
        "test_evaluation_ref"
    ]
    assert manifest["created_at_utc"].endswith("Z")
    assert isinstance(bundle, Pipeline)
    assert bundle.named_steps["preprocessor"] is not None
    assert bundle.named_steps["estimator"] is not None
    raw_input = _classification_frame().drop(columns=["target"]).iloc[:5]
    assert "customerID" in raw_input.columns
    assert bundle.predict(raw_input).shape == (5,)
    assert manifest["selection_metric"] == "accuracy"
    assert manifest["training_completed_at_utc"].endswith("Z")
    assert manifest["feature_schema"]["model_input_columns"] == ["signal", "group"]
    assert "customerID" not in manifest["feature_schema"]["model_input_columns"]
    assert "customerID" in manifest["input_schema"]["columns"]
    assert "customerID" in artifacts.read_csv(manifest["input"]["reference"]).columns
    assert "customerID" not in bundle.named_steps["preprocessor"].feature_names_in_
    assert "python" in manifest["library_versions"]
    assert "column_dtypes" in manifest["input_schema"]


def test_gate_rejection_skips_test_evaluation_and_approved_publication(
    tmp_path, monkeypatch
):
    input_path, config_path = _write_inputs(tmp_path, threshold=1.01)
    evaluated: list[bool] = []

    def unexpected_test_evaluation(*args, **kwargs):
        evaluated.append(True)
        raise AssertionError("test split must not be evaluated after gate rejection")

    monkeypatch.setattr(
        orchestration_training,
        "evaluate_classifier",
        unexpected_test_evaluation,
    )
    artifact_root = tmp_path / "artifacts"

    with pytest.raises(TrainingError, match="Model promotion rejected"):
        run_training_pipeline(
            artifact_root,
            config_path=config_path,
            data_path=input_path,
            run_id="rejected-run",
        )

    artifacts = RunArtifacts.open(artifact_root, "rejected-run")
    assert evaluated == []
    assert artifacts.read_json("promotion_gate.json")["passed"] is False
    assert not (artifacts.directory / "approved").exists()


def test_f1_yes_gate_rejection_skips_test_evaluation_and_publication(
    tmp_path, monkeypatch
):
    churn_frame = _classification_frame()
    churn_frame["target"] = np.where(churn_frame["target"] == 1, "Yes", "No")
    input_path, config_path = _write_inputs(
        tmp_path,
        threshold=1.01,
        gate_metric="f1_yes",
        frame=churn_frame,
    )
    evaluated: list[bool] = []

    def unexpected_test_evaluation(*args, **kwargs):
        evaluated.append(True)
        raise AssertionError("test split must not be evaluated after gate rejection")

    monkeypatch.setattr(
        orchestration_training,
        "evaluate_classifier",
        unexpected_test_evaluation,
    )
    artifact_root = tmp_path / "artifacts"

    with pytest.raises(TrainingError, match="Model promotion rejected"):
        run_training_pipeline(
            artifact_root,
            config_path=config_path,
            data_path=input_path,
            run_id="rejected-f1-yes-run",
        )

    artifacts = RunArtifacts.open(artifact_root, "rejected-f1-yes-run")
    gate_report = artifacts.read_json("promotion_gate.json")
    assert evaluated == []
    assert gate_report["metric_name"] == "f1_yes"
    assert gate_report["passed"] is False
    assert not (artifacts.directory / "approved").exists()


def test_successful_retries_are_isolated_in_distinct_run_directories(tmp_path):
    input_path, config_path = _write_inputs(tmp_path, threshold=0.5)
    artifact_root = tmp_path / "artifacts"

    first = run_training_pipeline(
        artifact_root,
        config_path=config_path,
        data_path=input_path,
    )
    second = run_training_pipeline(
        artifact_root,
        config_path=config_path,
        data_path=input_path,
    )

    assert first["run_id"] != second["run_id"]
    assert first["approved_manifest_ref"] != second["approved_manifest_ref"]
    first_artifacts = RunArtifacts.open(artifact_root, first["run_id"])
    second_artifacts = RunArtifacts.open(artifact_root, second["run_id"])
    first_manifest = first_artifacts.read_json(first["approved_manifest_ref"])
    second_manifest = second_artifacts.read_json(second["approved_manifest_ref"])
    assert first_manifest["run_id"] == first["run_id"]
    assert second_manifest["run_id"] == second["run_id"]
    assert first_manifest["input"]["fingerprint"] == second_manifest["input"][
        "fingerprint"
    ]
