from __future__ import annotations

import json
import math

import pytest

from automl.artifacts import RunArtifacts
from automl.config.settings import EvaluationGateConfig, Settings
from automl.exceptions import ConfigurationError, TrainingError
from automl.evaluation.gate import evaluate_promotion_gate
from automl.evaluation.metrics import ClassificationMetrics
from automl.models.trainer import TrainingSelectionResult
from automl.orchestration import apply_validation_promotion_gate


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [(0.91, 0.9, True), (0.9, 0.9, True), (0.89, 0.9, False)],
)
def test_gate_compares_selected_validation_score(score, threshold, expected):
    result = evaluate_promotion_gate(
        {"accuracy": score},
        EvaluationGateConfig(
            metric_name="accuracy",
            minimum_validation_score=threshold,
        ),
    )

    assert result.passed is expected
    assert result.metric_name == "accuracy"
    assert result.observed_score == score
    assert result.required_threshold == threshold
    assert result.reason


@pytest.mark.parametrize(
    "config",
    [
        None,
        EvaluationGateConfig(),
        EvaluationGateConfig(metric_name="accuracy"),
        EvaluationGateConfig(minimum_validation_score=0.8),
    ],
)
def test_gate_rejects_missing_configuration(config):
    with pytest.raises(ConfigurationError, match="Evaluation gate"):
        evaluate_promotion_gate({"accuracy": 0.9}, config)


def test_gate_rejects_missing_metric():
    with pytest.raises(TrainingError, match="do not contain 'accuracy'"):
        evaluate_promotion_gate(
            {"f1": 0.9},
            EvaluationGateConfig(
                metric_name="accuracy",
                minimum_validation_score=0.8,
            ),
        )


def test_gate_can_compare_yes_class_f1_from_validation_metrics():
    metrics = ClassificationMetrics(
        accuracy=0.9,
        precision=0.9,
        recall=0.9,
        f1=0.9,
        f1_yes=0.42,
    )

    result = evaluate_promotion_gate(
        metrics,
        EvaluationGateConfig(
            metric_name="f1_yes",
            minimum_validation_score=0.4,
        ),
    )

    assert result.metric_name == "f1_yes"
    assert result.observed_score == 0.42
    assert result.passed is True


def test_gate_rejects_missing_yes_class_f1():
    metrics = ClassificationMetrics(
        accuracy=0.9,
        precision=0.9,
        recall=0.9,
        f1=0.9,
    )

    with pytest.raises(TrainingError, match="do not contain 'f1_yes'"):
        evaluate_promotion_gate(
            metrics,
            EvaluationGateConfig(
                metric_name="f1_yes",
                minimum_validation_score=0.5,
            ),
        )


def test_gate_fails_closed_for_unsupported_metric():
    with pytest.raises(TrainingError, match="do not contain 'roc_auc'"):
        evaluate_promotion_gate(
            {"accuracy": 0.9, "f1_yes": 0.7},
            EvaluationGateConfig(
                metric_name="roc_auc",
                minimum_validation_score=0.5,
            ),
        )


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_gate_rejects_non_finite_validation_score(score):
    with pytest.raises(TrainingError, match="must be finite"):
        evaluate_promotion_gate(
            {"accuracy": score},
            EvaluationGateConfig(
                metric_name="accuracy",
                minimum_validation_score=0.8,
            ),
        )


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
def test_gate_rejects_non_finite_threshold(threshold):
    with pytest.raises(ConfigurationError, match="finite"):
        evaluate_promotion_gate(
            {"accuracy": 0.9},
            EvaluationGateConfig(
                metric_name="accuracy",
                minimum_validation_score=threshold,
            ),
        )


def _selection_handoff(tmp_path, *, score: float, threshold: float):
    artifacts = RunArtifacts.create(tmp_path / "artifacts", run_id="gate-run")
    settings = Settings(
        evaluation_gate={
            "metric_name": "accuracy",
            "minimum_validation_score": threshold,
        }
    )
    settings_ref = artifacts.write_json("settings.json", settings.model_dump(mode="json"))
    validation_ref = artifacts.write_json(
        "validation.json",
        {"warnings": ["Duplicate input rows were found"]},
    )
    selection = TrainingSelectionResult(
        trained_models={},
        validation_results={
            "chosen": ClassificationMetrics(
                accuracy=score,
                precision=score,
                recall=score,
                f1=score,
            ),
        },
        best_model_name="chosen",
        best_model=None,
        failed_models={},
    )
    selection_ref = artifacts.write_joblib("selection.joblib", selection)
    preprocessing_ref = artifacts.write_joblib("preprocessed.joblib", {"placeholder": True})
    return artifacts, {
        "artifact_root": str(artifacts.root),
        "run_id": artifacts.run_id,
        "settings_ref": settings_ref,
        "validation_ref": validation_ref,
        "selection_ref": selection_ref,
        "preprocessing_ref": preprocessing_ref,
    }


def test_gate_writes_durable_decision_and_preserves_validation_warnings(tmp_path):
    artifacts, handoff = _selection_handoff(tmp_path, score=0.91, threshold=0.9)

    result = apply_validation_promotion_gate(handoff)

    assert result["promotion_approved"] is True
    gate_report = artifacts.read_json(result["promotion_gate_ref"])
    assert gate_report == {
        "metric_name": "accuracy",
        "observed_score": 0.91,
        "required_threshold": 0.9,
        "passed": True,
        "reason": "Validation accuracy score 0.91 meets minimum 0.9",
        "run_id": "gate-run",
        "selected_model_name": "chosen",
        "validation_warnings": ["Duplicate input rows were found"],
    }
    assert json.loads((artifacts.directory / "promotion_gate.json").read_text()) == gate_report


def test_rejected_gate_persists_failure_and_does_not_publish_or_replace_approved_artifact(
    tmp_path,
):
    artifacts, handoff = _selection_handoff(tmp_path, score=0.7, threshold=0.8)
    previous_approved = {"model": "previous-approved"}
    artifacts.write_json("approved/manifest.json", previous_approved)

    with pytest.raises(TrainingError, match="Model promotion rejected"):
        apply_validation_promotion_gate(handoff)

    report = artifacts.read_json("promotion_gate.json")
    assert report["passed"] is False
    assert "below minimum" in report["reason"]
    assert artifacts.read_json("approved/manifest.json") == previous_approved
    assert not (artifacts.directory / "approved" / "model.joblib").exists()
