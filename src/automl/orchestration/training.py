"""Durable task adapters around the existing AutoML data and training APIs.

Each adapter returns a small JSON-serializable mapping containing run metadata
and artifact references. DataFrames, transformed arrays, fitted preprocessors,
and models remain in the shared artifact store rather than crossing task
boundaries in Airflow XCom.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import platform
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from automl.artifacts import RunArtifacts
from automl.config.settings import Settings, load_settings
from automl.data.ingestion import load_csv_data
from automl.data.validation import validate_dataset
from automl.exceptions import ModelArtifactError, TrainingError
from automl.evaluation.gate import evaluate_promotion_gate
from automl.evaluation.metrics import evaluate_classifier
from automl.logging_config import get_logger
from automl.models.trainer import TrainingSelectionResult, train_and_select_models
from automl.preprocessing import PreprocessingResult, preprocess_dataset


logger = get_logger(__name__)

_SETTINGS_REF = "settings.json"
_INGESTION_REF = "ingestion.json"
_VALIDATION_REF = "validation.json"
_PREPROCESSING_REF = "preprocessed.joblib"
_SELECTION_REF = "selection.joblib"
_GATE_REF = "promotion_gate.json"


def ingest_dataset(
    artifact_root: str | Path,
    config_path: str | Path | None = None,
    data_path: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Load configured CSV data and persist it with the run settings snapshot."""

    settings = load_settings(config_path)
    data_frame = load_csv_data(settings, data_path=data_path)
    source_path = Path(data_path or settings.data.input_path).expanduser().resolve()
    artifacts = RunArtifacts.create(artifact_root, run_id=run_id)

    dataset_ref = artifacts.write_csv("input.csv", data_frame)
    settings_snapshot = settings.model_dump(mode="json")
    settings_snapshot["data"]["input_path"] = str(source_path)
    settings_ref = artifacts.write_json(
        _SETTINGS_REF,
        settings_snapshot,
    )
    ingestion_ref = artifacts.write_json(
        _INGESTION_REF,
        {
            "dataset_ref": dataset_ref,
            "settings_ref": settings_ref,
            "source_path": str(source_path),
            "row_count": int(len(data_frame)),
            "column_count": int(len(data_frame.columns)),
        },
    )

    logger.info(
        "Persisted %s input rows and %s columns for run %s",
        len(data_frame),
        len(data_frame.columns),
        artifacts.run_id,
    )
    return _handoff(
        artifacts,
        dataset_ref=dataset_ref,
        settings_ref=settings_ref,
        ingestion_ref=ingestion_ref,
        row_count=int(len(data_frame)),
        column_count=int(len(data_frame.columns)),
    )


def validate_ingested_dataset(ingestion: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the persisted CSV and return warning and metadata references."""

    artifacts = _open_run(ingestion)
    ingestion_info = artifacts.read_json(_reference(ingestion, "ingestion_ref"))
    settings = _read_settings(artifacts, ingestion_info["settings_ref"])
    data_frame = artifacts.read_csv(ingestion_info["dataset_ref"])

    validation = validate_dataset(
        data_frame,
        target_column=settings.target_column,
        source_path=ingestion_info["source_path"],
    )
    validation_ref = artifacts.write_json(
        _VALIDATION_REF,
        {
            "metadata": {
                **asdict(validation.metadata),
                "column_dtypes": {
                    str(column): str(dtype)
                    for column, dtype in data_frame.dtypes.items()
                },
            },
            "warnings": validation.warnings,
        },
    )

    return _handoff(
        artifacts,
        dataset_ref=ingestion_info["dataset_ref"],
        settings_ref=ingestion_info["settings_ref"],
        validation_ref=validation_ref,
        row_count=validation.metadata.row_count,
        column_count=validation.metadata.column_count,
        fingerprint=validation.metadata.fingerprint,
        warnings=validation.warnings,
    )


def preprocess_validated_dataset(
    validation_handoff: Mapping[str, Any],
) -> dict[str, Any]:
    """Preprocess the validated CSV and persist the fitted result with joblib."""

    artifacts = _open_run(validation_handoff)
    validation_report = artifacts.read_json(_reference(validation_handoff, "validation_ref"))
    ingestion_info = artifacts.read_json(_INGESTION_REF)
    settings_ref = _reference(validation_handoff, "settings_ref")
    settings = _read_settings(artifacts, settings_ref)
    data_frame = artifacts.read_csv(_reference(validation_handoff, "dataset_ref"))

    if validation_report["metadata"]["target_column"] != settings.target_column:
        raise ModelArtifactError(
            "Validation report target does not match the run settings"
        )

    preprocessed = preprocess_dataset(data_frame, settings)
    preprocessing_ref = artifacts.write_joblib(_PREPROCESSING_REF, preprocessed)

    return _handoff(
        artifacts,
        dataset_ref=ingestion_info["dataset_ref"],
        settings_ref=settings_ref,
        validation_ref=_reference(validation_handoff, "validation_ref"),
        preprocessing_ref=preprocessing_ref,
        train_rows=int(preprocessed.X_train.shape[0]),
        validation_rows=int(preprocessed.X_validation.shape[0]),
        test_rows=int(preprocessed.X_test.shape[0]),
        feature_count=int(preprocessed.X_train.shape[1]),
    )


def train_and_select_artifact(
    preprocessing_handoff: Mapping[str, Any],
) -> dict[str, Any]:
    """Train candidates, select by validation metrics, and persist the result."""

    artifacts = _open_run(preprocessing_handoff)
    settings = _read_settings(
        artifacts,
        _reference(preprocessing_handoff, "settings_ref"),
    )
    preprocessed = artifacts.read_joblib(
        _reference(preprocessing_handoff, "preprocessing_ref")
    )
    if not isinstance(preprocessed, PreprocessingResult):
        raise ModelArtifactError(
            "Preprocessing artifact does not contain a PreprocessingResult"
        )

    selection = train_and_select_models(preprocessed, settings)
    if not isinstance(selection, TrainingSelectionResult):
        raise ModelArtifactError(
            "Trainer did not return a TrainingSelectionResult"
        )
    selection_ref = artifacts.write_joblib(_SELECTION_REF, selection)
    selected_metrics = selection.validation_results[selection.best_model_name]

    return _handoff(
        artifacts,
        settings_ref=_reference(preprocessing_handoff, "settings_ref"),
        preprocessing_ref=_reference(preprocessing_handoff, "preprocessing_ref"),
        validation_ref=_reference(preprocessing_handoff, "validation_ref"),
        selection_ref=selection_ref,
        best_model_name=selection.best_model_name,
        selection_metric=settings.training.metric,
        selection_score=selected_metrics.get(settings.training.metric),
        validation_results={
            model_name: metrics.as_dict()
            for model_name, metrics in selection.validation_results.items()
        },
        failed_models=selection.failed_models,
    )


def apply_validation_promotion_gate(
    selection_handoff: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the validation gate decision and fail the run when rejected.

    A rejected decision is written before raising, leaving downstream test
    evaluation/publication steps unable to proceed through normal task
    dependencies. This adapter never writes an approved model artifact.
    """

    artifacts = _open_run(selection_handoff)
    settings = _read_settings(
        artifacts,
        _reference(selection_handoff, "settings_ref"),
    )
    selection = artifacts.read_joblib(_reference(selection_handoff, "selection_ref"))
    if not isinstance(selection, TrainingSelectionResult):
        raise ModelArtifactError(
            "Selection artifact does not contain a TrainingSelectionResult"
        )
    try:
        selected_metrics = selection.validation_results[selection.best_model_name]
    except KeyError as exc:
        raise TrainingError(
            "Selected model has no validation metrics for the promotion gate"
        ) from exc

    decision = evaluate_promotion_gate(
        selected_metrics,
        settings.evaluation_gate,
    )
    validation_report = artifacts.read_json(
        _reference(selection_handoff, "validation_ref")
    )
    report = {
        **decision.as_dict(),
        "run_id": artifacts.run_id,
        "selected_model_name": selection.best_model_name,
        "validation_warnings": validation_report.get("warnings", []),
    }
    gate_ref = artifacts.write_json(_GATE_REF, report)
    handoff = _handoff(
        artifacts,
        settings_ref=_reference(selection_handoff, "settings_ref"),
        preprocessing_ref=_reference(selection_handoff, "preprocessing_ref"),
        selection_ref=_reference(selection_handoff, "selection_ref"),
        validation_ref=_reference(selection_handoff, "validation_ref"),
        promotion_gate_ref=gate_ref,
        selected_model_name=selection.best_model_name,
        metric_name=decision.metric_name,
        observed_score=decision.observed_score,
        required_threshold=decision.required_threshold,
        promotion_approved=decision.passed,
    )
    if not decision.passed:
        raise TrainingError(f"Model promotion rejected: {decision.reason}")
    return handoff


def evaluate_and_publish_approved_model(
    gate_handoff: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the held-out test split and publish an approved inference bundle.

    The durable gate report is checked before the test split is accessed. A
    unique publication directory and manifest-last write make incomplete
    retries distinguishable from completed approved publications.
    """

    artifacts = _open_run(gate_handoff)
    gate_ref = _reference(gate_handoff, "promotion_gate_ref")
    gate_report = artifacts.read_json(gate_ref)
    if not isinstance(gate_report, dict):
        raise ModelArtifactError("Promotion gate artifact is not a JSON object")
    if gate_report.get("passed") is not True or gate_handoff.get(
        "promotion_approved"
    ) is not True:
        raise TrainingError("Approved model publication requires a passing gate")

    settings_ref = _reference(gate_handoff, "settings_ref")
    settings = _read_settings(artifacts, settings_ref)
    preprocessing = artifacts.read_joblib(
        _reference(gate_handoff, "preprocessing_ref")
    )
    selection = artifacts.read_joblib(_reference(gate_handoff, "selection_ref"))
    if not isinstance(preprocessing, PreprocessingResult):
        raise ModelArtifactError(
            "Preprocessing artifact does not contain a PreprocessingResult"
        )
    if not isinstance(selection, TrainingSelectionResult):
        raise ModelArtifactError(
            "Selection artifact does not contain a TrainingSelectionResult"
        )
    if selection.best_model_name != gate_report.get("selected_model_name"):
        raise ModelArtifactError(
            "Promotion gate decision does not match the selected model"
        )
    if (
        selection.best_model is None
        or selection.best_model_name not in selection.trained_models
        or selection.best_model_name not in selection.validation_results
    ):
        raise ModelArtifactError(
            "Selection artifact is missing the selected model or its validation metrics"
        )

    # This is the first point in the workflow that reads the held-out test split.
    try:
        test_metrics = evaluate_classifier(
            selection.best_model,
            preprocessing.X_test,
            preprocessing.y_test.to_numpy(),
        )
    except Exception as exc:
        raise TrainingError(
            f"Final held-out test evaluation failed for model "
            f"'{selection.best_model_name}': {exc}"
        ) from exc
    ingestion = artifacts.read_json(_INGESTION_REF)
    validation_report = artifacts.read_json(
        _reference(gate_handoff, "validation_ref")
    )
    if (
        not isinstance(validation_report, dict)
        or not isinstance(validation_report.get("metadata"), dict)
        or not isinstance(validation_report["metadata"].get("fingerprint"), str)
    ):
        raise ModelArtifactError("Validation report is missing its input fingerprint")
    publication_id = uuid4().hex
    publication_prefix = f"approved/{publication_id}"
    bundle_ref = artifacts.write_joblib(
        f"{publication_prefix}/inference_bundle.joblib",
        build_inference_pipeline(preprocessing, selection.best_model),
    )
    test_metrics_dict = test_metrics.as_dict()
    evaluation_ref = artifacts.write_json(
        f"{publication_prefix}/test_evaluation.json",
        test_metrics_dict,
    )
    manifest = {
        "schema_version": 1,
        "run_id": artifacts.run_id,
        "publication_id": publication_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "training_completed_at_utc": selection.trained_at_utc,
        "selected_model_name": selection.best_model_name,
        "selection_metric": settings.training.metric,
        "selection_score": selection.validation_results[
            selection.best_model_name
        ].get(settings.training.metric),
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input": {
            "fingerprint": validation_report["metadata"]["fingerprint"],
            "reference": ingestion["dataset_ref"],
            "source_path": ingestion["source_path"],
        },
        "resolved_configuration": settings.model_dump(mode="json"),
        "input_schema": {
            "columns": validation_report["metadata"].get("column_names", []),
            "column_dtypes": validation_report["metadata"].get(
                "column_dtypes", {}
            ),
            "target_column": settings.target_column,
        },
        "feature_schema": {
            "model_input_columns": [
                str(name)
                for name in preprocessing.preprocessor.feature_names_in_
            ],
            "excluded_feature_columns": settings.data.excluded_feature_columns,
            "numerical_columns": preprocessing.feature_metadata.numerical_columns,
            "categorical_columns": preprocessing.feature_metadata.categorical_columns,
            "transformed_feature_names": (
                preprocessing.feature_metadata.transformed_feature_names
            ),
        },
        "validation_metrics": selection.validation_results[
            selection.best_model_name
        ].as_dict(),
        "candidate_validation_metrics": {
            model_name: metrics.as_dict()
            for model_name, metrics in selection.validation_results.items()
        },
        "final_test_metrics": test_metrics_dict,
        "gate_decision": gate_report,
        "validation_warnings": validation_report.get("warnings", []),
        "artifacts": {
            "inference_bundle_ref": bundle_ref,
            "test_evaluation_ref": evaluation_ref,
        },
    }
    # The manifest is the completion marker and is always written after its
    # bundle and evaluation report have been atomically persisted.
    manifest_ref = artifacts.write_json(f"{publication_prefix}/manifest.json", manifest)
    return _handoff(
        artifacts,
        selected_model_name=selection.best_model_name,
        inference_bundle_ref=bundle_ref,
        test_evaluation_ref=evaluation_ref,
        approved_manifest_ref=manifest_ref,
        final_test_metrics=test_metrics_dict,
    )


def run_training_pipeline(
    artifact_root: str | Path,
    config_path: str | Path | None = None,
    data_path: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the complete reusable package workflow outside Airflow as well."""

    ingestion = ingest_dataset(
        artifact_root,
        config_path=config_path,
        data_path=data_path,
        run_id=run_id,
    )
    validation = validate_ingested_dataset(ingestion)
    preprocessing = preprocess_validated_dataset(validation)
    selection = train_and_select_artifact(preprocessing)
    approved = apply_validation_promotion_gate(selection)
    return evaluate_and_publish_approved_model(approved)


def build_inference_pipeline(
    preprocessing: PreprocessingResult,
    estimator: Any,
) -> Pipeline:
    """Combine fitted preprocessing and estimator into one sklearn pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", preprocessing.preprocessor),
            ("estimator", estimator),
        ]
    )


def _open_run(handoff: Mapping[str, Any]) -> RunArtifacts:
    artifact_root = handoff.get("artifact_root")
    run_id = handoff.get("run_id")
    if not isinstance(artifact_root, str) or not isinstance(run_id, str):
        raise ModelArtifactError(
            "Task handoff must include string artifact_root and run_id values"
        )
    return RunArtifacts.open(artifact_root, run_id)


def _reference(handoff: Mapping[str, Any], key: str) -> str:
    reference = handoff.get(key)
    if not isinstance(reference, str):
        raise ModelArtifactError(f"Task handoff is missing artifact reference '{key}'")
    return reference


def _read_settings(artifacts: RunArtifacts, reference: str) -> Settings:
    try:
        return Settings.model_validate(artifacts.read_json(reference))
    except ModelArtifactError:
        raise
    except Exception as exc:
        raise ModelArtifactError("Could not load settings snapshot from run artifacts") from exc


def _handoff(artifacts: RunArtifacts, **metadata: Any) -> dict[str, Any]:
    """Build a small JSON-safe task result with shared-run artifact references."""

    return {
        "artifact_root": str(artifacts.root),
        "run_id": artifacts.run_id,
        **metadata,
    }
