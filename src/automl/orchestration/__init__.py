"""Reusable task adapters for orchestrating the AutoML workflow."""

from automl.orchestration.training import (
    apply_validation_promotion_gate,
    build_inference_pipeline,
    evaluate_and_publish_approved_model,
    ingest_dataset,
    preprocess_validated_dataset,
    run_training_pipeline,
    train_and_select_artifact,
    validate_ingested_dataset,
)

__all__ = [
    "apply_validation_promotion_gate",
    "build_inference_pipeline",
    "evaluate_and_publish_approved_model",
    "ingest_dataset",
    "validate_ingested_dataset",
    "preprocess_validated_dataset",
    "train_and_select_artifact",
    "run_training_pipeline",
]
