"""Model training package."""

from automl.models.trainer import (
    ModelTrainingResult,
    TrainingResult,
    build_classification_models,
    train_models,
    train_single_model,
)

__all__ = [
    "ModelTrainingResult",
    "TrainingResult",
    "build_classification_models",
    "train_models",
    "train_single_model",
]