"""Model training and selection for the AutoML pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from automl.config.settings import Settings
from automl.evaluation.metrics import (
    ClassificationMetrics,
    evaluate_classifier,
    select_best_model,
)
from automl.exceptions import DataValidationError, TrainingError
from automl.logging_config import get_logger
from automl.preprocessing import PreprocessingResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelTrainingResult:
    """Result produced by training a single model."""

    model_name: str
    model: ClassifierMixin
    validation_metrics: ClassificationMetrics


@dataclass(frozen=True)
class TrainingResult:
    """Complete result of the model training stage."""

    trained_models: dict[str, ClassifierMixin]
    validation_results: dict[str, ClassificationMetrics]
    best_model_name: str
    best_model: ClassifierMixin
    test_metrics: ClassificationMetrics
    failed_models: dict[str, str]


def build_classification_models(
    random_seed: int,
) -> dict[str, ClassifierMixin]:
    """
    Build fresh candidate classification models.

    The returned models are intentionally untrained.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_seed,
            n_jobs=-1,
        ),
    }


def train_single_model(
    model_name: str,
    model: ClassifierMixin,
    X_train: np.ndarray,
    y_train: Any,
) -> ClassifierMixin:
    """Fit one classification model using only training data."""
    if len(X_train) != len(y_train):
        raise DataValidationError(
            f"Training features and targets have different lengths "
            f"for model '{model_name}'"
        )

    if len(y_train) == 0:
        raise DataValidationError(
            f"Cannot train model '{model_name}' on an empty dataset"
        )

    logger.info(
        "Training model '%s' on %d rows and %d features",
        model_name,
        X_train.shape[0],
        X_train.shape[1],
    )

    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        raise TrainingError(
            f"Model '{model_name}' failed during training: {exc}"
        ) from exc

    logger.info(
        "Model '%s' training completed",
        model_name,
    )

    return model


def train_models(
    preprocessed_data: PreprocessingResult,
    settings: Settings,
) -> TrainingResult:
    """
    Train candidate models, select the best using validation data,
    and evaluate the selected model once on the held-out test data.

    Model selection never uses the test set.
    """
    _validate_training_data(preprocessed_data)

    candidate_models = build_classification_models(
        random_seed=settings.random_seed,
    )

    trained_models: dict[str, ClassifierMixin] = {}
    validation_results: dict[str, ClassificationMetrics] = {}
    failed_models: dict[str, str] = {}

    for model_name, model in candidate_models.items():
        try:
            trained_model = train_single_model(
                model_name=model_name,
                model=model,
                X_train=preprocessed_data.X_train,
                y_train=preprocessed_data.y_train.to_numpy(),
            )

            validation_metrics = evaluate_classifier(
                trained_model,
                preprocessed_data.X_validation,
                preprocessed_data.y_validation.to_numpy(),
            )

            trained_models[model_name] = trained_model
            validation_results[model_name] = validation_metrics

            logger.info(
                "Model '%s' validation accuracy=%.4f",
                model_name,
                validation_metrics.accuracy,
            )

        except TrainingError as exc:
            failed_models[model_name] = str(exc)

            logger.exception(
                "Model '%s' failed during training",
                model_name,
            )

        except Exception as exc:
            failed_models[model_name] = str(exc)

            logger.exception(
                "Model '%s' failed during validation",
                model_name,
            )

    if not validation_results:
        details = "; ".join(
            f"{name}: {error}"
            for name, error in failed_models.items()
        )

        raise TrainingError(
            "All candidate models failed during training/evaluation"
            + (f": {details}" if details else "")
        )

    best_model_name = select_best_model(
        validation_results=validation_results,
        metric=settings.training.metric,
    )

    best_model = trained_models[best_model_name]

    logger.info(
        "Selected model '%s'. Running final test evaluation.",
        best_model_name,
    )

    try:
        test_metrics = evaluate_classifier(
            best_model,
            preprocessed_data.X_test,
            preprocessed_data.y_test.to_numpy(),
        )
    except Exception as exc:
        raise TrainingError(
            f"Final test evaluation failed for model "
            f"'{best_model_name}': {exc}"
        ) from exc

    return TrainingResult(
        trained_models=trained_models,
        validation_results=validation_results,
        best_model_name=best_model_name,
        best_model=best_model,
        test_metrics=test_metrics,
        failed_models=failed_models,
    )


def _validate_training_data(
    preprocessed_data: PreprocessingResult,
) -> None:
    """Validate that all required model-training splits are usable."""
    if preprocessed_data.X_train.shape[0] == 0:
        raise DataValidationError("Training dataset is empty")

    if preprocessed_data.X_validation.shape[0] == 0:
        raise DataValidationError("Validation dataset is empty")

    if preprocessed_data.X_test.shape[0] == 0:
        raise DataValidationError("Test dataset is empty")

    if len(preprocessed_data.y_train) != preprocessed_data.X_train.shape[0]:
        raise DataValidationError(
            "Training features and targets have different lengths"
        )

    if (
        len(preprocessed_data.y_validation)
        != preprocessed_data.X_validation.shape[0]
    ):
        raise DataValidationError(
            "Validation features and targets have different lengths"
        )

    if len(preprocessed_data.y_test) != preprocessed_data.X_test.shape[0]:
        raise DataValidationError(
            "Test features and targets have different lengths"
        )

    if preprocessed_data.X_train.shape[1] == 0:
        raise DataValidationError(
            "Training dataset contains zero transformed features"
        )