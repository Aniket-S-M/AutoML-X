"""Preprocessing pipeline for validated tabular data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from automl.config.settings import Settings
from automl.exceptions import DataValidationError
from automl.logging_config import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetSplit:
    """Raw feature/target splits created before fitting preprocessing."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class FeatureMetadata:
    """Feature columns before and after preprocessing."""

    numerical_columns: list[str]
    categorical_columns: list[str]
    transformed_feature_names: list[str]


@dataclass(frozen=True)
class PreprocessingResult:
    """Model-ready transformed splits and the fitted reusable preprocessor."""

    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_metadata: FeatureMetadata


def split_features_target(
    data_frame: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from target without mutating the original DataFrame."""

    if target_column not in data_frame.columns:
        raise DataValidationError(f"Configured target column is missing: {target_column}")

    X = data_frame.drop(columns=[target_column]).copy()
    y = data_frame[target_column].copy()
    return X, y


def split_dataset(data_frame: pd.DataFrame, settings: Settings) -> DatasetSplit:
    """Split validated data into train, validation, and test sets."""

    X, y = split_features_target(data_frame, settings.target_column)
    test_size = settings.test_size
    validation_size = settings.validation_size

    if test_size + validation_size >= 1.0:
        raise DataValidationError("Configured test_size + validation_size must be less than 1.0")

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=settings.random_seed,
        stratify=_stratify_if_supported(y, test_size),
    )

    relative_validation_size = validation_size / (1.0 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=relative_validation_size,
        random_state=settings.random_seed,
        stratify=_stratify_if_supported(y_train_validation, relative_validation_size),
    )

    logger.info(
        "Split dataset into train=%s, validation=%s, test=%s rows",
        len(X_train),
        len(X_validation),
        len(X_test),
    )
    return DatasetSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def build_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build an unfitted sklearn preprocessing object."""

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocess_dataset(data_frame: pd.DataFrame, settings: Settings) -> PreprocessingResult:
    """Split data and fit preprocessing only on the training features."""

    split = split_dataset(data_frame, settings)
    numerical_columns, categorical_columns = infer_feature_columns(split.X_train)
    preprocessor = build_preprocessor(numerical_columns, categorical_columns)

    X_train = preprocessor.fit_transform(split.X_train)
    X_validation = preprocessor.transform(split.X_validation)
    X_test = preprocessor.transform(split.X_test)
    feature_names = _get_feature_names(preprocessor)

    logger.info(
        "Fitted preprocessing on %s training rows with %s transformed features",
        len(split.X_train),
        len(feature_names),
    )
    return PreprocessingResult(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=split.y_train,
        y_validation=split.y_validation,
        y_test=split.y_test,
        preprocessor=preprocessor,
        feature_metadata=FeatureMetadata(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            transformed_feature_names=feature_names,
        ),
    )


def transform_features(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Transform new features with an already fitted preprocessor."""

    return preprocessor.transform(X)


def infer_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numerical and categorical feature columns from a feature DataFrame."""

    numerical_columns = [str(column) for column in X.columns if is_numeric_dtype(X[column])]
    categorical_columns = [
        str(column) for column in X.columns if not is_numeric_dtype(X[column])
    ]
    return numerical_columns, categorical_columns


def _stratify_if_supported(
    y: pd.Series,
    test_size: float,
) -> pd.Series | None:
    """Return target labels when stratified splitting is feasible."""

    if not _looks_like_classification_target(y):
        return None

    class_counts = y.value_counts(dropna=False)

    if class_counts.empty or class_counts.min() < 2:
        return None

    test_count = int(np.ceil(len(y) * test_size))
    train_count = len(y) - test_count
    number_of_classes = len(class_counts)

    if test_count < number_of_classes:
        return None

    if train_count < number_of_classes:
        return None

    return y


def _looks_like_classification_target(y: pd.Series) -> bool:
    """Determine whether a target is suitable for classification."""

    if pd.api.types.is_object_dtype(y):
        return True

    if pd.api.types.is_string_dtype(y):
        return True

    if isinstance(y.dtype, pd.CategoricalDtype):
        return True

    if pd.api.types.is_bool_dtype(y):
        return True

    if pd.api.types.is_integer_dtype(y):
        return y.nunique(dropna=False) <= min(20, max(2, len(y) // 5))

    return False

def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except ValueError:
        return []
