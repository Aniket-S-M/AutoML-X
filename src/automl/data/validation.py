"""Dataset validation and metadata extraction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from automl.exceptions import DataValidationError
from automl.logging_config import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class DataMetadata:
    """Lightweight schema and quality metadata for a loaded dataset."""

    source_path: str
    row_count: int
    column_count: int
    column_names: list[str]
    target_column: str
    numerical_columns: list[str]
    categorical_columns: list[str]
    missing_value_counts: dict[str, int]
    duplicate_count: int
    fingerprint: str


@dataclass(frozen=True)
class DataValidationResult:
    """Validation result for downstream pipeline stages."""

    metadata: DataMetadata
    warnings: list[str]


def validate_dataset(
    data_frame: pd.DataFrame,
    target_column: str,
    source_path: str | Path = "",
) -> DataValidationResult:
    """Validate a loaded dataset and return metadata without changing the data."""

    _validate_shape(data_frame)
    _validate_target(data_frame, target_column)
    _validate_empty_columns(data_frame, target_column)
    _validate_infinite_values(data_frame)

    metadata = build_metadata(data_frame, target_column, source_path)
    warnings = _build_warnings(data_frame, target_column, metadata)

    for warning in warnings:
        logger.warning(warning)

    logger.info(
        "Validated dataset from %s with %s rows, %s columns, and %s duplicate rows",
        metadata.source_path,
        metadata.row_count,
        metadata.column_count,
        metadata.duplicate_count,
    )
    return DataValidationResult(metadata=metadata, warnings=warnings)


def build_metadata(
    data_frame: pd.DataFrame,
    target_column: str,
    source_path: str | Path = "",
) -> DataMetadata:
    """Build deterministic metadata for a validated dataset."""

    feature_columns = [column for column in data_frame.columns if column != target_column]
    numerical_columns = [
        column for column in feature_columns if is_numeric_dtype(data_frame[column])
    ]
    categorical_columns = [
        column for column in feature_columns if not is_numeric_dtype(data_frame[column])
    ]
    missing_counts = {
        str(column): int(count) for column, count in data_frame.isna().sum().items()
    }

    return DataMetadata(
        source_path=str(source_path),
        row_count=int(len(data_frame)),
        column_count=int(len(data_frame.columns)),
        column_names=[str(column) for column in data_frame.columns],
        target_column=target_column,
        numerical_columns=[str(column) for column in numerical_columns],
        categorical_columns=[str(column) for column in categorical_columns],
        missing_value_counts=missing_counts,
        duplicate_count=int(data_frame.duplicated().sum()),
        fingerprint=_fingerprint_dataframe(data_frame),
    )


def _validate_shape(data_frame: pd.DataFrame) -> None:
    if data_frame.empty:
        raise DataValidationError("Dataset contains no rows")
    if len(data_frame.columns) == 0:
        raise DataValidationError("Dataset contains no columns")


def _validate_target(data_frame: pd.DataFrame, target_column: str) -> None:
    if target_column not in data_frame.columns:
        raise DataValidationError(f"Configured target column is missing: {target_column}")
    if data_frame[target_column].isna().all():
        raise DataValidationError(
            f"Configured target column is completely missing: {target_column}"
        )


def _validate_empty_columns(data_frame: pd.DataFrame, target_column: str) -> None:
    empty_feature_columns = [
        str(column)
        for column in data_frame.columns
        if column != target_column and data_frame[column].isna().all()
    ]
    if empty_feature_columns:
        raise DataValidationError(
            "Feature columns contain only missing values: "
            + ", ".join(empty_feature_columns)
        )


def _validate_infinite_values(data_frame: pd.DataFrame) -> None:
    numeric_columns = [
        column for column in data_frame.columns if is_numeric_dtype(data_frame[column])
    ]
    invalid_columns = [
        str(column)
        for column in numeric_columns
        if np.isinf(data_frame[column].to_numpy(dtype=float)).any()
    ]
    if invalid_columns:
        raise DataValidationError(
            "Numeric columns contain invalid infinite values: "
            + ", ".join(invalid_columns)
        )


def _build_warnings(
    data_frame: pd.DataFrame,
    target_column: str,
    metadata: DataMetadata,
) -> list[str]:
    warnings: list[str] = []

    if metadata.duplicate_count > 0:
        warnings.append(f"Dataset contains {metadata.duplicate_count} duplicate rows")

    feature_missing_counts = {
        column: count
        for column, count in metadata.missing_value_counts.items()
        if column != target_column and count > 0
    }
    if feature_missing_counts:
        warnings.append(f"Feature columns contain missing values: {feature_missing_counts}")

    target_distribution = data_frame[target_column].value_counts(dropna=True, normalize=True)
    if len(target_distribution) > 1 and target_distribution.iloc[0] >= 0.9:
        warnings.append(
            "Target distribution is highly imbalanced: "
            f"{target_distribution.index[0]}={target_distribution.iloc[0]:.2%}"
        )

    return warnings


def _fingerprint_dataframe(data_frame: pd.DataFrame) -> str:
    hashed_rows = pd.util.hash_pandas_object(data_frame, index=True).to_numpy()
    return sha256(hashed_rows.tobytes()).hexdigest()
