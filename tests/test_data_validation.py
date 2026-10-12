import pandas as pd
import pytest

from automl.data.validation import build_metadata, validate_dataset
from automl.exceptions import DataValidationError


def test_valid_dataset_validation_returns_metadata():
    data_frame = pd.DataFrame(
        {
            "feature_num": [1.0, 2.0, 3.0],
            "feature_cat": ["A", "B", "A"],
            "target": ["yes", "no", "yes"],
        }
    )

    result = validate_dataset(data_frame, target_column="target", source_path="memory.csv")

    assert result.warnings == []
    assert result.metadata.source_path == "memory.csv"
    assert result.metadata.row_count == 3
    assert result.metadata.column_count == 3
    assert result.metadata.column_names == ["feature_num", "feature_cat", "target"]
    assert result.metadata.numerical_columns == ["feature_num"]
    assert result.metadata.categorical_columns == ["feature_cat"]
    assert result.metadata.missing_value_counts == {
        "feature_num": 0,
        "feature_cat": 0,
        "target": 0,
    }
    assert result.metadata.duplicate_count == 0
    assert len(result.metadata.fingerprint) == 64


def test_missing_target_column_raises_validation_error():
    data_frame = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(DataValidationError, match="target column is missing"):
        validate_dataset(data_frame, target_column="target")


def test_target_column_entirely_missing_raises_validation_error():
    data_frame = pd.DataFrame({"feature": [1, 2, 3], "target": [None, None, None]})

    with pytest.raises(DataValidationError, match="target column is completely missing"):
        validate_dataset(data_frame, target_column="target")


def test_duplicate_rows_are_reported_as_warning():
    data_frame = pd.DataFrame(
        {
            "feature": [1, 1, 2],
            "target": ["yes", "yes", "no"],
        }
    )

    result = validate_dataset(data_frame, target_column="target")

    assert result.metadata.duplicate_count == 1
    assert result.warnings == ["Dataset contains 1 duplicate rows"]


def test_missing_feature_values_are_reported_as_warning():
    data_frame = pd.DataFrame(
        {
            "feature": [1.0, None, 3.0],
            "target": ["yes", "no", "yes"],
        }
    )

    result = validate_dataset(data_frame, target_column="target")

    assert result.metadata.missing_value_counts["feature"] == 1
    assert "Feature columns contain missing values" in result.warnings[0]


def test_completely_empty_feature_column_raises_validation_error():
    data_frame = pd.DataFrame(
        {
            "feature": [None, None, None],
            "target": ["yes", "no", "yes"],
        }
    )

    with pytest.raises(DataValidationError, match="only missing values"):
        validate_dataset(data_frame, target_column="target")


def test_infinite_numeric_values_raise_validation_error():
    data_frame = pd.DataFrame(
        {
            "feature": [1.0, float("inf"), 3.0],
            "target": ["yes", "no", "yes"],
        }
    )

    with pytest.raises(DataValidationError, match="infinite values"):
        validate_dataset(data_frame, target_column="target")


def test_metadata_generation_is_deterministic():
    data_frame = pd.DataFrame({"feature": [1, 2], "target": ["yes", "no"]})

    first_metadata = build_metadata(data_frame, target_column="target")
    second_metadata = build_metadata(data_frame, target_column="target")

    assert first_metadata == second_metadata
    assert first_metadata.row_count == 2
    assert first_metadata.column_count == 2
