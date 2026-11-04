from pathlib import Path

import pandas as pd
import pytest

from automl.config.settings import Settings
from automl.data.ingestion import load_csv_data
from automl.exceptions import DataValidationError


def test_valid_csv_loads_successfully():
    settings = Settings(data={"input_path": Path("tests/fixtures/data/valid_dataset.csv")})

    data_frame = load_csv_data(settings)

    assert isinstance(data_frame, pd.DataFrame)
    assert data_frame.shape == (3, 3)
    assert list(data_frame.columns) == ["feature_num", "feature_cat", "target"]


def test_single_space_numeric_cells_are_loaded_as_missing(tmp_path):
    csv_path = tmp_path / "space_missing.csv"
    csv_path.write_text(
        "tenure,TotalCharges,Churn\n0, ,No\n1,29.85,Yes\n",
        encoding="utf-8",
    )
    settings = Settings(data={"input_path": csv_path})

    data_frame = load_csv_data(settings)

    assert pd.api.types.is_numeric_dtype(data_frame["TotalCharges"])
    assert pd.isna(data_frame.loc[0, "TotalCharges"])
    assert data_frame.loc[1, "TotalCharges"] == 29.85


def test_missing_file_raises_validation_error(tmp_path):
    settings = Settings(data={"input_path": tmp_path / "missing.csv"})

    with pytest.raises(DataValidationError, match="does not exist"):
        load_csv_data(settings)


def test_unsupported_input_raises_validation_error(tmp_path):
    unsupported_file = tmp_path / "dataset.json"
    unsupported_file.write_text("{}", encoding="utf-8")
    settings = Settings(data={"input_path": unsupported_file})

    with pytest.raises(DataValidationError, match="Unsupported data file type"):
        load_csv_data(settings)


def test_empty_csv_raises_validation_error(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    settings = Settings(data={"input_path": empty_file})

    with pytest.raises(DataValidationError, match="empty|no columns"):
        load_csv_data(settings)


def test_header_only_csv_raises_validation_error(tmp_path):
    header_only_file = tmp_path / "header_only.csv"
    header_only_file.write_text("feature,target\n", encoding="utf-8")
    settings = Settings(data={"input_path": header_only_file})

    with pytest.raises(DataValidationError, match="no data rows"):
        load_csv_data(settings)
