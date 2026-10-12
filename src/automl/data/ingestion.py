"""CSV ingestion utilities for tabular datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from automl.config.settings import Settings
from automl.exceptions import DataValidationError
from automl.logging_config import get_logger


logger = get_logger(__name__)


def load_csv_data(settings: Settings, data_path: str | Path | None = None) -> pd.DataFrame:
    """Load a configured CSV file into a pandas DataFrame without modifying it."""

    path = Path(data_path or settings.data.input_path)
    _validate_csv_path(path)

    try:
        data_frame = pd.read_csv(path)
    except EmptyDataError as exc:
        raise DataValidationError(f"CSV file is empty or has no columns: {path}") from exc
    except ParserError as exc:
        raise DataValidationError(f"CSV file could not be parsed: {path}") from exc
    except OSError as exc:
        raise DataValidationError(f"CSV file could not be read: {path}") from exc

    if data_frame.empty:
        raise DataValidationError(f"CSV file contains no data rows: {path}")
    if len(data_frame.columns) == 0:
        raise DataValidationError(f"CSV file contains no columns: {path}")

    logger.info(
        "Loaded CSV data from %s with %s rows and %s columns",
        path,
        len(data_frame),
        len(data_frame.columns),
    )
    return data_frame


def _validate_csv_path(path: Path) -> None:
    if not path.exists():
        raise DataValidationError(f"Configured data file does not exist: {path}")
    if not path.is_file():
        raise DataValidationError(f"Configured data path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise DataValidationError(f"Unsupported data file type for {path}; expected a .csv file")

    try:
        with path.open("r", encoding="utf-8"):
            pass
    except OSError as exc:
        raise DataValidationError(f"Configured data file is not readable: {path}") from exc
