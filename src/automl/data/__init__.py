"""Data ingestion and validation contracts."""

from automl.data.ingestion import load_csv_data
from automl.data.validation import DataMetadata, DataValidationResult, validate_dataset

__all__ = [
    "DataMetadata",
    "DataValidationResult",
    "load_csv_data",
    "validate_dataset",
]
