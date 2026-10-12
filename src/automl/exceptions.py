"""Domain exceptions shared across the AutoML package."""


class AutoMLError(Exception):
    """Base exception for AutoML pipeline errors."""


class ConfigurationError(AutoMLError):
    """Raised when configuration is missing, invalid, or inconsistent."""


class DataValidationError(AutoMLError):
    """Raised when input data does not satisfy the expected contract."""


class ModelArtifactError(AutoMLError):
    """Raised when a model artifact cannot be read, written, or validated."""


class TrainingError(AutoMLError):
    """Raised when model training cannot complete successfully."""
