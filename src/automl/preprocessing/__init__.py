"""Preprocessing contracts for model-ready datasets."""

from automl.preprocessing.pipeline import (
    DatasetSplit,
    FeatureMetadata,
    PreprocessingResult,
    build_preprocessor,
    preprocess_dataset,
    split_dataset,
    split_features_target,
    transform_features,
)

__all__ = [
    "DatasetSplit",
    "FeatureMetadata",
    "PreprocessingResult",
    "build_preprocessor",
    "preprocess_dataset",
    "split_dataset",
    "split_features_target",
    "transform_features",
]
