import joblib
import numpy as np
import pandas as pd

from automl.config.settings import Settings
from automl.preprocessing import (
    build_preprocessor,
    preprocess_dataset,
    split_dataset,
    split_features_target,
    transform_features,
)
from automl.preprocessing.pipeline import infer_feature_columns


def _settings() -> Settings:
    return Settings(
        project={"random_seed": 7},
        data={"target_column": "target", "test_size": 0.2, "validation_size": 0.2},
    )


def _classification_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [21, 22, 23, 24, 25, 41, 42, 43, 44, 45],
            "city": [
                "Bangalore",
                "Delhi",
                "Bangalore",
                "Delhi",
                "Bangalore",
                "Bangalore",
                "Delhi",
                "Bangalore",
                "Delhi",
                "Bangalore",
            ],
            "income": [
                30000,
                32000,
                34000,
                36000,
                38000,
                70000,
                72000,
                74000,
                76000,
                78000,
            ],
            "target": ["no"] * 5 + ["yes"] * 5,
        }
    )


def test_feature_target_separation_does_not_modify_original_data_frame():
    data_frame = _classification_frame()
    original = data_frame.copy(deep=True)

    X, y = split_features_target(data_frame, "target")

    assert "target" not in X.columns
    assert y.name == "target"
    pd.testing.assert_frame_equal(data_frame, original)


def test_numerical_and_categorical_features_are_inferred():
    X, _ = split_features_target(_classification_frame(), "target")

    numerical_columns, categorical_columns = infer_feature_columns(X)

    assert numerical_columns == ["age", "income"]
    assert categorical_columns == ["city"]


def test_numerical_feature_preprocessing_scales_training_columns():
    result = preprocess_dataset(_classification_frame(), _settings())
    numerical_count = len(result.feature_metadata.numerical_columns)

    assert np.allclose(result.X_train[:, :numerical_count].mean(axis=0), 0.0)
    assert np.allclose(result.X_train[:, :numerical_count].std(axis=0), 1.0)


def test_categorical_feature_preprocessing_encodes_categories():
    result = preprocess_dataset(_classification_frame(), _settings())

    assert any(
        name.startswith("city_") for name in result.feature_metadata.transformed_feature_names
    )
    assert result.X_train.shape[1] == len(result.feature_metadata.transformed_feature_names)


def test_missing_numerical_values_are_imputed():
    data_frame = _classification_frame()
    data_frame.loc[0, "income"] = np.nan

    result = preprocess_dataset(data_frame, _settings())

    assert not np.isnan(result.X_train).any()
    assert not np.isnan(result.X_validation).any()
    assert not np.isnan(result.X_test).any()


def test_missing_categorical_values_are_imputed_and_encoded():
    data_frame = _classification_frame()
    data_frame.loc[0, "city"] = None

    result = preprocess_dataset(data_frame, _settings())

    assert result.X_train.shape[1] == len(result.feature_metadata.transformed_feature_names)


def test_unseen_categorical_values_do_not_crash_transform():
    result = preprocess_dataset(_classification_frame(), _settings())
    inference_features = pd.DataFrame(
        {
            "age": [33],
            "city": ["Mumbai"],
            "income": [51000],
        }
    )

    transformed = transform_features(result.preprocessor, inference_features)

    assert transformed.shape == (1, len(result.feature_metadata.transformed_feature_names))


def test_train_validation_test_split_sizes_are_configured():
    split = split_dataset(_classification_frame(), _settings())

    assert len(split.X_train) == 6
    assert len(split.X_validation) == 2
    assert len(split.X_test) == 2
    assert len(split.y_train) == 6
    assert len(split.y_validation) == 2
    assert len(split.y_test) == 2


def test_split_is_deterministic_with_same_random_seed():
    first = split_dataset(_classification_frame(), _settings())
    second = split_dataset(_classification_frame(), _settings())

    assert first.X_train.index.tolist() == second.X_train.index.tolist()
    assert first.X_validation.index.tolist() == second.X_validation.index.tolist()
    assert first.X_test.index.tolist() == second.X_test.index.tolist()


def test_classification_split_uses_stratification_where_supported():
    split = split_dataset(_classification_frame(), _settings())

    assert set(split.y_train.unique()) == {"no", "yes"}
    assert set(split.y_validation.unique()) == {"no", "yes"}
    assert set(split.y_test.unique()) == {"no", "yes"}


def test_preprocessing_is_fitted_on_training_data_only():
    data_frame = _classification_frame()
    initial_split = split_dataset(data_frame, _settings())
    holdout_indices = initial_split.X_validation.index.union(initial_split.X_test.index)
    data_frame.loc[holdout_indices, "income"] = 1_000_000
    split = split_dataset(data_frame, _settings())
    numerical_columns, categorical_columns = infer_feature_columns(split.X_train)
    preprocessor = build_preprocessor(numerical_columns, categorical_columns)

    preprocessor.fit(split.X_train)

    fitted_median = preprocessor.named_transformers_["numerical"].named_steps[
        "imputer"
    ].statistics_[numerical_columns.index("income")]
    assert fitted_median == split.X_train["income"].median()
    assert fitted_median != data_frame["income"].median()


def test_transformed_output_has_expected_dimensions():
    result = preprocess_dataset(_classification_frame(), _settings())
    expected_columns = len(result.feature_metadata.transformed_feature_names)

    assert result.X_train.shape == (len(result.y_train), expected_columns)
    assert result.X_validation.shape == (len(result.y_validation), expected_columns)
    assert result.X_test.shape == (len(result.y_test), expected_columns)


def test_preprocessing_object_can_be_serialized_with_joblib(tmp_path):
    result = preprocess_dataset(_classification_frame(), _settings())
    output_path = tmp_path / "preprocessor.joblib"

    joblib.dump(result.preprocessor, output_path)

    assert output_path.exists()


def test_serialized_preprocessor_can_be_loaded_and_used(tmp_path):
    result = preprocess_dataset(_classification_frame(), _settings())
    output_path = tmp_path / "preprocessor.joblib"
    joblib.dump(result.preprocessor, output_path)
    loaded_preprocessor = joblib.load(output_path)

    transformed = loaded_preprocessor.transform(
        pd.DataFrame({"age": [33], "city": ["Mumbai"], "income": [51000]})
    )

    assert transformed.shape == (1, len(result.feature_metadata.transformed_feature_names))
