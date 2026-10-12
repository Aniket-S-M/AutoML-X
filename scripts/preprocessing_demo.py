"""Manual preprocessing demonstration without model training."""

from __future__ import annotations

import pandas as pd

from automl.config.settings import Settings
from automl.preprocessing import preprocess_dataset, transform_features


def main() -> None:
    data_frame = pd.DataFrame(
        {
            "age": [24, 31, 45, 52, 29, 38, 41, 33, 27, 48],
            "city": [
                "Bangalore",
                "Delhi",
                "Bangalore",
                "Delhi",
                "Bangalore",
                "Delhi",
                "Bangalore",
                "Delhi",
                "Bangalore",
                "Delhi",
            ],
            "income": [
                40000,
                52000,
                80000,
                91000,
                47000,
                68000,
                73000,
                61000,
                45000,
                88000,
            ],
            "target": ["low", "low", "high", "high", "low", "high", "high", "low", "low", "high"],
        }
    )
    settings = Settings(
        data={"target_column": "target", "test_size": 0.2, "validation_size": 0.2}
    )

    result = preprocess_dataset(data_frame, settings)
    inference_row = pd.DataFrame({"age": [36], "city": ["Mumbai"], "income": [64000]})
    transformed_inference = transform_features(result.preprocessor, inference_row)

    print("Input columns:", data_frame.columns.tolist())
    print("Transformed feature names:", result.feature_metadata.transformed_feature_names)
    print("X_train shape:", result.X_train.shape)
    print("y_train name:", result.y_train.name)
    print("Unseen-category inference shape:", transformed_inference.shape)


if __name__ == "__main__":
    main()
