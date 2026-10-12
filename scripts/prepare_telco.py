from pathlib import Path

import pandas as pd


SOURCE = Path("data/raw/dataset_1.csv")
OUTPUT = Path("data/raw/dataset.csv")


def main() -> None:
    df = pd.read_csv(SOURCE)

    print("Original shape:", df.shape)

    # Remove identifier column.
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric.
    # Invalid/blank values become NaN and will be handled
    # by the preprocessing pipeline's numerical imputer.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    # Rename target to match the AutoML pipeline contract.
    df = df.rename(columns={"Churn": "target"})

    df.to_csv(OUTPUT, index=False)

    print("Prepared shape:", df.shape)
    print("Saved to:", OUTPUT)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nMissing values:")
    print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()