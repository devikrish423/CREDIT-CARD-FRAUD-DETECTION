"""Prepare the credit card dataset for model training and evaluation."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
RANDOM_STATE = 42
TARGET_COLUMN = "Class"
SCALED_COLUMNS = ["Time", "Amount"]


@dataclass
class PreparedData:
    """Training and testing datasets after scaling and training-only SMOTE."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessing_pipeline: Pipeline


def _print_class_distribution(name: str, target: pd.Series) -> None:
    """Print counts and percentages for each target class."""
    counts = target.value_counts().sort_index()
    percentages = target.value_counts(normalize=True).sort_index() * 100
    summary = pd.DataFrame({"Count": counts, "Percentage": percentages})
    print(f"\n{name} class distribution:")
    print(summary.to_string(formatters={"Percentage": "{:.4f}%".format}))


def prepare_data(data_path: Path = DATA_PATH) -> PreparedData:
    """Load, split, scale, and resample the dataset without test-set leakage."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain the target column '{TARGET_COLUMN}'.")

    missing_features = set(SCALED_COLUMNS).difference(data.columns)
    if missing_features:
        raise ValueError(
            "Dataset is missing columns required for scaling: "
            + ", ".join(sorted(missing_features))
        )
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains missing values.")

    X = data.drop(columns=TARGET_COLUMN)
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessing_pipeline = Pipeline(
        steps=[
            (
                "scale_time_amount",
                ColumnTransformer(
                    transformers=[("standard_scaler", StandardScaler(), SCALED_COLUMNS)],
                    remainder="passthrough",
                ),
            )
        ]
    )
    X_train_scaled_array = preprocessing_pipeline.fit_transform(X_train)
    X_test_scaled_array = preprocessing_pipeline.transform(X_test)
    feature_names = preprocessing_pipeline.get_feature_names_out()
    X_train_scaled = pd.DataFrame(
        X_train_scaled_array, columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled_array, columns=feature_names, index=X_test.index
    )

    print(f"Dataset shape: {data.shape}")
    print(f"Training split before SMOTE: {X_train_scaled.shape}")
    print(f"Testing split (not resampled): {X_test_scaled.shape}")
    _print_class_distribution("Training before SMOTE", y_train)
    _print_class_distribution("Testing", y_test)

    X_train_resampled, y_train_resampled = SMOTE(
        random_state=RANDOM_STATE
    ).fit_resample(X_train_scaled, y_train)

    print(f"\nTraining split after SMOTE: {X_train_resampled.shape}")
    _print_class_distribution("Training after SMOTE", y_train_resampled)

    return PreparedData(
        X_train=X_train_resampled,
        X_test=X_test_scaled,
        y_train=y_train_resampled,
        y_test=y_test,
        preprocessing_pipeline=preprocessing_pipeline,
    )


if __name__ == "__main__":
    prepare_data()
