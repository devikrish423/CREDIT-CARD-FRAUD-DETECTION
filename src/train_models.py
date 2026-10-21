"""Train, evaluate, and save credit card fraud classifiers."""

from pathlib import Path
import re
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from src.preprocessing import PROJECT_ROOT, prepare_data
except ModuleNotFoundError:
    from preprocessing import PROJECT_ROOT, prepare_data


MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"
RANDOM_STATE = 42


def _model_slug(name: str) -> str:
    """Convert a model name into a filesystem-friendly filename."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _get_models() -> dict[str, object]:
    """Create the five requested classifiers with reproducible settings."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM (LinearSVC)": LinearSVC(
            random_state=RANDOM_STATE,
            max_iter=5000,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _positive_class_scores(model: object, features: pd.DataFrame) -> pd.Series:
    """Return a fraud score for ROC-AUC and PR-AUC calculations."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = list(model.classes_)
        return pd.Series(probabilities[:, classes.index(1)], index=features.index)
    if hasattr(model, "decision_function"):
        return pd.Series(model.decision_function(features), index=features.index)
    raise TypeError(f"{type(model).__name__} does not provide prediction scores.")


def _save_confusion_matrix(
    model_name: str,
    matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a confusion matrix as a labeled heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Greys")
    ax.set(
        xticks=[0, 1],
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticks=[0, 1],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"{model_name} Confusion Matrix",
    )
    fig.colorbar(image, ax=ax, label="Transactions")
    threshold = matrix.to_numpy().max() / 2
    for row in range(2):
        for column in range(2):
            value = matrix.iat[row, column]
            ax.text(
                column,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_and_evaluate() -> pd.DataFrame:
    """Train all classifiers, evaluate on the held-out test set, and save outputs."""
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data()
    if set(prepared.y_test.unique()) != {0, 1}:
        raise ValueError("The test data must contain both target classes, 0 and 1.")

    prepared.preprocessing_pipeline.set_output(transform="pandas")

    result_rows = []
    fitted_pipelines = {}
    confusion_matrices = {}
    models = _get_models()

    for model_name, estimator in models.items():
        print(f"\nTraining {model_name}...")
        fitted_pipeline = Pipeline(
            steps=[
                ("preprocessing", prepared.preprocessing_pipeline),
                ("classifier", estimator),
            ]
        )

        classifier = fitted_pipeline.named_steps["classifier"]
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            classifier.fit(prepared.X_train, prepared.y_train)

        predictions = classifier.predict(prepared.X_test)
        scores = _positive_class_scores(classifier, prepared.X_test)
        precision_curve, recall_curve, _ = precision_recall_curve(
            prepared.y_test, scores
        )
        matrix = confusion_matrix(
            prepared.y_test, predictions, labels=[0, 1]
        )
        matrix_frame = pd.DataFrame(
            matrix,
            index=["Actual legitimate (0)", "Actual fraud (1)"],
            columns=["Predicted legitimate (0)", "Predicted fraud (1)"],
        )
        confusion_matrices[model_name] = matrix_frame

        result_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(prepared.y_test, predictions),
                "precision": precision_score(
                    prepared.y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    prepared.y_test, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    prepared.y_test, predictions, zero_division=0
                ),
                "roc_auc": roc_auc_score(prepared.y_test, scores),
                "pr_auc": auc(recall_curve, precision_curve),
                "true_negatives": int(matrix[0, 0]),
                "false_positives": int(matrix[0, 1]),
                "false_negatives": int(matrix[1, 0]),
                "true_positives": int(matrix[1, 1]),
            }
        )

        artifact_path = MODELS_PATH / f"{_model_slug(model_name)}.joblib"
        joblib.dump(fitted_pipeline, artifact_path)
        fitted_pipelines[model_name] = fitted_pipeline
        print(f"Saved model pipeline: {artifact_path}")

    results = pd.DataFrame(result_rows).sort_values(
        by=["f1_score", "recall", "pr_auc"],
        ascending=False,
        ignore_index=True,
    )
    results_path = REPORTS_PATH / "model_results.csv"
    results.to_csv(results_path, index=False)

    for model_name, matrix in confusion_matrices.items():
        matrix_path = FIGURES_PATH / f"confusion_matrix_{_model_slug(model_name)}.png"
        _save_confusion_matrix(model_name, matrix, matrix_path)
        print(f"Saved confusion matrix: {matrix_path}")

    selected_model_name = results.iloc[0]["model"]
    selected_model_path = MODELS_PATH / "selected_model.joblib"
    joblib.dump(fitted_pipelines[selected_model_name], selected_model_path)

    print("\nTest-set results (sorted by F1-score, recall, then PR-AUC):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResults saved to: {results_path}")
    print(
        f"Selected model: {selected_model_name} "
        f"(highest F1-score; ties broken by recall and PR-AUC)"
    )
    print(f"Selected model pipeline saved to: {selected_model_path}")
    return results


if __name__ == "__main__":
    train_and_evaluate()
