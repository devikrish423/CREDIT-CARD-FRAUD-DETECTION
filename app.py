"""Streamlit dashboard for credit card fraud estimates."""

from pathlib import Path
import re
import sqlite3

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from database import (
    add_prediction,
    clear_prediction_history,
    get_prediction_history,
    initialize_database,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_results.csv"
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"
RANDOM_STATE = 42
METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --background-color: #ffffff;
        --secondary-background-color: #f2f2f2;
        --text-color: #292929;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #ffffff;
        color: #292929;
    }
    [data-testid="stMain"] :is(p, span, label, li, small, summary) {
        color: #3f3f3f;
    }
    [data-testid="stSidebar"] {
        background-color: #f2f2f2;
        color: #262626;
    }
    [data-testid="stSidebar"] :is(p, span, label, li, small, strong, em, a) {
        color: #262626 !important;
    }
    [data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    .block-container { max-width: 1180px; padding-top: 1.7rem; padding-bottom: 2rem; }
    [data-testid="stMain"] :is(h1, h2, h3, h4, h5, h6) {
        color: #111111;
        letter-spacing: -0.025em;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #d8d8d8;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        background-color: #ffffff;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricDelta"] {
        color: #505050 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #111111 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] [role="radio"] {
        color: #262626 !important;
        border-radius: 5px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] [role="radio"][aria-checked="true"] {
        background-color: #dedede;
        color: #111111 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] [role="radio"] * {
        color: #262626 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] [role="radio"][aria-checked="true"] * {
        color: #111111 !important;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        background-color: #171717;
        color: #ffffff !important;
        border: 1px solid #171717;
        border-radius: 6px;
        min-height: 2.4rem;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background-color: #333333;
        color: #ffffff !important;
        border-color: #333333;
    }
    div.stButton > button:focus-visible,
    div.stFormSubmitButton > button:focus-visible {
        outline: 2px solid #444444;
        outline-offset: 2px;
    }
    hr { border-color: #d8d8d8; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the existing trained Random Forest pipeline."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained Random Forest model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """Load model evaluation metrics from the training report."""
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(f"Model metrics file not found: {METRICS_PATH}")
    metrics = pd.read_csv(METRICS_PATH)
    required_columns = {"model", *METRIC_COLUMNS}
    missing_columns = required_columns.difference(metrics.columns)
    if missing_columns:
        raise ValueError(
            "Model metrics file is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    return metrics


def show_image(filename: str, caption: str) -> bool:
    """Display an existing project chart, reporting missing files clearly."""
    image_path = FIGURES_PATH / filename
    if not image_path.is_file():
        st.warning(f"Chart is not available: {image_path}")
        return False
    st.image(str(image_path), caption=caption, width="stretch")
    return True


def format_metric(value: float) -> str:
    """Format a stored metric value as a percentage."""
    return f"{value:.2%}"


@st.cache_data
def load_sample_test_transaction(
    feature_names: tuple[str, ...],
) -> tuple[dict[str, float], int, int]:
    """Load a real fraud example from the same held-out split used in training."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if "Class" not in data.columns:
        raise ValueError("Dataset is missing the required 'Class' target column.")
    missing_features = set(feature_names).difference(data.columns)
    if missing_features:
        raise ValueError(
            "Dataset is missing model features: " + ", ".join(sorted(missing_features))
        )

    X = data.loc[:, list(feature_names)]
    y = data["Class"]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    fraud_indices = y_test.index[y_test == 1]
    if fraud_indices.empty:
        raise ValueError("The held-out test dataset contains no fraud samples.")

    sample_index = int(fraud_indices[0])
    sample_features = {
        name: float(value)
        for name, value in X_test.loc[sample_index].items()
    }
    actual_class = int(y_test.loc[sample_index])
    return sample_features, actual_class, sample_index


def render_home(metrics: pd.DataFrame) -> None:
    st.title("Credit Card Fraud Detection")
    st.write(
        "Estimate whether a transaction may be fraudulent using the trained "
        "Random Forest model and the dataset's transaction features."
    )
    st.caption(
        "Model predictions are estimates, not guarantees. Review them alongside "
        "appropriate human and operational checks."
    )
    model_row = metrics.loc[metrics["model"] == "Random Forest"]
    if not model_row.empty:
        values = model_row.iloc[0]
        columns = st.columns(4)
        for column, label, key in zip(
            columns,
            ("Precision", "Recall", "F1-score", "PR-AUC"),
            ("precision", "recall", "f1_score", "pr_auc"),
            strict=True,
        ):
            column.metric(label, format_metric(float(values[key])))
    st.markdown("---")
    st.subheader("Explore the project")
    st.write(
        "Use **Dashboard** to view dataset charts, **Fraud Prediction** to evaluate "
        "a transaction, or **Model Performance** to compare the recorded test metrics."
    )


def render_dashboard() -> None:
    st.title("Dashboard")
    st.caption("Charts generated from the project dataset during data analysis.")
    chart_columns = st.columns(2)
    with chart_columns[0]:
        show_image("class_distribution.png", "Transaction class distribution")
        show_image("amount_distribution.png", "Transaction amount distribution")
    with chart_columns[1]:
        show_image("correlation_heatmap.png", "Feature correlation heatmap")


def render_prediction(model) -> None:
    st.title("Fraud Prediction")
    st.write(
        "Enter all transaction features used by the trained model. "
        "The prediction is an estimate, not a guarantee."
    )

    feature_names = list(getattr(model, "feature_names_in_", []))
    expected_features = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]
    if feature_names != expected_features:
        st.error(
            "The saved model feature schema does not match the expected dataset "
            "features. Prediction is unavailable until the model artifact is checked."
        )
        return

    load_sample_clicked = st.button("Load Sample Transaction")
    if load_sample_clicked:
        try:
            sample_features, actual_class, sample_index = load_sample_test_transaction(
                tuple(feature_names)
            )
            for feature_name, value in sample_features.items():
                st.session_state[f"prediction_{feature_name}"] = value
            st.session_state["sample_actual_class"] = actual_class
            st.session_state["sample_row_index"] = sample_index
            st.session_state.pop("latest_prediction", None)
        except (FileNotFoundError, OSError, ValueError) as error:
            st.error(f"Could not load a sample transaction: {error}")

    if "sample_actual_class" in st.session_state:
        actual_class = st.session_state["sample_actual_class"]
        actual_label = "Fraudulent (1)" if actual_class == 1 else "Legitimate (0)"
        st.info(
            f"Loaded held-out test sample (dataset row "
            f"{st.session_state['sample_row_index']}). Actual Class: {actual_label}. "
            "This reference label is not sent to the model."
        )

    for feature_name in feature_names:
        input_key = f"prediction_{feature_name}"
        if input_key not in st.session_state:
            st.session_state[input_key] = 0.0

    with st.form("fraud_prediction_form"):
        st.caption("Transaction details")
        transaction_columns = st.columns(2)
        feature_values = {}
        with transaction_columns[0]:
            feature_values["Time"] = st.number_input(
                "Time",
                min_value=0.0,
                step=1.0,
                key="prediction_Time",
                help="Seconds elapsed since the first transaction in the dataset.",
            )
        with transaction_columns[1]:
            feature_values["Amount"] = st.number_input(
                "Amount",
                min_value=0.0,
                step=1.0,
                key="prediction_Amount",
                help="Transaction amount in the dataset's currency units.",
            )

        st.caption("Anonymized transaction features")
        feature_columns = st.columns(3)
        for index, feature_name in enumerate(
            name for name in feature_names if name.startswith("V")
        ):
            with feature_columns[index % len(feature_columns)]:
                feature_values[feature_name] = st.number_input(
                    feature_name,
                    step=0.01,
                    format="%.6f",
                    key=f"prediction_{feature_name}",
                )

        submitted = st.form_submit_button("Predict transaction", width="stretch")

    if submitted:
        try:
            input_data = pd.DataFrame(
                [[feature_values[name] for name in feature_names]],
                columns=feature_names,
            )
            predicted_label = int(model.predict(input_data)[0])
            probability = None
            if hasattr(model, "predict_proba"):
                classes = list(model.classes_)
                if 1 in classes:
                    probability = float(model.predict_proba(input_data)[0][classes.index(1)])
            prediction_id = add_prediction(
                predicted_label=predicted_label,
                probability=probability,
                amount=float(feature_values["Amount"]),
            )
            st.session_state["latest_prediction"] = {
                "id": prediction_id,
                "label": predicted_label,
                "probability": probability,
            }
        except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
            st.error(f"Could not complete or save the prediction: {error}")

    latest_prediction = st.session_state.get("latest_prediction")
    if latest_prediction:
        label = latest_prediction["label"]
        if label == 1:
            st.error("Prediction: Potential Fraud")
        else:
            st.success("Prediction: Likely Normal")
        if latest_prediction["probability"] is not None:
            st.metric(
                "Estimated fraud probability",
                format_metric(latest_prediction["probability"]),
            )
        st.caption(
            f"Prediction ID: {latest_prediction['id']}. "
            "This is a model estimate, not a guarantee."
        )


def render_performance(metrics: pd.DataFrame) -> None:
    st.title("Model Performance")
    st.caption("Metrics below are loaded directly from reports/model_results.csv.")
    display_columns = ["model", *METRIC_COLUMNS]
    display_metrics = metrics[display_columns].copy()
    display_metrics[METRIC_COLUMNS] = display_metrics[METRIC_COLUMNS].map(
        format_metric
    )
    st.dataframe(
        display_metrics,
        hide_index=True,
        width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model"),
            "accuracy": st.column_config.TextColumn("Accuracy"),
            "precision": st.column_config.TextColumn("Precision"),
            "recall": st.column_config.TextColumn("Recall"),
            "f1_score": st.column_config.TextColumn("F1-score"),
            "roc_auc": st.column_config.TextColumn("ROC-AUC"),
            "pr_auc": st.column_config.TextColumn("PR-AUC"),
        },
    )

    st.subheader("Confusion matrices")
    model_names = metrics["model"].astype(str).tolist()
    chart_columns = st.columns(2)
    for index, model_name in enumerate(model_names):
        slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        with chart_columns[index % len(chart_columns)]:
            show_image(
                f"confusion_matrix_{slug}.png",
                f"{model_name} confusion matrix",
            )


def render_history() -> None:
    st.title("Prediction History")
    try:
        history = get_prediction_history()
    except (OSError, RuntimeError, sqlite3.Error) as error:
        st.error(f"Could not read prediction history: {error}")
        return

    if st.session_state.pop("history_cleared", False):
        st.success("Prediction history cleared.")

    if history:
        history_frame = pd.DataFrame(history).rename(
            columns={
                "id": "Prediction ID",
                "timestamp": "Timestamp",
                "predicted_label": "Predicted class (0=legitimate, 1=fraud)",
                "probability": "Fraud probability",
                "amount": "Amount",
            }
        )
        st.dataframe(history_frame, hide_index=True, width="stretch")
        if st.button("Clear prediction history", type="secondary"):
            try:
                clear_prediction_history()
                st.session_state.pop("latest_prediction", None)
                st.session_state["history_cleared"] = True
                st.rerun()
            except (OSError, RuntimeError, sqlite3.Error) as error:
                st.error(f"Could not clear prediction history: {error}")
    else:
        st.info("No predictions have been saved yet.")


def render_about() -> None:
    st.title("About")
    st.write(
        "This project demonstrates machine-learning-based credit card fraud "
        "classification using the existing trained Random Forest pipeline."
    )
    st.write(
        "The prediction form uses the model's complete input schema: `Time`, "
        "`V1`–`V28`, and `Amount`. `Amount` or `Time` alone are not sufficient."
    )
    st.write(
        "Predictions and the entered transaction amount are stored locally in "
        "SQLite. No card numbers or other cardholder details are requested or stored."
    )
    st.caption("Predictions are estimates, not guarantees.")


def main() -> None:
    try:
        model = load_model()
        metrics = load_metrics()
        initialize_database()
    except (FileNotFoundError, ValueError, OSError, RuntimeError, sqlite3.Error) as error:
        st.error(f"Application could not load the required project files: {error}")
        st.stop()

    if "Random Forest" not in set(metrics["model"].astype(str)):
        st.error("Random Forest metrics are not present in model_results.csv.")
        st.stop()

    with st.sidebar:
        st.markdown("### Fraud Detection")
        st.caption("Credit card transaction estimates")
        section = st.radio(
            "Navigation",
            [
                "Home",
                "Dashboard",
                "Fraud Prediction",
                "Model Performance",
                "Prediction History",
                "About",
            ],
            label_visibility="collapsed",
        )

    if section == "Home":
        render_home(metrics)
    elif section == "Dashboard":
        render_dashboard()
    elif section == "Fraud Prediction":
        render_prediction(model)
    elif section == "Model Performance":
        render_performance(metrics)
    elif section == "Prediction History":
        render_history()
    else:
        render_about()


if __name__ == "__main__":
    main()
