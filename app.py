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
        --background-color: #F5F6F8;
        --secondary-background-color: #FFFFFF;
        --text-color: #17191F;
        --primary-color: #635BFF;
        --border-color: #E5E7EB;
        --muted-color: #6B7280;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #F5F6F8;
        color: #17191F;
    }
    [data-testid="stMain"] :is(p, span, label, li, small, summary) {
        color: #4B5563;
    }
    [data-testid="stSidebar"] {
        background: #17191F;
        color: #F9FAFB;
    }
    [data-testid="stSidebar"] :is(p, span, label, li, small, strong, em, a) {
        color: #E5E7EB !important;
    }
    [data-testid="stSidebar"] > div { padding: 1rem 0.8rem 0.8rem; }
    .block-container {
        width: 100%;
        max-width: none;
        padding: 1.15rem 2rem 2rem;
    }
    [data-testid="stVerticalBlock"] { gap: 0.8rem; }
    [data-testid="stHorizontalBlock"] { gap: 1rem; }
    [data-testid="stMain"] h1 {
        color: #17191F;
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.15;
        letter-spacing: -0.04em;
        margin: 0;
    }
    [data-testid="stMain"] :is(h1, h2, h3, h4, h5, h6) {
        color: #17191F;
    }
    [data-testid="stMain"] :is(h2, h3, h4) {
        font-weight: 680;
        letter-spacing: -0.02em;
        margin: 0.3rem 0 0.1rem;
    }
    [data-testid="stMain"] p { line-height: 1.5; }
    [data-testid="stCaptionContainer"] { color: #6B7280; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 0.35rem; }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {
        width: 100%;
        border-radius: 8px;
        padding: 0.45rem 0.6rem;
        transition: background-color 150ms ease, color 150ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
        background: #292C35;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {
        background: #635BFF;
        box-shadow: inset 3px 0 #A9A4FF;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) * {
        color: #FFFFFF !important;
        font-weight: 650;
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.2;
        margin: 0.15rem 0 1.15rem;
    }
    .brand-shield {
        display: grid;
        width: 2.35rem;
        height: 2.35rem;
        place-items: center;
        border: 1px solid #4942B9;
        border-radius: 10px;
        background: #292653;
        color: #C5C2FF;
        font-size: 1.15rem;
    }
    .brand-caption {
        display: block;
        margin-top: 0.2rem;
        color: #AEB2BD;
        font-size: 0.72rem;
        font-weight: 450;
    }
    .sidebar-status {
        position: fixed;
        bottom: 1rem;
        left: 1rem;
        z-index: 10;
        width: 268px;
        padding: 0.8rem 0.65rem;
        border: 1px solid #343741;
        border-radius: 9px;
        background: #1F2128;
        color: #D1D5DB;
        font-size: 0.75rem;
    }
    .status-dot {
        display: inline-block;
        width: 0.48rem;
        height: 0.48rem;
        margin-right: 0.45rem;
        border-radius: 50%;
        background: #34D399;
    }
    .hero-banner {
        padding: 0.65rem 0.9rem;
        border: 1px solid #D9D7FF;
        border-radius: 10px;
        background: #EFEEFF;
        color: #423BB2;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .hero-card, .surface-card {
        padding: 1rem 1.1rem;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        background: #FFFFFF;
        box-shadow: 0 2px 8px rgba(23, 25, 31, 0.04);
        transition: box-shadow 160ms ease, transform 160ms ease;
    }
    .surface-card:hover {
        box-shadow: 0 5px 16px rgba(23, 25, 31, 0.08);
        transform: translateY(-1px);
    }
    .surface-card h3, .surface-card h4 {
        margin: 0 0 0.35rem;
        color: #17191F;
        font-size: 0.98rem;
    }
    .surface-card p {
        margin: 0;
        color: #6B7280;
        font-size: 0.84rem;
        line-height: 1.5;
    }
    .eyebrow {
        margin-bottom: 0.35rem;
        color: #635BFF;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }
    .section-intro {
        margin-top: -0.35rem;
        color: #6B7280;
        font-size: 0.88rem;
    }
    div[data-testid="stMetric"] {
        min-height: 5.4rem;
        padding: 0.65rem 0.75rem;
        border: 1px solid #E5E7EB;
        border-radius: 11px;
        background: #FFFFFF;
        box-shadow: 0 2px 7px rgba(23, 25, 31, 0.035);
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricDelta"] { color: #6B7280 !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #17191F !important;
        font-weight: 700;
        font-size: 1.35rem;
        line-height: 1.25;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.74rem;
        font-weight: 550;
    }
    [data-testid="stNumberInputContainer"],
    [data-testid="stTextInputRootElement"],
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border-color: #D1D5DB;
        border-radius: 7px;
        background: #FFFFFF;
    }
    [data-testid="stNumberInputField"],
    [data-testid="stTextInputRootElement"] input {
        color: #17191F !important;
        font-size: 0.88rem;
    }
    [data-testid="stNumberInputContainer"] button {
        color: #4B5563;
    }
    [data-testid="stForm"] {
        padding: 1rem;
        border-color: #E5E7EB;
        border-radius: 12px;
        background: #FFFFFF;
        box-shadow: 0 2px 8px rgba(23, 25, 31, 0.035);
    }
    [data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
    [data-testid="stWidgetLabel"] { margin-bottom: 0.15rem; }
    [data-testid="stImage"] { padding: 0.3rem; }
    [data-testid="stImage"] img {
        max-height: 340px;
        object-fit: contain;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        overflow: hidden;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        min-height: 2.45rem;
        padding: 0.45rem 0.9rem;
        border: 1px solid #635BFF;
        border-radius: 8px;
        background: #635BFF;
        color: #FFFFFF !important;
        font-weight: 650;
        transition: background-color 150ms ease, box-shadow 150ms ease, transform 150ms ease;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        border-color: #5149E5;
        background: #5149E5;
        color: #FFFFFF !important;
        box-shadow: 0 3px 10px rgba(99, 91, 255, 0.22);
        transform: translateY(-1px);
    }
    div.stButton > button:focus-visible,
    div.stFormSubmitButton > button:focus-visible {
        outline: 3px solid rgba(99, 91, 255, 0.35);
        outline-offset: 2px;
    }
    div.stButton > button[kind="secondary"] {
        border-color: #F2B8BE;
        background: #FFF5F5;
        color: #B42332 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #DC3545;
        background: #FDEBED;
        color: #A51D2A !important;
    }
    .st-key-load_sample_transaction button {
        border-color: #C9C6FF;
        background: #F0EFFF;
        color: #5149D8 !important;
    }
    .st-key-load_sample_transaction button:hover {
        border-color: #635BFF;
        background: #E5E3FF;
        color: #4038C4 !important;
    }
    .result-card {
        padding: 1rem 1.1rem;
        border: 1px solid #D1D5DB;
        border-left: 5px solid #198754;
        border-radius: 11px;
        background: #FFFFFF;
        box-shadow: 0 3px 12px rgba(23, 25, 31, 0.05);
    }
    .result-card.fraud {
        border-color: #F1C0C5;
        border-left-color: #DC3545;
        background: #FFF8F8;
    }
    .result-card.normal {
        border-color: #B9DEC9;
        border-left-color: #198754;
        background: #F7FCF9;
    }
    .result-title {
        color: #17191F;
        font-size: 1.1rem;
        font-weight: 750;
    }
    .result-detail { margin-top: 0.2rem; color: #6B7280; font-size: 0.82rem; }
    .workflow-step {
        height: 100%;
        padding: 0.9rem;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        background: #FFFFFF;
    }
    .workflow-step b { color: #17191F; }
    .workflow-step p { margin: 0.3rem 0 0; color: #6B7280; font-size: 0.82rem; }
    .st-key-predict_navigation button { width: 100%; }
    @media (max-width: 700px) {
        .block-container { padding: 0.8rem 0.85rem 1.2rem; }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 calc(50% - 0.5rem) !important;
            min-width: calc(50% - 0.5rem) !important;
        }
        [data-testid="stMetric"] { min-height: 5rem; padding: 0.55rem; }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
            font-size: 1.12rem;
        }
        [data-testid="stImage"] img { max-height: 280px; }
        [data-testid="stForm"] { padding: 0.7rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
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
def load_dataset_summary() -> dict[str, float | int]:
    """Read actual transaction counts and amount summaries from the dataset."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH, usecols=["Amount", "Class"])
    class_counts = data["Class"].value_counts()
    if not {0, 1}.issubset(class_counts.index):
        raise ValueError("Dataset must contain legitimate and fraudulent classes.")
    return {
        "transactions": int(len(data)),
        "fraud_count": int(class_counts[1]),
        "legitimate_count": int(class_counts[0]),
        "fraud_percentage": float(data["Class"].mean() * 100),
        "average_amount": float(data["Amount"].mean()),
    }


def render_metric_cards(
    metrics: pd.DataFrame,
    metric_keys: tuple[str, ...],
    icons: dict[str, str] | None = None,
) -> None:
    """Show recorded Random Forest metrics in a horizontal row."""
    model_row = metrics.loc[metrics["model"] == "Random Forest"]
    if model_row.empty:
        return

    labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }
    values = model_row.iloc[0]
    columns = st.columns(len(metric_keys))
    for column, key in zip(columns, metric_keys, strict=True):
        label = f"{icons[key]} {labels[key]}" if icons and key in icons else labels[key]
        column.metric(label, format_metric(float(values[key])))


def navigate_to_prediction() -> None:
    st.session_state["navigation_section"] = "Fraud Prediction"


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
    st.markdown(
        '<div class="hero-banner">🛡️ &nbsp; INTELLIGENT TRANSACTION RISK MONITORING</div>',
        unsafe_allow_html=True,
    )
    st.title("Credit Card Fraud Detection")
    st.write(
        "AI-powered transaction risk analysis to help identify potentially fraudulent activity."
    )
    render_metric_cards(
        metrics,
        ("precision", "recall", "f1_score", "roc_auc", "pr_auc"),
        {
            "precision": "◎",
            "recall": "↗",
            "f1_score": "◈",
            "roc_auc": "⌁",
            "pr_auc": "▤",
        },
    )
    action, note = st.columns([1, 3], vertical_alignment="center")
    with action:
        st.button(
            "Run a prediction",
            key="predict_navigation",
            icon="🔍",
            type="primary",
            on_click=navigate_to_prediction,
        )
    with note:
        st.caption("Predictions are estimates, not guarantees.")
    st.subheader("Detection workflow")
    workflow_columns = st.columns(3)
    workflow = (
        ("01", "Transaction features", "Provide Time, Amount, and anonymized V1–V28 inputs."),
        ("02", "Risk scoring", "The saved Random Forest pipeline evaluates the complete feature set."),
        ("03", "Review the estimate", "See the predicted class and fraud probability when available."),
    )
    for column, (number, title, description) in zip(
        workflow_columns, workflow, strict=True
    ):
        with column:
            st.markdown(
                f'<div class="workflow-step"><div class="eyebrow">{number}</div>'
                f"<b>{title}</b><p>{description}</p></div>",
                unsafe_allow_html=True,
            )


def render_dashboard(metrics: pd.DataFrame) -> None:
    st.title("Analytics Dashboard")
    st.markdown(
        '<div class="section-intro">A snapshot of transaction activity and model quality, '
        "computed from the project dataset and evaluation report.</div>",
        unsafe_allow_html=True,
    )
    try:
        summary = load_dataset_summary()
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Could not load dataset statistics: {error}")
        return

    statistic_columns = st.columns(4)
    statistics = (
        ("▦  Transactions", f"{summary['transactions']:,}"),
        ("✓  Legitimate", f"{summary['legitimate_count']:,}"),
        ("!  Fraudulent", f"{summary['fraud_count']:,}"),
        ("◉  Fraud share", f"{summary['fraud_percentage']:.3f}%"),
    )
    for column, (label, value) in zip(statistic_columns, statistics, strict=True):
        column.metric(label, value)
    st.markdown(
        f'<div class="surface-card"><span class="eyebrow">Model at a glance</span>'
        f"<b>Random Forest</b><p>Average transaction amount in the dataset: "
        f"<strong>{summary['average_amount']:.2f}</strong> (dataset units). "
        "Performance values below are from the saved evaluation results.</p></div>",
        unsafe_allow_html=True,
    )
    render_metric_cards(
        metrics,
        ("precision", "recall", "f1_score", "roc_auc", "pr_auc"),
    )
    st.subheader("Transaction patterns")
    st.caption("Dataset analysis charts")
    chart_columns = st.columns(2)
    with chart_columns[0]:
        show_image("class_distribution.png", "Transaction class distribution")
    with chart_columns[1]:
        show_image("amount_distribution.png", "Transaction amount distribution")
    st.subheader("Feature relationships")
    heatmap_columns = st.columns([1, 3, 1])
    with heatmap_columns[1]:
        show_image("correlation_heatmap.png", "Feature correlation heatmap")


def render_prediction(model) -> None:
    st.title("Fraud Prediction")
    st.markdown(
        '<div class="section-intro">Evaluate a transaction using the complete feature set. '
        "Predictions are estimates, not guarantees.</div>",
        unsafe_allow_html=True,
    )

    feature_names = list(getattr(model, "feature_names_in_", []))
    expected_features = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]
    if feature_names != expected_features:
        st.error(
            "The saved model feature schema does not match the expected dataset "
            "features. Prediction is unavailable until the model artifact is checked."
        )
        return

    load_sample_clicked = st.button(
        "Load Sample Transaction",
        icon="📄",
        type="primary",
        key="load_sample_transaction",
    )
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
            f"Sample from held-out test data · row {st.session_state['sample_row_index']} "
            f"· Actual Class: {actual_label} · Reference label is not sent to the model."
        )

    for feature_name in feature_names:
        input_key = f"prediction_{feature_name}"
        if input_key not in st.session_state:
            st.session_state[input_key] = 0.0

    with st.form("fraud_prediction_form"):
        st.markdown('<div class="eyebrow">Transaction details</div>', unsafe_allow_html=True)
        feature_values = {}
        feature_columns = st.columns(4)
        detail_features = [name for name in ("Time", "Amount") if name in feature_names]
        for index, feature_name in enumerate(detail_features):
            with feature_columns[index % len(feature_columns)]:
                if feature_name == "Time":
                    feature_values[feature_name] = st.number_input(
                        feature_name,
                        min_value=0.0,
                        step=1.0,
                        key=f"prediction_{feature_name}",
                        help="Seconds elapsed since the first transaction in the dataset.",
                    )
                elif feature_name == "Amount":
                    feature_values[feature_name] = st.number_input(
                        feature_name,
                        min_value=0.0,
                        step=1.0,
                        key=f"prediction_{feature_name}",
                        help="Transaction amount in the dataset's currency units.",
                    )
        st.markdown('<div class="eyebrow">PCA features · V1–V28</div>', unsafe_allow_html=True)
        pca_features = [name for name in feature_names if name.startswith("V")]
        pca_columns = st.columns(4)
        for index, feature_name in enumerate(pca_features):
            with pca_columns[index % len(pca_columns)]:
                feature_values[feature_name] = st.number_input(
                    feature_name,
                    step=0.01,
                    format="%.6f",
                    key=f"prediction_{feature_name}",
                )

        submitted = st.form_submit_button(
            "Analyze transaction",
            icon="🔍",
            width="stretch",
        )

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
        result_class = "fraud" if label == 1 else "normal"
        result_label = "Potential Fraud" if label == 1 else "Likely Normal"
        result_icon = "⚠️" if label == 1 else "✓"
        probability_detail = ""
        if latest_prediction["probability"] is not None:
            probability_detail = (
                f'<div class="result-detail">Estimated fraud probability: '
                f'<strong>{format_metric(latest_prediction["probability"])}</strong></div>'
            )
        st.markdown(
            f'<div class="result-card {result_class}">'
            f'<div class="eyebrow">Model assessment</div>'
            f'<div class="result-title">{result_icon} &nbsp; {result_label}</div>'
            f"{probability_detail}"
            f'<div class="result-detail">Prediction ID {latest_prediction["id"]} · '
            "Model output is an estimate, not a guarantee.</div></div>",
            unsafe_allow_html=True,
        )


def render_performance(metrics: pd.DataFrame) -> None:
    st.title("Model Performance")
    st.markdown(
        '<div class="section-intro">All values are loaded from the saved test-set evaluation report.</div>',
        unsafe_allow_html=True,
    )
    random_forest = metrics.loc[metrics["model"] == "Random Forest"]
    if not random_forest.empty:
        st.markdown(
            '<div class="surface-card"><span class="eyebrow">Selected model</span>'
            "<h3>🌲 &nbsp; Random Forest</h3>"
            "<p>Selected for the strongest F1-score among the evaluated models, "
            "with fraud precision and recall shown below.</p></div>",
            unsafe_allow_html=True,
        )
        render_metric_cards(
            metrics,
            ("precision", "recall", "f1_score", "roc_auc", "pr_auc"),
        )

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
    st.caption("Held-out test set · one chart per evaluated model")
    model_names = metrics["model"].astype(str).tolist()
    chart_columns = st.columns(2)
    for index, model_name in enumerate(model_names):
        slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        with chart_columns[index % len(chart_columns)]:
            st.markdown(
                f'<div class="eyebrow">{model_name}</div>',
                unsafe_allow_html=True,
            )
            show_image(
                f"confusion_matrix_{slug}.png",
                "Actual vs predicted class counts",
            )


def render_history() -> None:
    st.title("Prediction History")
    st.markdown(
        '<div class="section-intro">Locally saved model estimates. No cardholder or card-number data is stored.</div>',
        unsafe_allow_html=True,
    )
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
        st.dataframe(
            history_frame,
            hide_index=True,
            width="stretch",
            height=min(420, 38 + 35 * len(history)),
        )
        action, count = st.columns([1, 4], vertical_alignment="center")
        with action:
            clear_clicked = st.button(
                "Clear History",
                type="secondary",
                icon="🗑️",
                key="clear_prediction_history",
            )
        with count:
            st.caption(f"{len(history):,} prediction record(s)")
        if clear_clicked:
            try:
                clear_prediction_history()
                st.session_state.pop("latest_prediction", None)
                st.session_state["history_cleared"] = True
                st.rerun()
            except (OSError, RuntimeError, sqlite3.Error) as error:
                st.error(f"Could not clear prediction history: {error}")
    else:
        st.markdown(
            '<div class="surface-card empty-state"><div class="empty-icon">◷</div>'
            "<h3>No predictions yet</h3>"
            "<p>Completed transaction assessments will appear here with their "
            "timestamp, predicted class, probability, and amount.</p></div>",
            unsafe_allow_html=True,
        )


def render_about() -> None:
    st.title("About")
    st.markdown(
        '<div class="section-intro">An interpretable project demonstration for machine-learning-based '
        "transaction risk screening.</div>",
        unsafe_allow_html=True,
    )
    cards = st.columns(3)
    about_cards = (
        (
            "🎯",
            "Objective",
            "Compare fraud-classification models and surface risk estimates using all "
            "30 dataset features: Time, V1–V28, and Amount.",
        ),
        (
            "🧰",
            "Technology",
            "Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, "
            "Matplotlib, Seaborn, Streamlit, SQLite, and joblib.",
        ),
        (
            "🔒",
            "Privacy",
            "Prediction history is stored locally. The form never requests card "
            "numbers or cardholder information.",
        ),
    )
    for column, (icon, title, description) in zip(cards, about_cards, strict=True):
        with column:
            st.markdown(
                f'<div class="surface-card about-card"><div class="about-icon">{icon}</div>'
                f"<h3>{title}</h3><p>{description}</p></div>",
                unsafe_allow_html=True,
            )
    st.subheader("Workflow")
    workflow_columns = st.columns(4)
    workflow = (
        ("01", "Prepare", "Stratified split, scale Time and Amount, SMOTE on training data."),
        ("02", "Compare", "Evaluate five classifiers on the same untouched test split."),
        ("03", "Predict", "Use the saved Random Forest pipeline and full feature set."),
        ("04", "Review", "See saved metrics, confusion matrices, and local prediction history."),
    )
    for column, (number, title, description) in zip(
        workflow_columns, workflow, strict=True
    ):
        with column:
            st.markdown(
                f'<div class="workflow-step"><div class="eyebrow">{number}</div>'
                f"<b>{title}</b><p>{description}</p></div>",
                unsafe_allow_html=True,
            )
    st.caption("Model predictions are estimates, not guarantees.")


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
        st.markdown(
            '<div class="brand-lockup"><div class="brand-shield">🛡️</div>'
            "<div>Fraud Detection<span class=\"brand-caption\">FINTECH ANALYTICS</span></div></div>",
            unsafe_allow_html=True,
        )
        navigation_icons = {
            "Home": "⌂",
            "Dashboard": "▦",
            "Fraud Prediction": "⌕",
            "Model Performance": "◈",
            "Prediction History": "◷",
            "About": "ⓘ",
        }
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
            format_func=lambda item: f"{navigation_icons[item]}   {item}",
            key="navigation_section",
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="sidebar-status"><span class="status-dot"></span>'
            "Model ready · Random Forest</div>",
            unsafe_allow_html=True,
        )

    if section == "Home":
        render_home(metrics)
    elif section == "Dashboard":
        render_dashboard(metrics)
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
