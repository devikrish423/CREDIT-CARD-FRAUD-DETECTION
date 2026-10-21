"""Explore the credit card fraud dataset and save summary visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"


def analyze_dataset(data_path: Path = DATA_PATH) -> None:
    """Print dataset diagnostics and save the requested plots."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    required_columns = {"Amount", "Class"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print(f"Dataset shape: {data.shape[0]:,} rows x {data.shape[1]} columns")
    print("\nColumns:")
    print(", ".join(data.columns))
    print("\nData types:")
    print(data.dtypes.to_string())
    print("\nMissing values by column:")
    print(data.isna().sum().to_string())

    duplicate_count = int(data.duplicated().sum())
    print(f"\nDuplicate rows: {duplicate_count:,}")

    class_counts = data["Class"].value_counts().sort_index()
    class_percentages = data["Class"].value_counts(normalize=True).sort_index() * 100
    class_summary = pd.DataFrame(
        {"Count": class_counts, "Percentage": class_percentages}
    )
    class_summary.index.name = "Class"
    class_summary.index = class_summary.index.map(
        {0: "Legitimate (0)", 1: "Fraudulent (1)"}
    )
    print("\nTransaction class distribution:")
    print(class_summary.to_string(formatters={"Percentage": "{:.4f}%".format}))

    unknown_classes = set(data["Class"].dropna().unique()).difference({0, 1})
    if unknown_classes:
        raise ValueError(f"Unexpected class labels: {sorted(unknown_classes)}")

    class_labels = {0: "Legitimate (0)", 1: "Fraudulent (1)"}
    class_plot_data = class_counts.rename(index=class_labels).rename_axis("Class")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=class_plot_data.index,
        y=class_plot_data.values,
        hue=class_plot_data.index,
        palette=["#555555", "#111111"],
        legend=False,
        ax=ax,
    )
    ax.set(title="Transaction Class Distribution", xlabel="Transaction class", ylabel="Count")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "class_distribution.png", dpi=150)
    plt.close(fig)

    amount_plot_data = data.loc[data["Class"].isin(class_labels), ["Amount", "Class"]].copy()
    amount_plot_data["Class"] = amount_plot_data["Class"].map(class_labels)
    amount_plot_data["Log1p amount"] = np.log1p(amount_plot_data["Amount"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (label, group) in zip(
        axes, amount_plot_data.groupby("Class", sort=False), strict=True
    ):
        sns.histplot(
            data=group,
            x="Log1p amount",
            stat="density",
            bins=50,
            color="#555555" if label.startswith("Legitimate") else "#111111",
            ax=ax,
        )
        ax.set(title=label, xlabel="log(1 + Amount)", ylabel="Density")
    fig.suptitle("Transaction Amount Distribution by Class")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "amount_distribution.png", dpi=150)
    plt.close(fig)

    correlation = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.15,
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\nCharts saved to: {FIGURES_PATH}")
    for filename in (
        "class_distribution.png",
        "amount_distribution.png",
        "correlation_heatmap.png",
    ):
        print(f"- {filename}")


if __name__ == "__main__":
    analyze_dataset()
