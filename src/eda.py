"""
eda.py
------
Module 3 (Reporting/Visualization side): Exploratory Data Analysis.

Produces simple, saved charts so the pipeline has something visual
to include as screenshots/results in the project report.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import os

from .utils import DATA_DIR, get_logger

logger = get_logger(__name__)


def run_eda(df) -> None:
    """Generate and save a small set of EDA plots."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(df["study_hours"], bins=15, color="#4C72B0")
    axes[0, 0].set_title("Study Hours Distribution")

    axes[0, 1].scatter(df["attendance_pct"], df["previous_score"],
                        c=df["result"], cmap="coolwarm", alpha=0.6)
    axes[0, 1].set_title("Attendance vs Previous Score (colored by result)")
    axes[0, 1].set_xlabel("Attendance %")
    axes[0, 1].set_ylabel("Previous Score")

    df["result"].value_counts().sort_index().plot(
        kind="bar", ax=axes[1, 0], color=["#C44E52", "#55A868"]
    )
    axes[1, 0].set_title("Pass(1) / Fail(0) Count")
    axes[1, 0].set_xticklabels(["Fail", "Pass"], rotation=0)

    df.drop(columns=["result"]).boxplot(ax=axes[1, 1])
    axes[1, 1].set_title("Feature Spread (Boxplot)")
    axes[1, 1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, "eda_overview.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved EDA chart to {out_path}")
