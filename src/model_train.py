"""
model_evaluate.py
------------------
Module 3 (core): Evaluation & Reporting.

Computes standard classification metrics and a confusion-matrix plot,
and writes a text evaluation report to outputs/evaluation_report.txt.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from .utils import DATA_DIR, REPORT_PATH, get_logger

logger = get_logger(__name__)


def evaluate_model(model, model_name, X_test, y_test) -> dict:
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
    }

    cm = confusion_matrix(y_test, preds)
    _plot_confusion_matrix(cm)

    report_text = classification_report(y_test, preds, target_names=["Fail", "Pass"])
    with open(REPORT_PATH, "w") as f:
        f.write(f"Model: {model_name}\n\n")
        f.write(report_text)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    logger.info(f"Evaluation complete: {metrics}")
    logger.info(f"Full report written to {REPORT_PATH}")
    return metrics


def _plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Fail", "Pass"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Fail", "Pass"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {out_path}")
