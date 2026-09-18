"""
model_train.py
---------------
Module 2 (core): Model Training.

Trains two candidate classifiers (Logistic Regression, Decision Tree),
picks the better one on validation accuracy, and persists it to disk
with joblib so predict.py can reuse it without retraining every time.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from .utils import RANDOM_SEED, MODEL_PATH, SCALER_PATH, get_logger

logger = get_logger(__name__)


def train_candidates(X_train, y_train, X_test, y_test):
    """Train candidate models and return the best one with its name & accuracy."""
    candidates = {
        "LogisticRegression": LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=4),
    }

    best_model, best_name, best_acc = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"{name} validation accuracy: {acc:.3f}")
        if acc > best_acc:
            best_model, best_name, best_acc = model, name, acc

    logger.info(f"Selected best model: {best_name} (accuracy={best_acc:.3f})")
    return best_model, best_name, best_acc


def save_model(model, scaler):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
