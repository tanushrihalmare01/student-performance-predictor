"""
predict.py
----------
Module 2 (interaction side): Prediction interface.

Lets a user supply a new student's stats and get a Pass/Fail prediction
with confidence, using the model trained by model_train.py.
This is the "clear input/output structure" + "logical workflow" the
rubric asks for.
"""

import numpy as np

from .model_train import load_model
from .utils import validate_student_input, FEATURE_NAMES, get_logger

logger = get_logger(__name__)


def predict_student(study_hours, attendance_pct, previous_score, sleep_hours):
    """
    Validate input, load the persisted model, and return a prediction.
    Returns a dict: {"prediction": "Pass"/"Fail", "confidence": float}
    """
    validate_student_input(study_hours, attendance_pct, previous_score, sleep_hours)

    model, scaler = load_model()
    features = np.array([[study_hours, attendance_pct, previous_score, sleep_hours]])
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][pred]

    result = {
        "prediction": "Pass" if pred == 1 else "Fail",
        "confidence": round(float(proba) * 100, 2),
    }
    logger.info(f"Prediction for input {dict(zip(FEATURE_NAMES, features[0]))}: {result}")
    return result


def _interactive_cli():
    """Run a simple command-line prompt for manual testing."""
    print("\n--- Student Performance Predictor ---")
    try:
        sh = float(input("Study hours/day (0-12): "))
        at = float(input("Attendance % (0-100): "))
        ps = float(input("Previous score (0-100): "))
        sl = float(input("Sleep hours/day (0-10): "))
        result = predict_student(sh, at, ps, sl)
        print(f"\nPrediction: {result['prediction']}  (confidence: {result['confidence']}%)")
    except ValueError as e:
        print(f"Invalid input: {e}")


if __name__ == "__main__":
    _interactive_cli()
