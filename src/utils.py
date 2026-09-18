"""
utils.py
--------
Shared configuration, logging setup, and helper functions used across
the Student Performance Predictor project.

Non-functional requirements addressed here:
- Maintainability: single source of truth for config values.
- Logging/Monitoring: consistent logging format across all modules.
- Error handling: a small helper to validate numeric ranges.
"""

import logging
import os

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_SAMPLES = 500                      # size of the synthetic dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_PATH = os.path.join(DATA_DIR, "model.joblib")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.joblib")
REPORT_PATH = os.path.join(DATA_DIR, "evaluation_report.txt")

FEATURE_NAMES = ["study_hours", "attendance_pct", "previous_score", "sleep_hours"]
TARGET_NAME = "result"  # 0 = Fail, 1 = Pass

os.makedirs(DATA_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger so every module logs consistently."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def validate_student_input(study_hours, attendance_pct, previous_score, sleep_hours):
    """
    Validate user-supplied values before prediction.
    Raises ValueError with a clear message on invalid input.
    (Error handling strategy required by the rubric.)
    """
    checks = {
        "study_hours": (study_hours, 0, 24),
        "attendance_pct": (attendance_pct, 0, 100),
        "previous_score": (previous_score, 0, 100),
        "sleep_hours": (sleep_hours, 0, 24),
    }
    for field, (value, lo, hi) in checks.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric, got {type(value).__name__}")
        if not (lo <= value <= hi):
            raise ValueError(f"{field} must be between {lo} and {hi}, got {value}")
    return True
