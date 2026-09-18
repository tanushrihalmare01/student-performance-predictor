"""
data_loader.py
---------------
Module 1: Data Input & Processing

Generates a reproducible synthetic student-performance dataset (so the
project runs offline with no external downloads), then cleans, scales
and splits it into train/test sets.

In a real deployment this module could be swapped to read a CSV/DB
without touching any other module (separation of concerns -> maintainability).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import RANDOM_SEED, N_SAMPLES, FEATURE_NAMES, TARGET_NAME, get_logger

logger = get_logger(__name__)


def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Create a synthetic but realistic student performance dataset."""
    rng = np.random.default_rng(RANDOM_SEED)

    study_hours = rng.normal(4, 2, n_samples).clip(0, 12)
    attendance_pct = rng.normal(75, 15, n_samples).clip(30, 100)
    previous_score = rng.normal(60, 15, n_samples).clip(0, 100)
    sleep_hours = rng.normal(6.5, 1.5, n_samples).clip(3, 10)

    # underlying "true" score used to derive pass/fail, plus noise
    composite = (
        0.35 * study_hours * 8
        + 0.30 * attendance_pct
        + 0.25 * previous_score
        + 0.10 * sleep_hours * 5
        + rng.normal(0, 8, n_samples)
    )
    result = (composite >= composite.mean()).astype(int)

    df = pd.DataFrame(
        {
            "study_hours": study_hours.round(2),
            "attendance_pct": attendance_pct.round(2),
            "previous_score": previous_score.round(2),
            "sleep_hours": sleep_hours.round(2),
            "result": result,
        }
    )
    logger.info(f"Generated synthetic dataset with {n_samples} rows.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data-cleaning step: drop duplicates/nulls (defensive, dataset is synthetic)."""
    before = len(df)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    after = len(df)
    if before != after:
        logger.info(f"Cleaned data: removed {before - after} rows.")
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2):
    """Split into train/test and scale features. Returns arrays + fitted scaler."""
    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Split data: {len(X_train)} train / {len(X_test)} test rows.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def load_data():
    """Convenience entry point used by main.py."""
    df = generate_dataset()
    df = clean_data(df)
    return df
