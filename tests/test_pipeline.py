"""
test_pipeline.py
-----------------
Basic validation/unit tests (rubric: "Testing wherever applicable").
Run with:  python -m pytest tests/  (from project root)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.data_loader import generate_dataset, clean_data, split_and_scale
from src.utils import validate_student_input, FEATURE_NAMES


def test_dataset_shape():
    df = generate_dataset(n_samples=100)
    assert len(df) == 100
    assert set(FEATURE_NAMES + ["result"]).issubset(df.columns)


def test_dataset_no_nulls():
    df = clean_data(generate_dataset(n_samples=100))
    assert df.isnull().sum().sum() == 0


def test_split_shapes():
    df = generate_dataset(n_samples=100)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, test_size=0.2)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert X_train.shape[1] == len(FEATURE_NAMES)


def test_validate_student_input_valid():
    assert validate_student_input(5, 80, 70, 7) is True


def test_validate_student_input_invalid_range():
    with pytest.raises(ValueError):
        validate_student_input(-1, 80, 70, 7)


def test_validate_student_input_invalid_type():
    with pytest.raises(ValueError):
        validate_student_input("five", 80, 70, 7)
