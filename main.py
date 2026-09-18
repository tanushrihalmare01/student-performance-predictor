"""
main.py
-------
Orchestrates the full pipeline end-to-end:
  1. Load & preprocess data           (src/data_loader.py)
  2. Run EDA and save charts          (src/eda.py)
  3. Train & select best model        (src/model_train.py)
  4. Evaluate & save report/plots     (src/model_evaluate.py)
  5. Run a sample prediction          (src/predict.py)

Run with:  python main.py
"""

from src.data_loader import load_data, split_and_scale
from src.eda import run_eda
from src.model_train import train_candidates, save_model
from src.model_evaluate import evaluate_model
from src.predict import predict_student
from src.utils import get_logger

logger = get_logger("main")


def run_pipeline():
    logger.info("=== Step 1: Loading & preprocessing data ===")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    logger.info("=== Step 2: Running EDA ===")
    run_eda(df)

    logger.info("=== Step 3: Training models ===")
    model, model_name, val_acc = train_candidates(X_train, y_train, X_test, y_test)
    save_model(model, scaler)

    logger.info("=== Step 4: Evaluating best model ===")
    metrics = evaluate_model(model, model_name, X_test, y_test)

    logger.info("=== Step 5: Sample prediction ===")
    sample = predict_student(study_hours=5, attendance_pct=80, previous_score=70, sleep_hours=7)
    logger.info(f"Sample prediction result: {sample}")

    print("\n================ PIPELINE COMPLETE ================")
    print(f"Best model     : {model_name}")
    print(f"Accuracy       : {metrics['accuracy']:.3f}")
    print(f"Precision      : {metrics['precision']:.3f}")
    print(f"Recall         : {metrics['recall']:.3f}")
    print(f"F1-score       : {metrics['f1_score']:.3f}")
    print(f"Sample predict : {sample}")
    print("Charts & report saved in outputs/")
    print("=====================================================\n")


if __name__ == "__main__":
    run_pipeline()
