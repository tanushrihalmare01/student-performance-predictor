# Student Performance Predictor


An AI/ML mini-project that predicts whether a student will **Pass or Fail**
based on their study habits, attendance, past performance, and sleep,
using classical machine learning (Logistic Regression & Decision Tree).

Built for the **Fundamentals of AI & ML** course (VITyarthi – Build Your Own Project).

## Overview

The project follows a complete, end-to-end ML pipeline:

1. **Data Input & Processing** – generates and cleans a synthetic student dataset
   (500 records) so the project runs fully offline and reproducibly.
2. **Model Training & Prediction** – trains and compares two classifiers
   (Logistic Regression, Decision Tree), auto-selects the best one, and
   exposes a simple prediction interface for new students.
3. **Reporting & Visualization** – runs EDA, evaluates the model
   (accuracy/precision/recall/F1, confusion matrix), and saves charts + a
   text report.

## Features

- Synthetic, reproducible dataset generation (no internet/dataset download needed)
- Data cleaning & feature scaling
- Two ML models trained and automatically compared; best one selected
- Full evaluation report: accuracy, precision, recall, F1-score, confusion matrix
- EDA visualizations (distribution plots, scatter plot, boxplot)
- CLI prediction tool for new student input, with input validation
- Model persistence (joblib) so predictions don't require retraining
- Unit tests covering data pipeline and input validation
- Structured logging throughout the pipeline

## Technologies / Tools Used

- Python 3
- scikit-learn (Logistic Regression, Decision Tree, metrics)
- pandas / numpy (data handling)
- matplotlib (visualization)
- joblib (model persistence)
- pytest (testing)

## Project Structure

```
student-performance-predictor/
├── main.py                     # Orchestrates the full pipeline
├── requirements.txt
├── README.md
├── statement.md
├── src/
│   ├── utils.py                # Config, logging, input validation
│   ├── data_loader.py          # Data generation, cleaning, splitting
│   ├── eda.py                  # Exploratory data analysis / charts
│   ├── model_train.py          # Model training & selection
│   ├── model_evaluate.py       # Metrics, confusion matrix, report
│   └── predict.py              # Prediction CLI / function
├── tests/
│   └── test_pipeline.py        # Unit tests
└── outputs/                    # Generated charts, model, reports (created on run)
```

## Steps to Install & Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd student-performance-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (trains model, generates charts & report)
python main.py

# 4. Try a prediction interactively
python -m src.predict
```

Outputs (EDA chart, confusion matrix, evaluation report, saved model) are
written to the `outputs/` folder.

## Instructions for Testing

```bash
python -m pytest tests/ -v
```

This runs unit tests covering:
- Dataset generation shape/columns
- Data cleaning (no nulls)
- Train/test split correctness
- Input validation (valid and invalid cases)

## Screenshots / Results

<img width="1280" height="719" alt="image" src="https://github.com/user-attachments/assets/2905b232-7e42-4b80-8ee5-2eaf9d2127ce" />

