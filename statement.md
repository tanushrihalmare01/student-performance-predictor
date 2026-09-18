# Problem Statement

## Problem Statement

Educators often struggle to identify, early enough, which students are at
risk of failing a course. Manually reviewing study habits, attendance, and
past scores for every student does not scale and tends to catch problems
too late for meaningful intervention. There is a need for a lightweight
tool that can flag at-risk students early using a few simple, easily
collected indicators, so that timely academic support can be provided.

## Scope of the Project

This project builds a supervised machine-learning system that predicts
whether a student is likely to **Pass** or **Fail**, based on four inputs:
study hours per day, attendance percentage, previous academic score, and
average sleep hours. The scope covers the full pipeline: data
preparation, exploratory analysis, model training and comparison, model
evaluation, and a simple prediction interface. It does not cover
integration with a live college database or a web front-end — the focus
is on demonstrating correct application of core AI/ML concepts
(classification, feature scaling, train/test evaluation) rather than
production deployment.

## Target Users

- **Teachers / academic mentors** who want an early, data-driven signal
  of which students may need extra support.
- **Students** who want to self-check how their current habits
  (study time, attendance, sleep) relate to likely academic outcomes.
- **Academic administrators** looking for a simple prototype of
  predictive analytics in an educational setting.

## High-Level Features

- Automatic generation of a clean, structured student dataset
- Exploratory data analysis with visual charts
- Training and comparison of two ML classifiers (Logistic Regression,
  Decision Tree), with automatic selection of the best-performing model
- Quantitative evaluation: accuracy, precision, recall, F1-score,
  confusion matrix
- A simple prediction interface where a new student's details can be
  entered to get an instant Pass/Fail prediction with confidence score
- Input validation and error handling for user-supplied data
- Persisted model so predictions can be made instantly without retraining
