# Employee Attrition MLOps Project

## Checkpoint 2 --- Code Quality & Experiment Tracking

------------------------------------------------------------------------

## Project Overview

This project aims to build an end-to-end MLOps pipeline to predict
employee attrition using machine learning.\
Checkpoint 2 focuses on improving software quality, testing discipline,
and experiment tracking.

------------------------------------------------------------------------

## Checkpoint 2 Objectives

The following deliverables were implemented:

-   ✔ Pre-commit hooks configured (Black, isort, Flake8)
-   ✔ Unit tests with ≥ 60% coverage
-   ✔ Tests runnable locally using pytest
-   ✔ MLflow integrated for experiment tracking
-   ✔ Logging of parameters, metrics, and model artifacts
-   ✔ Clear experiment naming strategy
-   ✔ Clean and meaningful Git commit history
-   ✔ Updated README documentation

------------------------------------------------------------------------

##  Project Structure

    employee-attrition-mlops/
    │
    ├── src/
    │   └── attrition/
    │       ├── data/
    │       ├── models/
    │       ├── utils/
    │
    ├── scripts/
    │   └── train_pipeline.py
    │
    ├── tests/
    │
    ├── pyproject.toml
    ├── uv.lock
    ├── .pre-commit-config.yaml
    └── README.md

------------------------------------------------------------------------

## Unit Testing

Tests are implemented using `pytest` and coverage is measured using
`pytest-cov`.

Run tests:

    pytest --cov=src

Current coverage: **≥ 70%**

------------------------------------------------------------------------

## Code Quality

Pre-commit hooks ensure:

-   Automatic formatting (Black)
-   Import sorting (isort)
-   Linting (Flake8)

Activate pre-commit:

    pre-commit install

Run manually:

    pre-commit run --all-files

------------------------------------------------------------------------

## MLflow Experiment Tracking

MLflow is integrated to track:

-   Model parameters
-   Evaluation metrics
-   Model artifacts

Run training:

    python scripts/train_pipeline.py

Start MLflow UI:

    mlflow ui

Open in browser:

    http://127.0.0.1:5000

Experiment name used:

    Attrition_Experiments

Run name example:

    LR_baseline_v1

------------------------------------------------------------------------

##  Reproducibility

Environment managed using UV:

    uv sync

Dependencies tracked in: - `pyproject.toml` - `uv.lock`

------------------------------------------------------------------------

## Team Collaboration

-   All members contribute via GitHub
-   Clean commit messages following conventional format:
    -   feat:
    -   test:
    -   chore:
    -   docs:
    -   refactor:

------------------------------------------------------------------------

