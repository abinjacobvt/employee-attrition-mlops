Employee Attrition MLOps Project
MSc Data Engineering & Artificial Intelligence – ESILV
📌 Project Overview

This project implements a complete end-to-end MLOps pipeline to predict employee attrition using machine learning.

The objective is to simulate a production-ready ML system that includes:

Data preprocessing and model training

Experiment tracking using MLflow

Code quality and testing discipline

REST API model serving using FastAPI

Docker containerization

Team-based Git collaboration

The project is structured across multiple checkpoints to progressively build a full MLOps workflow.

🎯 Problem Statement

Employee attrition prediction helps organizations identify employees who are likely to leave the company.

Using historical HR data, we build a classification model that predicts:

0 → No Attrition

1 → Attrition

✅ Checkpoint 1 — Baseline ML Pipeline

The first stage focused on building a reliable and reproducible ML pipeline.

Implementations

Dataset loading from CSV

Target encoding (Yes → 1, No → 0)

Feature preprocessing using ColumnTransformer

Train/test split (80/20)

Logistic Regression classifier

Accuracy evaluation (~89%)

Why a Pipeline?

We used a Pipeline to ensure:

Consistent preprocessing during training

Identical transformations during inference

Production-ready architecture

✅ Checkpoint 2 — Code Quality & Experiment Tracking

This stage focused on improving software engineering practices and reproducibility.

🔹 Code Quality

Pre-commit hooks configured for:

Black → automatic formatting

isort → import sorting

Flake8 → linting

Activate pre-commit:

pre-commit install

Run manually:

pre-commit run --all-files
🔹 Unit Testing

Testing tools used:

pytest

pytest-cov

Run tests:

pytest --cov=src

Current test coverage: ≥ 70%

This ensures reliability and maintainability of the codebase.

🔹 MLflow Experiment Tracking

MLflow is integrated to track:

Model parameters

Evaluation metrics

Model artifacts

Experiment runs

Train the Model
python scripts/train_pipeline.py
Launch MLflow UI
mlflow ui

Open in browser:

http://127.0.0.1:5000

Experiment name:

Attrition_Experiments

Each run logs:

Model type

Hyperparameters

Accuracy score

Serialized pipeline model

✅ Checkpoint 3 — Model Serving & Deployment

This stage focuses on production-level serving and containerization.

🔹 Model Export

After training, the full sklearn pipeline is exported as:

model.joblib

This allows:

Decoupling training from inference

Independent model serving

Clean Docker deployment

🔹 FastAPI REST API

The trained model is served using FastAPI.

Available Endpoints
1️⃣ Health Check
GET /health

Response:

{
  "status": "healthy"
}
2️⃣ Prediction Endpoint
POST /predict

Validated using Pydantic schema

Applies full preprocessing pipeline

Returns binary prediction

Example Request:

{
  "Age": 35,
  "BusinessTravel": "Travel_Rarely",
  "DailyRate": 800,
  "Department": "Sales",
  "DistanceFromHome": 10,
  "Education": 3,
  "EducationField": "Life Sciences",
  "EnvironmentSatisfaction": 3,
  "Gender": "Male",
  "HourlyRate": 60,
  "JobInvolvement": 3,
  "JobLevel": 2,
  "JobRole": "Sales Executive",
  "JobSatisfaction": 4,
  "MaritalStatus": "Single",
  "MonthlyIncome": 5000,
  "MonthlyRate": 15000,
  "NumCompaniesWorked": 2,
  "OverTime": "Yes",
  "PercentSalaryHike": 15,
  "PerformanceRating": 3,
  "RelationshipSatisfaction": 3,
  "StockOptionLevel": 1,
  "TotalWorkingYears": 10,
  "TrainingTimesLastYear": 2,
  "WorkLifeBalance": 3,
  "YearsAtCompany": 5,
  "YearsInCurrentRole": 3,
  "YearsSinceLastPromotion": 1,
  "YearsWithCurrManager": 2
}

Example Response:

{
  "prediction": 0
}
🔹 Logging & Monitoring

The API includes:

Request logging

Prediction logging

Error handling

Response time measurement

🔹 Docker Containerization

The application is containerized using Python 3.11-slim.

Build Docker Image
docker build -t attrition-api .
Run Docker Container
docker run -p 8000:8000 attrition-api

Access Swagger UI:

http://localhost:8000/docs
📂 Project Structure
employee-attrition-mlops/
│
├── data/raw/employee_attrition.csv
├── src/attrition/
│   ├── api/
│   │   ├── main.py
│   │   ├── model_loader.py
│   │   └── schema.py
│   └── models/
│       └── train.py
│
├── scripts/train_pipeline.py
├── tests/
├── model.joblib
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
🔁 Reproducibility

Environment management:

uv sync

Dependencies tracked in:

pyproject.toml

uv.lock

Python version:

>= 3.11
👥 Team Collaboration

GitHub-based development

Feature-based commits

Pull request workflow

Conventional commit messages:

feat:

test:

docs:

chore:

refactor:

 Final Outcome

This project demonstrates:

End-to-end ML pipeline development

Experiment tracking with MLflow

API-based model serving

Containerized deployment

Professional MLOps practices

Collaborative team workflow