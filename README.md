# Employee Attrition MLOps Project — Final Report

**MSc Data Science and Computer Science — ESILV**
**Course:** MLOps | **Period:** January 5, 2026 – March 15, 2026

### Team Members
- Abin Jacob  
- Srinivasan Dhakshanamoorthy  
- ANBALAGAN Venkat Subbramani  
- MURALIKRISHNAN Vishnu


![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![MLflow](https://img.shields.io/badge/MLflow-experiments-orange)
![Docker](https://img.shields.io/badge/Docker-container-blue)
![pytest](https://img.shields.io/badge/coverage-70%25-brightgreen)

---

##  Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Definition & Data](#2-problem-definition--data)
3. [Model Performance](#3-model-performance)
4. [System Architecture](#4-system-architecture)
5. [MLOps Practices](#5-mlops-practices)
6. [Monitoring & Reliability](#6-monitoring--reliability)
7. [Team Collaboration](#7-team-collaboration)
8. [Limitations & Future Work](#8-limitations--future-work)
9. [Demo Video](#9-demo-video)

---

## 1. Project Overview

This project implements a **complete end-to-end MLOps pipeline** for predicting employee attrition using machine learning. It was developed as part of the MSc Data Engineering & Artificial Intelligence program at ESILV, simulating a production-ready ML system.

The project was built progressively across four checkpoints, covering everything from a baseline training pipeline to a fully containerized, API-served, and monitored system.

### Key Objectives

- Build a reproducible ML training pipeline
- Track experiments with MLflow
- Serve the trained model via a REST API using FastAPI
- Containerize the full application with Docker
- Apply professional software engineering practices throughout

---

## 2. Problem Definition & Data

### Problem Statement

**Employee attrition** refers to the voluntary departure of employees from an organization. Predicting attrition allows HR departments to take proactive retention measures, reducing recruitment costs and knowledge loss.

This project frames attrition as a **binary classification task**:

| Label | Meaning |
|-------|---------|
| `0` | No Attrition (employee stays) |
| `1` | Attrition (employee leaves) |

### Dataset

- **Source:** IBM HR Analytics Employee Attrition & Performance dataset (via Kaggle)
- **Format:** CSV — `data/raw/employee_attrition.csv`
- **Target column:** `Attrition` (encoded as `Yes → 1`, `No → 0`)
- **Split:** 80% training / 20% test

### Features

The dataset includes 30 employee-level features across demographics, job role, compensation, and satisfaction metrics, including:

- `Age`, `Gender`, `MaritalStatus`
- `Department`, `JobRole`, `JobLevel`
- `MonthlyIncome`, `DailyRate`, `HourlyRate`
- `OverTime`, `BusinessTravel`, `DistanceFromHome`
- `JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`
- `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`

### Preprocessing

A `ColumnTransformer` pipeline handles:

- **Numerical features:** StandardScaler
- **Categorical features:** OneHotEncoder

---

## 3. Model Performance

The baseline Logistic Regression model achieved the following performance on the held-out test set:

| Metric | Score |
|--------|-------|
| Accuracy | ~0.89 |
| Precision | ~0.84 |
| Recall | ~0.77 |
| F1-score | ~0.80 |

These results provide a strong baseline while keeping the model simple and interpretable. All metrics were logged automatically to MLflow for reproducibility and comparison across runs.

---

## 4. System Architecture

### Project Structure

```
employee-attrition-mlops/
│
├── data/raw/employee_attrition.csv       # Raw dataset
├── src/attrition/
│   ├── api/
│   │   ├── main.py                       # FastAPI app
│   │   ├── model_loader.py               # Loads model.joblib
│   │   └── schema.py                     # Pydantic request/response schema
│   └── models/
│       └── train.py                      # Training logic
│
├── scripts/
│   └── train_pipeline.py                 # Entry point for training
├── tests/                                # Unit tests (pytest)
├── model.joblib                          # Serialized sklearn pipeline
├── mlflow.db                             # MLflow tracking database
├── Dockerfile                            # Container definition
├── pyproject.toml                        # Project metadata & dependencies
├── uv.lock                               # Locked dependency versions
└── README.md                             # This file
```

### ML Pipeline

```
Raw CSV Data
    └── Preprocessing (ColumnTransformer)
            ├── StandardScaler      (numerical features)
            └── OneHotEncoder       (categorical features)
                    └── Logistic Regression Classifier
                            └── model.joblib  (exported artifact)
```

This ensures **identical transformations** are applied at both training and inference time.

### Serving Architecture

```
Client (HTTP)
    └── FastAPI (uvicorn)
            ├── POST /predict  →  model_loader.py  →  model.joblib  →  prediction
            └── GET  /health   →  {"status": "healthy"}
```

The model is decoupled from the training step — inference loads `model.joblib` directly, enabling independent deployment.

---

## 5. MLOps Practices

### 5.1 Environment Management

- **Tool:** [uv](https://github.com/astral-sh/uv) — a fast Python package manager
- Dependencies declared in `pyproject.toml`
- Exact versions locked in `uv.lock` for full reproducibility
- Python version pinned: `>=3.11, <3.12`

### 5.2 Code Quality

Pre-commit hooks are configured in `.pre-commit-config.yaml` and run automatically on every commit:

| Hook | Purpose |
|------|---------|
| `black` | Automatic code formatting |
| `isort` | Import sorting |
| `flake8` | Linting (style & errors) |

**Install hooks:**
```bash
pre-commit install
```

**Run manually:**
```bash
pre-commit run --all-files
```

### 5.3 Unit Testing

- **Framework:** `pytest` with `pytest-cov`
- **Coverage:** ≥ 70%
- Tests are located in the `tests/` directory and cover preprocessing, model loading, and API endpoints.

**Run tests with coverage:**
```bash
pytest --cov=src
```

### 5.4 Experiment Tracking with MLflow

MLflow is integrated to provide full experiment reproducibility. Each training run logs:

| Logged Item | Description |
|-------------|-------------|
| Model type | e.g., `LogisticRegression` |
| Hyperparameters | e.g., `max_iter`, `C` |
| Accuracy score | Test set accuracy (~89%) |
| Model artifact | Full sklearn pipeline |

**Train the model:**
```bash
python scripts/train_pipeline.py
```

**Launch MLflow UI:**
```bash
mlflow ui
```
Access at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

- **Experiment name:** `Attrition_Experiments`
- **Backend store:** `mlflow.db` (SQLite)

### 5.5 Model Serving (FastAPI)

The trained model is served as a REST API with the following endpoints:

#### `GET /health`

```json
{ "status": "healthy" }
```

#### `POST /predict`

**Example request:**
```json
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
```

**Example response:**
```json
{ "prediction": 0 }
```

The API uses **Pydantic** for request validation, ensuring type safety and clear error messages.

**Start the API locally:**
```bash
uvicorn src.attrition.api.main:app --reload
```

Access Swagger UI at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5.6 Docker Containerization

The application is fully containerized using a `python:3.11-slim` base image.

**Build the image:**
```bash
docker build -t attrition-api .
```

**Run the container:**
```bash
docker run -p 8000:8000 attrition-api
```

Access Swagger UI at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 6. Monitoring & Reliability

### Request & Prediction Logging

The FastAPI application includes structured logging for every request and response:

- **Incoming request:** method, path, input payload
- **Prediction result:** model output, timestamp
- **Errors:** exception type, stack trace
- **Response time:** measured per request for latency monitoring

### Health Check Endpoint

A dedicated `/health` endpoint allows uptime monitoring tools (e.g., Docker healthcheck, load balancers) to verify that the service is alive.

### Error Handling

All prediction errors are caught and returned with appropriate HTTP status codes and descriptive messages, preventing silent failures in production.

### Model Reliability

- The model is loaded **once at startup** (not per-request) to minimize latency.
- The `model.joblib` artifact packages the full preprocessing + classifier pipeline, eliminating any risk of preprocessing drift between training and inference.

### Monitoring Strategy

| Layer | Monitoring Approach |
|-------|---------------------|
| API availability | `/health` endpoint |
| Request volume | Request logging per endpoint |
| Prediction distribution | Prediction logging (0 vs 1 ratio) |
| Errors | Exception logging with HTTP 500 responses |
| Response time | Per-request timing logs |

---

## 7. Team Collaboration

### Git & GitHub Workflow

- Feature-based branching strategy
- Pull Requests required for merging into `main`
- Each PR reviewed and approved by at least one other team member
- Commit history reflects individual contributions

### Commit Convention

| Prefix | Purpose |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `test:` | Adding or updating tests |
| `docs:` | Documentation changes |
| `chore:` | Maintenance tasks |
| `refactor:` | Code restructuring |

### Checkpoint Delivery

| Checkpoint | Due Date | Status |
|------------|----------|--------|
| CP1 — Project Setup & Foundations | Feb 1, 2026 | ✅ Completed |
| CP2 — Code Quality & Experiment Tracking | Feb 15, 2026 | ✅ Completed |
| CP3 — Model Serving & Containerization | Mar 1, 2026 | ✅ Completed |
| CP4 — Monitoring, Polish & Final Report | Mar 15, 2026 | ✅ Completed |

---

## 8. Limitations & Future Work

### Current Limitations

| Area | Limitation |
|------|-----------|
| **Model** | Only Logistic Regression was evaluated; no hyperparameter search |
| **Data** | Static CSV dataset with no automated ingestion or updates |
| **Monitoring** | Logging is stdout-based only; no dashboarding or alerting |
| **CI/CD** | No automated GitHub Actions pipeline for testing or deployment |
| **Model versioning** | MLflow artifacts are local; no model registry promotion workflow |
| **Class imbalance** | Dataset is imbalanced (~84% no-attrition); no oversampling applied |

### Future Work

- **Model improvement:** Experiment with Random Forest, XGBoost, and ensemble methods; address class imbalance with SMOTE or class weighting
- **CI/CD pipeline:** Add GitHub Actions workflow to run tests, lint, and build Docker image on every push
- **Model registry:** Use MLflow Model Registry to promote models from `Staging → Production`
- **Automated retraining:** Trigger retraining when data drift is detected
- **Monitoring dashboard:** Integrate Prometheus + Grafana for real-time metrics visualization
- **Data pipeline:** Replace manual CSV with automated ingestion from a database or data warehouse
- **Authentication:** Add API key or OAuth authentication to the FastAPI service for production use

---

## 9. Demo Video

Link : https://youtu.be/PqZBywHYHJI

The demo covers:
- CI pipeline, Docker build, and app startup
- FastAPI endpoint walkthrough (`/health` and `/predict`)
- Example request and response via Swagger UI

---

## 🔁 Reproducibility

```bash
# 1. Install dependencies
uv sync

# 2. Train the model
python scripts/train_pipeline.py

# 3. Launch MLflow UI
mlflow ui

# 4. Start the API
uvicorn src.attrition.api.main:app --reload

# 5. Run tests
pytest --cov=src

# 6. Build & run with Docker
docker build -t attrition-api .
docker run -p 8000:8000 attrition-api
```

---
