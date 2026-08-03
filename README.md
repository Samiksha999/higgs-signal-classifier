![Tests](https://github.com/Samiksha999/higgs-signal-classifier/actions/workflows/ci.yml/badge.svg)

# Higgs Boson Signal Classifier
...

A machine learning pipeline that classifies Higgs boson signal events from background noise using ATLAS detector data from CERN's Open Data Portal.

## Project Overview
This project builds and deploys a classifier distinguishing Higgs boson signal from background events, with SHAP-based explainability and a REST API endpoint.

## Why This Problem is Hard
- **Class imbalance** — signal events are rare compared to background
- **28 physics features** — mix of primitive (PRI_) and derived (DER_) detector measurements
- **Undefined values** — some features return -999.0 for undefined measurements

## Models
| Model | AUC-ROC |
|---|---|
| Logistic Regression (baseline) | 0.5067 |
| Random Forest | 0.4981 |
| XGBoost | 0.4928 |

## Tech Stack
Python · XGBoost · scikit-learn · SHAP · FastAPI · Docker · pytest · GitHub Actions

## Project Structure
higgs-signal-classifier/
├── src/
│ ├── eda.py
│ ├── features.py
│ ├── train.py
│ ├── explainability.py
│ └── api.py
├── tests/
│ └── test_api.py
├── .github/workflows/ci.yml
├── Dockerfile
└── requirements.txt

## Run Locally
```bash
pip install -r requirements.txt
python -m uvicorn src.api:app --reload
```

## Run Tests
```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## Run with Docker
```bash
docker build -t higgs-classifier .
docker run -p 8000:8000 higgs-classifier
```

## Connection to AI Alignment
Interpretable ML via SHAP directly relates to AI transparency — understanding which features drive predictions mirrors the challenge of understanding what drives AI decisions. This project demonstrates that explainability is achievable even in complex physics classification tasks.

## Dataset
ATLAS Higgs Boson Machine Learning Challenge — CERN Open Data Portal
