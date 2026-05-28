# FraudLens

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PaySim-20B2AA?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download)
[![CI](https://github.com/PaulOkogwu/fraud-detection-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/PaulOkogwu/fraud-detection-dashboard/actions/workflows/ci.yml)

## 1. Overview

FraudLens is a machine learning fraud detection dashboard and CLI toolkit built for portfolio demonstration and technical exploration. It supports transaction data ingestion, feature engineering, model training/inference, comparative evaluation, and explainability reporting in a single codebase.

## 2. Key Features

- Streamlit dashboard for fraud analytics and model insights
- CLI workflow for model training and sample prediction
- Model comparison on a shared train/test split
- Explainability via tree-model feature importance (with optional SHAP)
- Test suite with GitHub Actions CI validation
- Dockerized dashboard deployment path

## 3. Tech Stack

- Python 3.11
- pandas, NumPy
- scikit-learn
- XGBoost (optional in comparison flow if unavailable)
- matplotlib, seaborn
- Streamlit
- pytest + GitHub Actions
- Docker

## 4. Machine Learning Pipeline

1. Load transaction dataset
2. Clean and normalize schema (`src/data_loader.py`)
3. Engineer behavioral/time/type features (`src/features.py`)
4. Train classifier (`src/model.py`)
5. Evaluate and inspect outputs in CLI/dashboard

## 5. Dashboard Features

- Dataset overview (rows, columns, fraud/non-fraud counts)
- Fraud analytics visualizations:
  - Fraud vs non-fraud distribution
  - Transaction type breakdown
  - Amount distribution
  - Fraud rate by transaction type
- Sample prediction workflow with confusion matrix display

Run dashboard:

```bash
streamlit run app.py
```

## 6. Model Comparison

The dashboard includes a comparison workflow for:

- Logistic Regression
- Random Forest
- XGBoost (if available)

Metrics reported per model:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Reusable artifacts are saved to:

- `reports/model_comparison_summary.csv`
- `reports/model_comparison_details.json`

## 7. Explainability

FraudLens provides feature-importance-based explainability for tree-based models as the stable default. The dashboard shows top contributing features in both table and chart form.

Optional SHAP explanations are available when SHAP can be imported cleanly; otherwise the dashboard continues with built-in feature importance without failing.

## 8. Setup Instructions

```bash
git clone https://github.com/PaulOkogwu/fraud-detection-dashboard.git
cd fraud-detection-dashboard

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CLI usage:

```bash
# Train
python main.py --mode train --model_type rf

# Predict
python main.py --mode predict --model_path models/fraud_model.pkl
```

## 9. Docker Instructions

Build image:

```bash
docker build -t fraudlens-dashboard .
```

Run container:

```bash
docker run --rm -p 8501:8501 fraudlens-dashboard
```

Open: `http://localhost:8501`

## 10. Testing / CI

Run tests locally:

```bash
pytest tests/
```

CI runs on push and pull request to `main` via:

- `.github/workflows/ci.yml`

## 11. Dataset Note

PaySim is a synthetic financial transaction dataset. Reported model metrics on this dataset can be high because synthetic patterns are cleaner than real banking environments. These results should not be treated as proof of production readiness for noisy, adversarial, real-world fraud systems.

This repository includes `data/demo_transactions.csv` as a lightweight demo file only, intended for dashboard screenshots and quick local testing.

## 12. Attribution

This project is adapted from ARPAHLS/cfd and remains licensed under the MIT License.

## 13. License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
