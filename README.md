# FraudLens

FraudLens is an adapted, MIT-licensed fraud detection project for portfolio and educational use. It keeps the original core pipeline and model workflow while presenting a cleaner project identity for independent use.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PaySim-20B2AA?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download)

## Overview

**FraudLens** is a modular fraud detection system designed to identify suspicious financial transactions. Built on the PaySim dataset, it supports training and prediction through a straightforward CLI pipeline.

## Key Features

- Modular architecture for data ingestion, feature engineering, modeling, and evaluation
- Random Forest and Gradient Boosting model support
- JSON-based audit logging for traceability
- CLI-based training and prediction workflow
- Unit, integration, and leakage-focused tests

## Model Performance

The current release includes a Random Forest classifier trained on PaySim data.

| Metric | Score | Notes |
| :--- | :--- | :--- |
| ROC-AUC | 0.999 | Strong discrimination on synthetic data |
| Precision | 1.00 | No false positives in the referenced test run |
| Recall | 1.00 | Full fraud capture in the referenced test run |

> Performance on PaySim is based on synthetic transaction behavior and may not directly match real-world production environments.

## Project Structure

```text
fraud-detection-dashboard/
  docs/
  models/
  src/
  tests/
  main.py
  requirements.txt
```

## Quick Start

### 1. Installation

```bash
# Clone your adapted repository
git clone https://github.com/PaulOkogwu/fraud-detection-dashboard.git
cd fraud-detection-dashboard

# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Usage

Train:

```bash
python main.py --mode train --model_type rf
```

Predict:

```bash
python main.py --mode predict --model_path models/fraud_model.pkl
```

Run tests:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Attribution

This project is adapted from ARPAHLS/cfd and remains licensed under the MIT License.
