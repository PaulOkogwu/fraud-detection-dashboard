# Testing Strategy & Guide

This document outlines the testing framework for **FraudLens**. The project uses a dual-layer strategy to validate both software behavior and model behavior.

## 1. Unit Tests (`tests/test_components.py`)

Purpose: verify individual components in isolation.

Coverage:
- Data cleaning and schema normalization
- Feature engineering transformations
- Model wrapper initialization, training, and prediction on synthetic inputs

Run:

```bash
pytest tests/test_components.py
```

## 2. Integration Tests (`tests/test_integration.py`)

Purpose: verify end-to-end behavior with the saved model and representative data.

Coverage:
- Pipeline execution from load -> clean -> feature engineering -> prediction
- Probability output validity (0.0 to 1.0)
- Serialized model loading from `models/fraud_model.pkl`

Run:

```bash
pytest tests/test_integration.py
```

## 3. Leakage Tests (`tests/test_leakage.py`)

Purpose: ensure the model is not relying on target-leaking or perfect-proxy features.

Coverage:
- Correlation checks against the target variable
- Verification that engineered features are not perfect predictors

Run:

```bash
pytest tests/test_leakage.py
```

## Running the Full Suite

```bash
pytest tests/
```
