"""
Tests for data leakage in feature engineering.
"""
import numpy as np
import pandas as pd
import pytest
from src.features import feature_engineering
from src.data_loader import clean_data

@pytest.fixture(name="clean_sample_data")
def fixture_clean_sample_data():
    """
    Fixture that creates and cleans a deterministic synthetic sample.
    This avoids external file dependencies while preserving a realistic schema.
    """
    rng = np.random.default_rng(42)
    n_rows = 600

    old_balance_orig = rng.uniform(1_000, 100_000, size=n_rows)
    amount = rng.uniform(10, 5_000, size=n_rows)
    old_balance_dest = rng.uniform(1_000, 120_000, size=n_rows)

    # Create realistic balances with small noise so engineered error features vary.
    new_balance_orig = old_balance_orig - amount + rng.normal(0, 120, size=n_rows)
    new_balance_dest = old_balance_dest + amount + rng.normal(0, 120, size=n_rows)

    # Build a target with signal + noise to avoid perfect-predictor artifacts.
    fraud_score = (
        (amount > 3_000).astype(int)
        + (np.abs(new_balance_orig - (old_balance_orig - amount)) > 160).astype(int)
        + rng.integers(0, 2, size=n_rows)
    )
    is_fraud = (fraud_score >= 2).astype(int)

    data = pd.DataFrame(
        {
            "step": rng.integers(1, 744, size=n_rows),
            "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"], size=n_rows),
            "amount": amount,
            "nameOrig": [f"C{i:08d}" for i in range(n_rows)],
            "oldbalanceOrg": old_balance_orig,
            "newbalanceOrig": new_balance_orig,
            "nameDest": [f"M{i:08d}" for i in range(n_rows)],
            "oldbalanceDest": old_balance_dest,
            "newbalanceDest": new_balance_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": rng.integers(0, 2, size=n_rows),
        }
    )

    return clean_data(data)

def test_no_single_feature_leakage(clean_sample_data):
    """
    Ensures that no single feature has a 1.0 correlation with the target.
    A correlation of 1.0 (or -1.0) often indicates data leakage (e.g. label included in features).
    """
    # Create features
    df_features = feature_engineering(clean_sample_data.copy())

    # Calculate correlations
    if 'isFraud' not in df_features.columns:
        pytest.skip("Target column 'isFraud' missing from data")

    correlations = df_features.corr()['isFraud'].drop('isFraud')

    # Check if any feature has perfect correlation
    # We use 0.99 as a threshold to be safe, though 1.0 is the strict definition of a leak
    leaks = correlations[abs(correlations) >= 0.99]

    assert len(leaks) == 0, f"Potential data leakage detected in features: {leaks.index.tolist()}"

def test_error_balance_not_perfect_predictor(clean_sample_data):
    """
    Specific check for 'errorBalanceOrig' which was suspected of leakage.
    """
    df_features = feature_engineering(clean_sample_data.copy())

    # It shouldn't be perfectly correlated
    corr = df_features['errorBalanceOrig'].corr(df_features['isFraud'])
    assert abs(corr) < 0.99, f"errorBalanceOrig is too highly correlated: {corr}"
