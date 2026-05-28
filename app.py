"""
Streamlit dashboard for FraudLens.
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import clean_data, load_data
from src.features import feature_engineering
from src.model import FraudDetector


st.set_page_config(page_title="FraudLens Dashboard", layout="wide")
st.title("FraudLens Dashboard")
st.caption("Dataset overview and sample fraud predictions")


def _load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    text = uploaded_file.getvalue().decode("utf-8")
    return pd.read_csv(StringIO(text))


st.sidebar.header("Data Source")
source_mode = st.sidebar.radio(
    "Choose dataset source",
    ("Upload CSV", "Load from path"),
)

raw_df = None

if source_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded is not None:
        raw_df = _load_uploaded_csv(uploaded)
else:
    default_path = "data/PS_20174392719_1491204439457_log.csv"
    dataset_path = st.sidebar.text_input("Dataset path", value=default_path)
    if dataset_path:
        path_obj = Path(dataset_path)
        if path_obj.exists():
            raw_df = load_data(str(path_obj))
        else:
            st.warning(f"Dataset file not found: {dataset_path}")

if raw_df is None:
    st.info("Provide a dataset to begin.")
    st.stop()

st.subheader("Dataset Overview")
clean_df = clean_data(raw_df.copy())

row_count, col_count = clean_df.shape
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{row_count:,}")
col2.metric("Columns", f"{col_count:,}")

if "isFraud" in clean_df.columns:
    fraud_count = int((clean_df["isFraud"] == 1).sum())
    non_fraud_count = int((clean_df["isFraud"] == 0).sum())
    col3.metric("Fraud Transactions", f"{fraud_count:,}")
    col4.metric("Non-Fraud Transactions", f"{non_fraud_count:,}")
else:
    col3.metric("Fraud Transactions", "N/A")
    col4.metric("Non-Fraud Transactions", "N/A")
    st.warning("Column 'isFraud' not found; fraud counts and evaluation are limited.")

if "type" in clean_df.columns:
    st.markdown("**Transaction Type Breakdown**")
    st.dataframe(
        clean_df["type"].value_counts(dropna=False).rename_axis("type").reset_index(name="count"),
        use_container_width=True,
    )
else:
    st.warning("Column 'type' not found; cannot display transaction type breakdown.")

with st.expander("Preview dataset"):
    st.dataframe(clean_df.head(25), use_container_width=True)

st.subheader("Sample Predictions")
model_path = st.text_input("Model path", value="models/fraud_model.pkl")
sample_size = st.slider("Number of rows to score", min_value=1, max_value=200, value=10)

if st.button("Run Predictions"):
    if "isFraud" not in clean_df.columns:
        st.error("Column 'isFraud' is required for sample prediction comparison.")
        st.stop()

    try:
        detector = FraudDetector()
        detector.load_model(model_path)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Unable to load model: {exc}")
        st.stop()

    try:
        engineered = feature_engineering(clean_df.copy())
        sample = engineered.sample(min(sample_size, len(engineered)), random_state=42)
        y_true = sample["isFraud"]
        X_sample = sample.drop(columns=["isFraud"])

        preds = detector.predict(X_sample)
        probs = detector.predict_proba(X_sample)[:, 1]

        result_df = X_sample.copy()
        result_df["actual_isFraud"] = y_true.values
        result_df["predicted_isFraud"] = preds
        result_df["predicted_fraud_probability"] = probs

        st.success("Predictions complete.")
        st.dataframe(result_df, use_container_width=True)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Prediction failed: {exc}")
