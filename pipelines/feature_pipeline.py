import pandas as pd
import numpy as np
import logging

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------
# SAFE QCUT (robust binning)
# ----------------------------
def safe_qcut(series, q, labels):
    """
    Safe version of qcut:
    - Handles small datasets (inference-safe)
    - Prevents bin/label mismatch
    """
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except Exception:
        bins = min(len(series.unique()), q)
        if bins < 2:
            return pd.Series([labels[0]] * len(series), index=series.index)
        return pd.cut(series, bins=bins, labels=labels[:bins])


# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting feature engineering...")

    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")

    df = df.copy()

    # ----------------------------
    # Convert numeric columns
    # ----------------------------
    numeric_cols = [
        "credit_score",
        "age",
        "tenure",
        "balance",
        "products_number",
        "estimated_salary"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ----------------------------
    # BASIC CLEANING
    # ----------------------------
    categorical_defaults = {
        "country": "France",
        "gender": "Male"
    }

    for col, default in categorical_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # ----------------------------
    # FEATURE CREATION
    # ----------------------------

    # Balance intensity (customer financial exposure)
    if "balance" in df.columns and "credit_score" in df.columns:
        df["BalanceToScoreRatio"] = np.where(
            df["credit_score"] == 0,
            0,
            df["balance"] / df["credit_score"]
        )

    # Income efficiency (salary vs balance)
    if "estimated_salary" in df.columns and "balance" in df.columns:
        df["SalaryToBalanceRatio"] = np.where(
            df["balance"] == 0,
            0,
            df["estimated_salary"] / (df["balance"] + 1)
        )

    # Activity score (behavior proxy)
    if "active_member" in df.columns and "products_number" in df.columns:
        df["ActivityScore"] = df["active_member"] * df["products_number"]

    # ----------------------------
    # CATEGORICAL FEATURES (segmentation)
    # ----------------------------

    if "age" in df.columns:
        df["AgeGroup"] = safe_qcut(
            df["age"],
            q=3,
            labels=["Young", "MidAge", "Senior"]
        )

    if "balance" in df.columns:
        df["BalanceGroup"] = safe_qcut(
            df["balance"],
            q=3,
            labels=["LowBalance", "MidBalance", "HighBalance"]
        )

    if "credit_score" in df.columns:
        df["CreditScoreGroup"] = safe_qcut(
            df["credit_score"],
            q=3,
            labels=["Poor", "Good", "Excellent"]
        )

    # ----------------------------
    # FINAL LOG
    # ----------------------------
    logging.info(
        "Feature engineering completed. Added features: "
        "['BalanceToScoreRatio', 'SalaryToBalanceRatio', 'ActivityScore', "
        "'AgeGroup', 'BalanceGroup', 'CreditScoreGroup']"
    )

    logging.info(f"New DataFrame shape: {df.shape}")

    return df


# ----------------------------
# SAVE FEATURES (optional utility)
# ----------------------------
def save_features(df: pd.DataFrame, path: str):
    import joblib
    joblib.dump(df, path)
    logging.info(f"Features saved at {path}")


# ----------------------------
# TEST RUN
# ----------------------------
if __name__ == "__main__":
    logging.info("Running feature_engineering test...")

    try:
        df_test = pd.read_csv("data/churn.csv")
        df_fe = feature_engineering(df_test)
        logging.info(df_fe.head())
        logging.info("Feature engineering test completed successfully!")
    except Exception as e:
        logging.error(f"Error during feature engineering test: {e}")