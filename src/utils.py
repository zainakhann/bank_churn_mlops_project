import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from dotenv import load_dotenv
import yaml


# ----------------------------
# Load environment variables / secrets
# ----------------------------
load_dotenv(dotenv_path="secrets/.env")


# ----------------------------
# Ensure required directories exist
# ----------------------------
os.makedirs("logs", exist_ok=True)


# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(
    filename='logs/churn_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


# ----------------------------
# Load config.yaml safely
# ----------------------------
try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logging.info("Config loaded successfully")
except FileNotFoundError:
    logging.error("config/config.yaml not found")
    raise


# ----------------------------
# Config variables
# ----------------------------
DATA_PATH = config["data"]["raw_path"]
TARGET_COL = config["data"]["target_column"]
TEST_SIZE = config["data"]["test_size"]
RANDOM_STATE = config["project"]["random_state"]

PIPELINE_PATH = config["paths"]["pipeline_path"]


# ----------------------------
# Feature definitions (fallback safe)
# ----------------------------
DEFAULT_NUMERIC = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "estimated_salary"
]

DEFAULT_CATEGORICAL = [
    "country",
    "gender",
    "credit_card",
    "active_member"
]

FEATURES_NUMERIC = config["data"].get("numeric_features") or DEFAULT_NUMERIC
FEATURES_CATEGORICAL = config["data"].get("categorical_features") or DEFAULT_CATEGORICAL


# Remove target leakage if present
if TARGET_COL in FEATURES_NUMERIC:
    FEATURES_NUMERIC.remove(TARGET_COL)

if TARGET_COL in FEATURES_CATEGORICAL:
    FEATURES_CATEGORICAL.remove(TARGET_COL)


logging.info(f"Numeric features: {FEATURES_NUMERIC}")
logging.info(f"Categorical features: {FEATURES_CATEGORICAL}")


# ----------------------------
# Data Utilities
# ----------------------------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load CSV data and handle missing values safely.
    """
    df = pd.read_csv(path)
    logging.info(f"Data loaded from {path}, shape: {df.shape}")

    # Drop ID column
    if "customer_id" in df.columns:
        df.drop("customer_id", axis=1, inplace=True)

    # Handle numeric columns
    for col in FEATURES_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Handle categorical columns
    for col in FEATURES_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Ensure target is integer (0/1)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    logging.info("Missing values handled + preprocessing applied")

    return df


# ----------------------------
# Split Data
# ----------------------------
def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
):
    """
    Split dataset into train/test sets.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    X = df.drop(columns=[target])
    y = df[target]

    logging.info(f"Splitting data: test_size={test_size}, random_state={random_state}")

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ----------------------------
# Save pipeline
# ----------------------------
def save_pipeline(pipeline, path: str = PIPELINE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logging.info(f"Pipeline saved at {path}")


# ----------------------------
# Load pipeline
# ----------------------------
def load_pipeline(path: str = PIPELINE_PATH):
    if not os.path.exists(path):
        logging.error(f"Pipeline not found at {path}")
        raise FileNotFoundError(f"Pipeline not found at {path}")

    logging.info(f"Pipeline loaded from {path}")
    return joblib.load(path)


# ----------------------------
# Self-test (safe)
# ----------------------------
if __name__ == "__main__":
    print("Running utils test...")

    try:
        df = load_data()
        print("✅ Data loaded successfully")
        print(df.head())

        X_train, X_test, y_train, y_test = split_data(df)
        print("✅ Data split successful")
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    except Exception as e:
        print("❌ Error during utils test:", str(e))