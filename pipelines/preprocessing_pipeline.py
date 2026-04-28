import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import yaml


# ----------------------------
# Load config
# ----------------------------
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


def build_preprocessor(df):
    """
    Build preprocessing pipeline using config-defined features
    (fallback to auto-detection if not provided)
    """

    if df is None:
        raise ValueError("DataFrame cannot be None")

    target = config["data"]["target_column"]  # "churn"

    # ----------------------------
    # Prefer config features
    # ----------------------------
    numeric_features = config["data"].get("numeric_features")
    categorical_features = config["data"].get("categorical_features")

    # ----------------------------
    # Fallback to correct bank dataset structure
    # ----------------------------
    if not numeric_features:
        numeric_features = [
            "credit_score",
            "age",
            "tenure",
            "balance",
            "products_number",
            "estimated_salary"
        ]

    if not categorical_features:
        categorical_features = [
            "country",
            "gender",
            "credit_card",
            "active_member"
        ]

    # ----------------------------
    # Remove target if accidentally included
    # ----------------------------
    if target in numeric_features:
        numeric_features.remove(target)

    if target in categorical_features:
        categorical_features.remove(target)

    # ----------------------------
    # Remove ID column
    # ----------------------------
    if "customer_id" in categorical_features:
        categorical_features.remove("customer_id")

    if "customer_id" in numeric_features:
        numeric_features.remove("customer_id")

    print(f"Preprocessor created | Numeric: {numeric_features} | Categorical: {categorical_features}")

    # ----------------------------
    # Pipelines
    # ----------------------------
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ----------------------------
    # Column Transformer
    # ----------------------------
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


# ----------------------------
# Test block
# ----------------------------
if __name__ == "__main__":
    import pandas as pd

    print("Running preprocessing_pipeline test...")

    df = pd.read_csv("data/churn.csv").head(5)

    preprocessor = build_preprocessor(df)

    print("\nColumnTransformer created successfully!")
    print(preprocessor)