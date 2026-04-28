import os
import joblib
import pandas as pd
import logging
from pipelines.feature_pipeline import feature_engineering

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "models"


# ----------------------------
# Load latest PIPELINE
# ----------------------------
def load_latest_pipeline():
    try:
        files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".pkl") and not f.startswith("metadata")]

        if not files:
            raise FileNotFoundError("No trained pipeline found")

        latest_file = sorted(files)[-1]
        pipeline_path = os.path.join(MODEL_PATH, latest_file)

        logging.info(f"Loading pipeline: {pipeline_path}")

        pipeline = joblib.load(pipeline_path)

        logging.info("Pipeline loaded successfully")

        return pipeline

    except Exception as e:
        logging.error(f"Error loading pipeline: {e}")
        raise


# ----------------------------
# Validate RAW input (BANK DATASET)
# ----------------------------
def validate_input(data: pd.DataFrame):

    required_cols = [
        "credit_score",
        "country",
        "gender",
        "age",
        "tenure",
        "balance",
        "products_number",
        "credit_card",
        "active_member",
        "estimated_salary"
    ]

    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        logging.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    logging.info("Input validated successfully")

    return data[required_cols]


# ----------------------------
# Predict function
# ----------------------------
def predict(data: pd.DataFrame):

    try:
        # Step 1: load pipeline
        logging.info("Loading pipeline...")
        pipeline = load_latest_pipeline()

        # Step 2: validate RAW input
        logging.info("Validating input data...")
        data = validate_input(data)

        # Step 3: feature engineering
        logging.info("Performing feature engineering...")
        data = feature_engineering(data)

        logging.debug(f"Feature engineering completed:\n{data.head()}")

        # Step 4: prediction
        logging.info("Making predictions...")
        preds = pipeline.predict(data)

        logging.info(f"Predictions generated: {preds}")

        return preds

    except Exception as e:
        logging.exception("Prediction failed")
        raise e


# ----------------------------
# Example run (for testing)
# ----------------------------
if __name__ == "__main__":

    sample = pd.DataFrame([{
        "credit_score": 619,
        "country": "France",
        "gender": "Female",
        "age": 42,
        "tenure": 2,
        "balance": 0.0,
        "products_number": 1,
        "credit_card": 1,
        "active_member": 1,
        "estimated_salary": 101348.88
    }])

    try:
        result = predict(sample)
        print("Prediction:", result)

    except Exception as e:
        print(f"Error: {str(e)}")