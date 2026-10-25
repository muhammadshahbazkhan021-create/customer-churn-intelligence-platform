import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Internal modules import
from churn_project.data.clean_data import clean_customer_data
from churn_project.features.build_features import create_domain_features

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global memory caches for model artifacts (Singleton pattern implementation)
_model_cache: Dict[str, Any] = {}
_preprocessor_cache: Dict[str, Any] = {}

def load_artifacts(model_path: str, preprocessor_path: str) -> Tuple[Any, Any]:
    """
    Loads model and preprocessor artifacts with memory caching to prevent redundant disk I/O.
    """
    global _model_cache, _preprocessor_cache

    if model_path not in _model_cache:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")
        logger.info(f"Loading model into memory cache from: {model_path}")
        _model_cache[model_path] = joblib.load(model_path)

    if preprocessor_path not in _preprocessor_cache:
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at: {preprocessor_path}")
        logger.info(f"Loading preprocessor into memory cache from: {preprocessor_path}")
        _preprocessor_cache[preprocessor_path] = joblib.load(preprocessor_path)

    return _model_cache[model_path], _preprocessor_cache[preprocessor_path]

def assign_risk_segment(probability: float) -> str:
    """
    Maps continuous classification probability directly to business risk segments.
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

def predict_churn(
    input_data: pd.DataFrame,
    model_path: str = "models/churn_xgb_model.joblib",
    preprocessor_path: str = "models/preprocessor.joblib",
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Executes raw inference tracking, transforms incoming structures, 
    and appends risk scores with multi-stage labels.
    """
    logger.info(f"Received inference request for {len(input_data)} records.")
    
    if input_data.empty:
        raise ValueError("Inference input DataFrame is empty.")

    # 1. Load optimized artifacts via Global Cache Layer
    model, preprocessor = load_artifacts(model_path, preprocessor_path)

    # 2. Raw Input Minimum Validation Check
    required_minimum_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    missing_cols = [col for col in required_minimum_features if col not in input_data.columns]
    if missing_cols:
        error_msg = f"Inference structural validation failed. Missing tracking metrics: {missing_cols}"
        logger.error(error_msg)
        raise KeyError(error_msg)

    # 3. Clean Data Sequence Execution (Inference Mode)
    cleaned_df = clean_customer_data(input_data, is_training=False)

    # 4. Generate Domain Engineering Matrix Extensions
    featured_df = create_domain_features(cleaned_df)

    # 5. Transform Matrix Features via Fitted Pipeline Artifact
    X_processed = preprocessor.transform(featured_df)

    # 6. Apply XGBoost Classification Inference Engine
    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # 7. Compile Final Output Package Frame
    output_df = input_data.copy()
    output_df['churn_probability'] = np.round(probabilities, 4)
    output_df['churn_prediction'] = predictions
    output_df['risk_segment'] = output_df['churn_probability'].apply(assign_risk_segment)

    logger.info("Inference completed successfully. Returning metadata payload mapping.")
    return output_df[['churn_probability', 'churn_prediction', 'risk_segment']]



if __name__ == "__main__":
    # Mocking single transaction data packet mimicking external customer profile stream
    dummy_customer_payload = {
        'customerID': '9999-ABCD',
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.25,
        'TotalCharges': '170.50'  # Passed as string format to check conversion framework
    }

    # Wrap inside standard dataframe index map
    df_test = pd.DataFrame([dummy_customer_payload])

    print("--- Executing Local Inference Pipeline Verification Integration ---")
    try:
        # Running simulation against default project locations
        results = predict_churn(
            input_data=df_test,
            model_path="models/churn_xgb_model.joblib",
            preprocessor_path="models/preprocessor.joblib"
        )
        print("\n=== Result Metrics Output ===")
        print(results.to_markdown(index=False))
        
    except FileNotFoundError:
        print("\n[Notice] Saved artifacts models binaries are currently missing from disk storage paths.")
        print("Please finalize running the model training pipeline stage using: python -m churn_project.models.train")
