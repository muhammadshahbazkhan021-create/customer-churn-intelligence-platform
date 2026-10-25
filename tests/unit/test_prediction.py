import os
import pytest
import pandas as pd

from churn_project.models.predict import assign_risk_segment, predict_churn

MODEL_PATH = "artifacts/models/best_model.joblib"
PREPROCESSOR_PATH = "artifacts/preprocessors/preprocessor.joblib"


@pytest.mark.parametrize("prob, expected_tier", [
    (0.15, "Low"),
    (0.29, "Low"),
    (0.30, "Medium"),
    (0.55, "Medium"),
    (0.70, "High"),      # Function assigns >= 0.7 to High
    (0.71, "High"),
    (0.95, "High")
])
def test_risk_segment_mapping_thresholds(prob, expected_tier):
    """Test 1: Boundary condition validations for parametric risk segments."""
    assert assign_risk_segment(prob) == expected_tier


def test_inference_pipeline_outputs():
    """Test 2: Verify structural completeness and domain limits of predictions."""
    valid_input = pd.DataFrame({
        "gender": ["Male", "Female", "Female"],
        "SeniorCitizen": [0, 0, 1],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "No"],
        "tenure": [1, 24, 60],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Credit card (automatic)", "Mailed check"],
        "MonthlyCharges": [50.5, 80.2, 22.0],
        "TotalCharges": [50.5, 1924.8, 1320.0]
    })
    
    scored_output = predict_churn(
        valid_input,
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH
    )
    
    required_cols = ["churn_probability", "churn_prediction", "risk_segment"]
    for col in required_cols:
        assert col in scored_output.columns
        
    assert scored_output["churn_probability"].min() >= 0.0
    assert scored_output["churn_probability"].max() <= 1.0
    assert scored_output["churn_prediction"].isin([0, 1]).all()


def test_empty_dataframe_raises_value_error():
    """Test 3: Assert robust failures on empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        predict_churn(empty_df, model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH)