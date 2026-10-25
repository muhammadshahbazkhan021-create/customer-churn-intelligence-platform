import os
import sys
import pytest
import pandas as pd
from fastapi.testclient import TestClient

# Ensure project root is available on sys.path for test execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app


@pytest.fixture(scope="session")
def api_client():
    """Provides an isolated TestClient instance across the test session."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_customer_payload():
    """Returns a schema-compliant single customer payload matching CustomerInputSchema."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 358.20,
    }


@pytest.fixture
def sample_raw_dataframe():
    """Generates a small DataFrame with known edge cases for cleaning verification."""
    return pd.DataFrame(
        {
            "customerID": ["001-A", "002-B", "003-C"],
            "gender": ["Male", "Female", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [1, 0, 48],
            "PhoneService": ["Yes", "No", "Yes"],
            "MultipleLines": ["No", "No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "No", "No internet service"],
            "TechSupport": ["No", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No", "No"],
            "PaymentMethod": [
                "Mailed check",
                "Electronic check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [45.0, 70.0, 20.0],
            "TotalCharges": ["45.0", " ", "960.0"],
            "Churn": ["No", "Yes", "No"],
        }
    )