import pytest
import pandas as pd
import numpy as np

from churn_project.data.clean_data import clean_customer_data


@pytest.fixture
def sample_raw_data():
    """Generates standard polluted data payload for unit testing."""
    return pd.DataFrame({
        "customerID": ["7590-VHVEG", "5575-GNVDE", "3668-QNECE"],
        "tenure": [1, 0, 12],
        "TotalCharges": ["29.85", " ", "105.6"],
        "MultipleLines": ["No phone service", "No", "Yes"],
        "OnlineSecurity": ["No", "No internet service", "Yes"],
        "Churn": ["No", "No", "Yes"]
    })


def test_customer_id_is_dropped(sample_raw_data):
    """Test 1: Ensure customerID column is consistently dropped."""
    cleaned_df = clean_customer_data(sample_raw_data, is_training=True)
    assert "customerID" not in cleaned_df.columns


def test_total_charges_casting_and_imputation(sample_raw_data):
    """Test 2: Ensure empty strings map correctly and tenure=0 imputes 0.0."""
    cleaned_df = clean_customer_data(sample_raw_data, is_training=True)
    
    # DataType casting validation
    assert pd.api.types.is_float_dtype(cleaned_df["TotalCharges"])
    # Tenure == 0 mapping assertions
    assert cleaned_df.iloc[1]["TotalCharges"] == 0.0
    assert cleaned_df.iloc[0]["TotalCharges"] == 29.85


def test_redundant_categories_normalization(sample_raw_data):
    """Test 3: Verify redundant string definitions consolidate to 'No'."""
    cleaned_df = clean_customer_data(sample_raw_data, is_training=True)
    assert cleaned_df.iloc[0]["MultipleLines"] == "No"
    assert cleaned_df.iloc[1]["OnlineSecurity"] == "No"


def test_churn_column_modes(sample_raw_data):
    """Test 4: Validate target column state changes between Training and Inference operations."""
    # Training Mode: target maps to binary 1/0
    train_df = clean_customer_data(sample_raw_data, is_training=True)
    assert train_df["Churn"].tolist() == [0, 0, 1]
    
    # Inference Mode: target columns are completely isolated/ignored
    inference_df = clean_customer_data(sample_raw_data, is_training=False)
    assert "Churn" not in inference_df.columns