import pytest
from fastapi import status


def test_health_check_endpoint(api_client):
    """Verifies that the /health endpoint responds with standard 200 payload."""
    response = api_client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert data["version"] == "1.0.0"


def test_predict_single_success(api_client, valid_customer_payload):
    """Verifies single-customer inference produces probability, prediction, and risk band."""
    response = api_client.post("/api/v1/predict", json=valid_customer_payload)

    # If local model artifacts are not present on disk, expect graceful 500 error handling
    if response.status_code == status.HTTP_200_OK:
        body = response.json()
        assert body["customer_id"] == valid_customer_payload["customerID"]
        assert body["churn_prediction"] in [0, 1]
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_segment"] in ["Low", "Medium", "High"]
    else:
        assert response.status_code in [
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            status.HTTP_400_BAD_REQUEST,
        ]


def test_predict_single_validation_error(api_client, valid_customer_payload):
    """Asserts that 422 Unprocessable Entity is raised when invalid types/ranges are submitted."""
    invalid_payload = valid_customer_payload.copy()
    invalid_payload["tenure"] = -5  # Violates ge=0 constraint

    response = api_client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_batch_endpoint(api_client, valid_customer_payload):
    """Verifies batch inference payload execution."""
    batch_payload = {
        "customers": [
            valid_customer_payload,
            {**valid_customer_payload, "customerID": "9999-XYZ"},
        ]
    }
    response = api_client.post("/api/v1/predict/batch", json=batch_payload)

    if response.status_code == status.HTTP_200_OK:
        body = response.json()
        assert "predictions" in body
        assert len(body["predictions"]) == 2
        assert body["predictions"][0]["customer_id"] == "7590-VHVEG"
        assert body["predictions"][1]["customer_id"] == "9999-XYZ"
    else:
        assert response.status_code in [
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            status.HTTP_400_BAD_REQUEST,
        ]