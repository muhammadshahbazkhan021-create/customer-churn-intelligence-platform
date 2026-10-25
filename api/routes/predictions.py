
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_prediction_service
from api.schemas import (
    CustomerInputSchema,
    PredictionResponseSchema,
    BatchCustomerInputSchema,
    BatchPredictionResponseSchema,
)
from api.services.prediction_service import PredictionService

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponseSchema,
)
def predict_customer(
    customer: CustomerInputSchema,
    prediction_service: PredictionService = Depends(
        get_prediction_service
    ),
):
    """
    Generate a churn prediction for one customer.
    """

    try:
        result = prediction_service.predict_single(customer)
        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during prediction.",
        ) from exc


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponseSchema,
)
def predict_batch(
    batch: BatchCustomerInputSchema,
    prediction_service: PredictionService = Depends(
        get_prediction_service
    ),
):
    """
    Generate churn predictions for multiple customers.
    """

    try:
        result = prediction_service.predict_batch(batch)
        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during batch prediction.",
        ) from exc
