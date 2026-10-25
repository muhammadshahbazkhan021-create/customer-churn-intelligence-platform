
from pathlib import Path

from fastapi import APIRouter, Depends

from api.dependencies import get_prediction_service

router = APIRouter()


@router.get("/health")
def health_check(
    prediction_service=Depends(get_prediction_service),
):
    """
    Check API health and model availability.
    """

    model_loaded = False

    try:
        # Check whether the prediction service can access its model.
        model_loaded = prediction_service.is_model_available()

    except Exception:
        model_loaded = False

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "version": "1.0.0",
    }
