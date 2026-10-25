from functools import lru_cache
from api.services.prediction_service import PredictionService
from api.config import settings

@lru_cache()
def get_prediction_service() -> PredictionService:
    return PredictionService(
        model_path=settings.MODEL_PATH,
        preprocessor_path=settings.PREPROCESSOR_PATH
    )