# api/config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Customer Churn Intelligence API"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # Model Artifact Paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "artifacts/models/best_model.joblib")
    PREPROCESSOR_PATH: str = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessors/preprocessor.joblib")
    DEFAULT_THRESHOLD: float = float(os.getenv("DEFAULT_THRESHOLD", 0.5))

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()