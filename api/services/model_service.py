import os
import json
from typing import List, Dict, Any


class ModelService:
    """
    Service responsible for reading and serving metadata, structural configurations,
    and performance evaluation metrics directly from the current Model Registry artifacts.
    """
    def __init__(self, metrics_path: str = "artifacts/metrics/evaluation_metrics.json"):
        self.metrics_path = metrics_path
        
        # Hardcoded static architectural baselines based on model design specs
        self.architecture_type = "XGBClassifier"
        self.target_threshold = 0.5
        self.features = [
            "tenure", 
            "monthly_charges", 
            "total_charges", 
            "contract", 
            "internet_service"
        ]

    def _load_metrics_file(self) -> Dict[str, Any]:
        """Safely reads the evaluated metrics file if it exists on disk."""
        if not os.path.exists(self.metrics_path):
            return {"warning": f"Metrics artifact file not found at '{self.metrics_path}'"}
            
        try:
            with open(self.metrics_path, "r") as file:
                return json.load(file)
        except (json.JSONDecodeError, IOError):
            return {"error": "Failed to safely parse evaluation metrics json artifact."}

    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Aggregates operational parameters alongside validation performance profiles.
        """
        training_metrics = self._load_metrics_file()
        
        metadata = {
            "model_architecture": self.architecture_type,
            "feature_count": len(self.features),
            "classification_threshold": self.target_threshold,
            "evaluation_metrics": training_metrics
        }
        return metadata

    def get_feature_names(self) -> List[str]:
        """
        Returns the formal ordered sequence of expected input feature keys.
        """
        return self.features
