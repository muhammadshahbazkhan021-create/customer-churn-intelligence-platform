import logging
import pandas as pd
from typing import List

# Internal validation schemas aur inference models import karein
from api.schemas import (
    CustomerInputSchema, 
    PredictionResponseSchema, 
    BatchCustomerInputSchema, 
    BatchPredictionResponseSchema
)
from churn_project.models.predict import predict_churn, load_artifacts

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionService:
    """
    Acts as the operational bridge between FastAPI request validation endpoints 
    and downstream machine learning serialization models.
    """
    def __init__(
        self, 
        model_path: str = "models/churn_xgb_model.joblib", 
        preprocessor_path: str = "models/preprocessor.joblib"
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        
        # Trigger explicit warmup routine during initialization state
        self._warmup_service_artifacts()

    def _warmup_service_artifacts(self) -> None:
        """
        Forces artifact load registers to resolve on initial application startup loop,
        eliminating warm-up latency lags for the very first endpoint customer request.
        """
        try:
            logger.info("Initializing pre-emptive service memory artifact warmup context...")
            load_artifacts(self.model_path, self.preprocessor_path)
            logger.info("Service operational warmup execution finished successfully.")
        except Exception as e:
            logger.error(f"Critical System Core alert: Warmup cycle failed loading memory state: {str(e)}")
            # Real applications do not raise here if files generate on pipeline triggers later,
            # but we log it as an operational hazard tracker indicator flags.

    def predict_single(self, customer: CustomerInputSchema) -> PredictionResponseSchema:
        """
        Converts a single customer input validation schema payload block into a vector array row,
        routes it through the inference model, and transforms it back to an API output response.
        """
        logger.info(f"Processing singular telemetry request sequence for customer ID: {customer.customerID}")
        
        # 1. Transform schema validation dictionary mapping into standard single row Pandas row
        input_dict = customer.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # 2. Fire predictive inference calculation block pipeline
        inference_results = predict_churn(
            input_data=input_df,
            model_path=self.model_path,
            preprocessor_path=self.preprocessor_path
        )
        
        # Extract individual prediction item parameters array index scalar values
        row_result = inference_results.iloc[0]
        
        # 3. Encapsulate parameters array outputs clean into return frame schema
        return PredictionResponseSchema(
            customer_id=customer.customerID,
            churn_prediction=int(row_result['churn_prediction']),
            churn_probability=float(row_result['churn_probability']),
            risk_segment=str(row_result['risk_segment'])
        )

    def predict_batch(self, batch: BatchCustomerInputSchema) -> BatchPredictionResponseSchema:
        """
        Processes entire collections of input rows in parallel via vector arrays, 
        completely avoiding slow, expensive Python loops for high-throughput batch tracking.
        """
        logger.info(f"Processing vector batch calculation stream array size: {len(batch.customers)} items.")
        
        # 1. Convert collection arrays into clean high performance multi-dimensional row array structure
        batch_dicts = [customer.model_dump() for customer in batch.customers]
        batch_df = pd.DataFrame(batch_dicts)
        
        # 2. Route full frame array through optimized vectorized computation pipeline blocks
        inference_results = predict_churn(
            input_data=batch_df,
            model_path=self.model_path,
            preprocessor_path=self.preprocessor_path
        )
        
        # 3. Synchronize vector alignments back to fast response tracking schema nodes
        compiled_output_responses: List[PredictionResponseSchema] = []
        
        for idx, row in inference_results.iterrows():
            original_customer_record = batch.customers[idx]
            
            response_node = PredictionResponseSchema(
                customer_id=original_customer_record.customerID,
                churn_prediction=int(row['churn_prediction']),
                churn_probability=float(row['churn_probability']),
                risk_segment=str(row['risk_segment'])
            )
            compiled_output_responses.append(response_node)
            
        logger.info(f"Batch transformation collection arrays synchronized successfully for {len(compiled_output_responses)} records.")
        return BatchPredictionResponseSchema(predictions=compiled_output_responses)
