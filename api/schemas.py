from typing import List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerInputSchema(BaseModel):
    customerID: Optional[str] = Field(default=None)
    gender: Literal["Female", "Male"] = "Male"
    SeniorCitizen: Literal[0, 1] = 0
    Partner: Literal["Yes", "No"] = "No"
    Dependents: Literal["Yes", "No"] = "No"
    tenure: int = Field(..., ge=0)
    PhoneService: Literal["Yes", "No"] = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "DSL"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = "Electronic check"
    MonthlyCharges: float = Field(..., ge=0.0)
    TotalCharges: Union[float, str] = Field(..., description="Numeric float or raw string charges")

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v):
        if isinstance(v, str):
            clean_str = v.strip()
            if clean_str == "":
                return 0.0
            try:
                val = float(clean_str)
                if val < 0.0:
                    raise ValueError("TotalCharges must be non-negative")
                return val
            except ValueError:
                raise ValueError("TotalCharges must be numeric")
        if isinstance(v, (int, float)) and v < 0.0:
            raise ValueError("TotalCharges must be non-negative")
        return float(v)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.50,
                "TotalCharges": 1026.0
            }
        }
    )


class PredictionResponseSchema(BaseModel):
    customer_id: Optional[str] = None
    churn_prediction: int
    churn_probability: float
    risk_segment: str


class BatchCustomerInputSchema(BaseModel):
    customers: List[CustomerInputSchema]


class BatchPredictionResponseSchema(BaseModel):
    predictions: List[PredictionResponseSchema]