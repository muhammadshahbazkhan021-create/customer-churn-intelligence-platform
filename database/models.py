from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from database.connection import Base


class PredictionLog(Base):
    """
    SQLAlchemy Model representing the 'prediction_logs' table.
    Stores input features, model inference metadata, and timestamps for audit.
    """
    __tablename__ = "prediction_logs"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Customer Identifier (Indexed for fast lookups/history queries)
    customer_id = Column(String, nullable=True, index=True)
    
    # Input Features from Churn Model
    tenure = Column(Integer, nullable=False)
    monthly_charges = Column(Float, nullable=False)
    total_charges = Column(Float, nullable=False)
    contract = Column(String, nullable=False)
    internet_service = Column(String, nullable=False)
    
    # Model Outputs & Analytics
    churn_probability = Column(Float, nullable=False)  # Continuous output (0.0 to 1.0)
    churn_prediction = Column(Integer, nullable=False)   # Binary output (0 or 1)
    risk_segment = Column(String, nullable=False)       # Low, Medium, High
    
    # Audit Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
