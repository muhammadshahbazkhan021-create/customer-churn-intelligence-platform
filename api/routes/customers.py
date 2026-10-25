import os
import json
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any

# Internal project imports
from database.connection import get_db
from database.models import PredictionLog

router = APIRouter(prefix="/customers", tags=["Customers & History"])


@router.get("/{customer_id}/history", status_code=status.HTTP_200_OK)
def _log_to_dict(log: PredictionLog) -> Dict[str, Any]:
    return {
        "id": log.id,
        "customer_id": log.customer_id,
        "tenure": log.tenure,
        "monthly_charges": log.monthly_charges,
        "total_charges": log.total_charges,
        "contract": log.contract,
        "internet_service": log.internet_service,
        "churn_probability": log.churn_probability,
        "churn_prediction": log.churn_prediction,
        "risk_segment": log.risk_segment,
        "created_at": log.created_at.isoformat() if log.created_at else None,
    }
def get_customer_history(customer_id: str, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """
    Fetches the entire historical prediction logs logged for a specific customer ID.
    Raises a 404 HTTP Exception if no historical logs are discovered.
    """
    logs = (
        db.query(PredictionLog)
        .filter(PredictionLog.customer_id == customer_id)
        .order_by(PredictionLog.created_at.desc())
        .all()
    )
    
    if not logs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction history found for customer ID: '{customer_id}'"
        )
    
        
    # Standardizing object to dict extraction
    return [log.__dict__ for log in logs]


@router.get("/recent-predictions", status_code=status.HTTP_200_OK)
def get_recent_predictions(
    limit: int = Query(default=50, ge=1, le=500), 
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent system-wide prediction logs.
    Sorted chronologically by 'created_at' in descending order.
    """
    recent_logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    
    return [log.__dict__ for log in recent_logs]
