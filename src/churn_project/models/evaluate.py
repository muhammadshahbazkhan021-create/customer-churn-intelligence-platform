import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(
    model: Any, 
    X_test_proc: np.ndarray, 
    y_test: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates a trained model on processed test data and compiles a comprehensive
    performance report, including metrics, confusion matrix, and threshold optimization.

    Parameters:
    -----------
    model : Any
        The trained model artifact (e.g., XGBClassifier).
    X_test_proc : np.ndarray
        The preprocessed/transformed feature matrix for testing.
    y_test : np.ndarray
        True binary target labels (0 or 1).
    threshold : float, default 0.5
        The classification decision threshold for mapping probabilities to binary classes.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing all calculated metrics, confusion matrix counts, and recommendations.
    """
    logger.info("Initializing comprehensive performance evaluation payload...")

    # Probabilities extract karein (positive class indices)
    y_proba = model.predict_proba(X_test_proc)[:, 1]
    
    # Custom threshold apply karke predictions calculate karein
    y_pred = (y_proba >= threshold).astype(int)

    # 1. Base Classification Metrics Calculation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    # 2. Precision-Recall AUC (PR-AUC) computation
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recalls, precisions)

    # 3. Confusion Matrix Breakdown Extraction
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # 4. Threshold Recommendation Engine (Scan to optimize F1-Score)
    logger.info("Running optimization grid to scan for best decision threshold...")
    best_f1 = 0.0
    best_threshold = 0.5
    
    # Generate 100 candidate thresholds from 0.01 to 0.99
    candidate_thresholds = np.linspace(0.01, 0.99, 100)
    for t in candidate_thresholds:
        current_preds = (y_proba >= t).astype(int)
        current_f1 = f1_score(y_test, current_preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    # Base Metrics Structure compiled into dictionary
    evaluation_report = {
        "meta": {
            "applied_threshold": float(threshold),
            "total_records": int(len(y_test))
        },
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "threshold_recommendation": {
            "optimal_f1_threshold": float(np.round(best_threshold, 4)),
            "maximized_f1_score": float(np.round(best_f1, 4))
        }
    }

    logger.info(f"Evaluation calculated. Base F1: {f1:.4f} | Optimal F1 (at {best_threshold:.2f}): {best_f1:.4f}")
    return evaluation_report

def save_evaluation_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Serializes evaluation performance metrics directly to disk in structured JSON format.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        The evaluation reporting payload map generated via evaluate_model().
    output_path : str
        Target storage location on disk where the file will be saved.
    """
    try:
        # Parent directory exist karta hai ya nahi check karein aur ensure karein
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Successfully exported evaluation telemetry metadata to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to serialize metrics payload report to path: {output_path}. Error: {str(e)}")
        raise e

if __name__ == "__main__":
    from xgboost import XGBClassifier
    
    print("--- Simulating Isolated Unit Performance Metrics Calculation ---")
    
    # Simple Mock Matrix Configurations for pipeline verification
    X_mock = np.random.rand(100, 5)
    y_mock = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    
    # Initialize basic dummy state classifier models
    mock_xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    mock_xgb.fit(X_mock, y_mock)
    
    # Compute simulation results
    sample_report = evaluate_model(model=mock_xgb, X_test_proc=X_mock, y_test=y_mock)
    
    # Attempt local logging tracking export verification
    test_export_path = "artifacts/metrics/evaluation_metrics.json"
    save_evaluation_metrics(metrics=sample_report, output_path=test_export_path)
    
    print("\n=== Sample Generated Metric Report Payload ===")
    print(json.dumps(sample_report, indent=2))
