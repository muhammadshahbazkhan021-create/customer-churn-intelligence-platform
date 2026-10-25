import os
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Modules import karein
from churn_project.data.load_data import load_raw_data
from churn_project.data.clean_data import clean_customer_data
from churn_project.features.build_features import (
    create_domain_features,
    build_preprocessor,
    fit_preprocessor
)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_churn_model(
    data_path: str, 
    model_save_path: str, 
    preprocessor_save_path: str, 
    cv_splits: int = 5
) -> dict:
    """
    Executes the end-to-end training pipeline: loads, cleans, engineers features, 
    preprocesses, handles class imbalance, trains XGBoost, evaluates, and saves artifacts.
    """
    logger.info("Initializing full end-to-end model training pipeline.")

    # 1. Data Ingestion & Cleaning
    raw_df = load_raw_data(file_path=data_path)
    cleaned_df = clean_customer_data(raw_df, is_training=True)
    
    # 2. Domain Feature Engineering
    featured_df = create_domain_features(cleaned_df)

    # 3. Stratified Split
    X = featured_df.drop(columns=['Churn'])
    y = featured_df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Stratified Train/Test split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Dynamic Feature Identification
    # Identify numerical columns based on dtype
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    # Categorical columns include text and categorical boundaries like custom cohorts
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Dynamically detected {len(numerical_cols)} numerical and {len(categorical_cols)} categorical fields.")

    # 5. Build, Fit and Transform via Preprocessing Pipeline
    base_preprocessor = build_preprocessor(categorical_cols=categorical_cols, numerical_cols=numerical_cols)
    fitted_preprocessor = fit_preprocessor(X_train, base_preprocessor)
    
    X_train_proc = fitted_preprocessor.transform(X_train)
    X_test_proc = fitted_preprocessor.transform(X_test)

    # 6. Imbalance Management: Dynamic scale_pos_weight calculation
    # Formula: total_negative_examples / total_positive_examples
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / max(1, pos_count)
    logger.info(f"Class imbalance distribution monitored. Computed scale_pos_weight: {scale_weight:.4f}")

    # 7. XGBoost Initialization & Cross-Validation
    xgb_model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss'
    )

    logger.info(f"Evaluating model performance with {cv_splits}-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_model, X_train_proc, y_train, cv=cv, scoring='roc_auc')
    logger.info(f"Cross-Validation ROC-AUC Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # 8. Full Training on Train Set
    logger.info("Training final XGBoost classifier...")
    xgb_model.fit(X_train_proc, y_train)

    # 9. Test Evaluation Metrics
    y_pred = xgb_model.predict(X_test_proc)
    y_proba = xgb_model.predict_proba(X_test_proc)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cv_roc_auc_mean": np.mean(cv_scores)
    }

    logger.info("Evaluation Complete. Calculated Test Performance Metrics:")
    for metric_name, val in metrics.items():
        logger.info(f" - {metric_name.upper()}: {val:.4f}")

    # 10. Artifact Persistence
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_save_path), exist_ok=True)
    
    joblib.dump(xgb_model, model_save_path)
    joblib.dump(fitted_preprocessor, preprocessor_save_path)
    logger.info(f"Successfully serialized trained model to {model_save_path}")
    logger.info(f"Successfully serialized fitted preprocessor pipeline to {preprocessor_save_path}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full XGBoost pipeline training.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to raw csv file.")
    parser.add_argument("--model_path", type=str, default="models/churn_xgb_model.joblib", help="Target path for saving trained model.")
    parser.add_argument("--preprocessor_path", type=str, default="models/preprocessor.joblib", help="Target path for saving fitted pipeline.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation splits.")
    
    args = parser.parse_args()

    train_churn_model(
        data_path=args.data_path,
        model_save_path=args.model_path,
        preprocessor_save_path=args.preprocessor_path,
        cv_splits=args.cv
    )
