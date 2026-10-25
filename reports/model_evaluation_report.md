# Model Evaluation & Benchmark Report

## Validation Metrics (Holdout Set)
- **ROC-AUC Score:** 0.849
- **Accuracy:** 81.4%
- **F1-Score:** 0.635
- **Optimal Threshold:** 0.42 (tuned for recall maximization to prevent False Negatives)

## Conclusion
The tuned XGBoost model with `scale_pos_weight` addresses class imbalance without artificial data distortion. Artifacts are validated for production deployment.