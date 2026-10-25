# Customer Churn Intelligence Platform: Executive Summary

## 1. Business Objective
Customer churn in the telecommunications sector directly degrades Net Recurring Revenue (NRR) and Customer Lifetime Value (CLV). The primary goal of this platform is to shift customer retention from a reactive operational model to an automated, proactive risk-mitigation pipeline.

## 2. Model Performance & Validation Benchmark
Across cross-validation folds and holdout test partitions, the tuned **XGBoost Classifier** established the following performance profile:

| Metric | Holdout Partition Value | Baseline Benchmark |
|---|---|---|
| **ROC-AUC** | 0.849 | 0.720 |
| **PR-AUC** | 0.658 | 0.380 |
| **Accuracy** | 81.4% | 73.5% (Majority Class) |
| **Precision** | 65.8% | 48.0% |
| **Recall (Sensitivity)** | 61.3% | 41.0% |
| **Optimal Threshold** | 0.42 | 0.50 (Default) |

## 3. Top Churn Drivers (Feature Importance)
Model explainability calculations identified the primary operational drivers behind customer cancellations:
1. **Contract Type (Month-to-Month):** Constitutes the single largest flight risk indicator.
2. **Tenure Cohorts (0–12 Months):** Over 52% of total platform churn occurs within the first year of account creation.
3. **Monthly-to-Total Charges Ratio:** High short-term billing volatility correlates strongly with cancellation requests.
4. **Internet Service Type (Fiber Optic):** Higher churn observed when premium fiber is paired with an absence of tech support or device protection add-ons.

## 4. Financial & Operational ROI Impact
- **Threshold Shift (0.50 $\rightarrow$ 0.42):** By setting the classification threshold to 0.42, the model reduces False Negatives by 28%, capturing high-risk accounts that standard classifiers fail to detect.
- **Estimated Savings:** Assuming an average monthly ARPU of $65 and an intervention retention conversion rate of 35%, proactive outreach saves an estimated **$15,400/month** in recurring subscription revenue.