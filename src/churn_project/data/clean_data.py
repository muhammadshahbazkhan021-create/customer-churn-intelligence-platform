import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_customer_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    cleaned_df = df.copy()
    logger.info(f"Starting data cleaning. Shape: {cleaned_df.shape} | Mode: {'Training' if is_training else 'Inference'}")

    # 1. customerID Drop
    if 'customerID' in cleaned_df.columns:
        cleaned_df.drop(columns=['customerID'], inplace=True)
        logger.info("Dropped 'customerID' column.")

    # 2. Whitespace Clean & Explicit String Casting
    string_cols = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    for col in string_cols:
        if col != 'Churn':
            cleaned_df[col] = cleaned_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            cleaned_df[col] = cleaned_df[col].replace(['', ' ', 'None', 'nan', 'NaN'], np.nan)

    # 3. TotalCharges Numeric Conversion & Tenure-Aware Imputation
    if 'TotalCharges' in cleaned_df.columns:
        cleaned_df['TotalCharges'] = pd.to_numeric(cleaned_df['TotalCharges'], errors='coerce')
        if 'tenure' in cleaned_df.columns:
            # Impute 0.0 only for zero tenure
            zero_tenure_mask = (cleaned_df['tenure'] == 0) & (cleaned_df['TotalCharges'].isnull())
            cleaned_df.loc[zero_tenure_mask, 'TotalCharges'] = 0.0
            
            # Non-zero tenure fallbacks: tenure * MonthlyCharges if available
            if 'MonthlyCharges' in cleaned_df.columns:
                nonzero_nulls = (cleaned_df['tenure'] > 0) & (cleaned_df['TotalCharges'].isnull())
                cleaned_df.loc[nonzero_nulls, 'TotalCharges'] = (
                    cleaned_df.loc[nonzero_nulls, 'tenure'] * cleaned_df.loc[nonzero_nulls, 'MonthlyCharges']
                )

    # 4. SeniorCitizen to String Category
    if 'SeniorCitizen' in cleaned_df.columns:
        cleaned_df['SeniorCitizen'] = cleaned_df['SeniorCitizen'].astype(str)

    # 5. Redundant Categories Standardize
    redundant_mappings = {'No internet service': 'No', 'No phone service': 'No'}
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        if col != 'Churn':
            cleaned_df[col] = cleaned_df[col].replace(redundant_mappings)

    # 6. Sanity Checks
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        negatives = (cleaned_df[col] < 0).sum()
        if negatives > 0:
            logger.warning(f"Sanity Check: {negatives} negative values in '{col}'!")

    # 7. Target Encoding
    if is_training:
        if 'Churn' not in cleaned_df.columns:
            raise KeyError("Training mode active but 'Churn' column missing.")
        
        cleaned_df['Churn'] = cleaned_df['Churn'].astype(str).str.strip().map({'Yes': 1, 'No': 0})
        if cleaned_df['Churn'].isnull().any():
            raise ValueError("Target validation failed: Churn contains unexpected values.")
        cleaned_df['Churn'] = cleaned_df['Churn'].astype(int)
    else:
        if 'Churn' in cleaned_df.columns:
            cleaned_df.drop(columns=['Churn'], inplace=True)

    logger.info(f"Cleaning complete. Output shape: {cleaned_df.shape}")
    return cleaned_df