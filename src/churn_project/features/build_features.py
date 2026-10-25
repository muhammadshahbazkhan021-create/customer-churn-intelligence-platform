import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    fe_df = df.copy()
    logger.info("Executing Domain Feature Engineering...")

    if "tenure" in fe_df.columns:
        bins = [-1, 12, 24, 48, np.inf]
        labels = ["0-12_months", "12-24_months", "24-48_months", "48+_months"]
        fe_df["Tenure_Cohort"] = pd.cut(fe_df["tenure"], bins=bins, labels=labels).astype(str)

    if "MonthlyCharges" in fe_df.columns and "TotalCharges" in fe_df.columns:
        fe_df["Monthly_to_Total_Ratio"] = fe_df["MonthlyCharges"] / (fe_df["TotalCharges"] + 1.0)

    if "TotalCharges" in fe_df.columns and "tenure" in fe_df.columns:
        fe_df["Charges_per_Tenure"] = fe_df["TotalCharges"] / (fe_df["tenure"] + 1.0)

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    available_services = [col for col in service_cols if col in fe_df.columns]
    if available_services:
        fe_df["Service_Count"] = (fe_df[available_services] == "Yes").sum(axis=1)
    else:
        fe_df["Service_Count"] = 0

    return fe_df


def build_preprocessor(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder="drop"
    )


def fit_preprocessor(df: pd.DataFrame, preprocessor: ColumnTransformer) -> ColumnTransformer:
    X = df.drop(columns=["Churn"]) if "Churn" in df.columns else df
    return preprocessor.fit(X)