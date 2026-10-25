import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Customer Churn Intelligence Platform")
st.markdown("### Production-Grade Predictive Analytics & Retention System")

BASE_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")

# Check Backend API Status with Graceful Local Mode Fallback
api_status = False
try:
    health_resp = requests.get(f"{BASE_API_URL}/health", timeout=2)
    if health_resp.status_code == 200:
        api_status = True
except Exception:
    api_status = False

col1, col2, col3 = st.columns(3)

with col1:
    if api_status:
        st.success("🟢 **Backend API:** Online (FastAPI)")
    else:
        st.info("🟡 **Inference Engine:** Standalone Mode (Direct Artifacts)")

with col2:
    st.metric(label="Model Architecture", value="XGBoost Classifier")

with col3:
    st.metric(label="Optimized Threshold", value="0.42 (High Recall)")

st.divider()

st.markdown("""
Welcome to the **Customer Churn Intelligence Platform**. This system enables enterprise retention teams to:

1. **Explore Cohorts:** Analyze customer demographics, contracts, and service subscriptions.
2. **Predict Risk:** Run real-time single evaluations or upload bulk customer CSV files.
3. **Audit Performance:** Review model evaluation coefficients, confusion matrix distribution, and business ROI savings.

👈 **Navigate using the sidebar to explore features.**
""")


# 4. KPI Metrics
st.subheader("Global Churn Key Performance Indicators (KPIs)")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Customers Analyzed", 
        value="10,240", 
        delta="+12% this month"
    )

with col2:
    st.metric(
        label="Baseline Churn Rate", 
        value="26.5%", 
        delta="-1.2%", 
        delta_color="inverse" # Negative change is good for churn
    )

with col3:
    st.metric(
        label="Estimated Retention Value", 
        value="$142,500", 
        delta="+$15,300 saved"
    )

st.divider()

# 5. Navigation Guide
st.subheader("Explore the Platform")
st.write("Use the sidebar navigation to access the following modules:")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.info("""
    #### 📈 1. Executive Dashboard
    * **Purpose:** High-level summary for stakeholders.
    * **Features:** Churn trends, demographic insights, and overall business revenue impact visualization.
    """)

with nav_col2:
    st.info("""
    #### 🔮 2. Single Prediction
    * **Purpose:** Analyze individual customer risk.
    * **Features:** Input a single customer's data to get an instant churn probability score and risk breakdown.
    """)

with nav_col3:
    st.info("""
    #### 📁 3. Bulk Analytics
    * **Purpose:** Batch processing for marketing teams.
    * **Features:** Upload a CSV file of thousands of customers to generate a downloadable churn risk report.
    """)
