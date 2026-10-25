import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Intelligence", 
    page_icon="📊", 
    layout="wide"
)

# 2. Backend API Health Check Banner
def check_backend_status():
    api_url = "http://127.0.0.1:8000/api/v1/health"
    try:
        # 2 seconds timeout taake app hang na ho agar backend down ho
        response = requests.get(api_url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            # Assuming your health API returns {"status": "healthy", "model_loaded": True}
            is_model_loaded = data.get("model_loaded", True) 
            return True, is_model_loaded
        return False, False
    except requests.exceptions.RequestException:
        return False, False

# Check status and display banner
is_online, model_loaded = check_backend_status()

if is_online:
    st.success(f"🟢 **API Status:** Online | **Model Loaded:** {model_loaded}")
else:
    st.warning("⚠️ **Backend API Connection Failed:** Please ensure your FastAPI server is running on http://127.0.0.1:8000")

# 3. Project Overview
st.title("📊 Customer Churn Intelligence Portal")
st.markdown("""
### Project Overview
Welcome to the Customer Churn Intelligence platform. Customer churn occurs when customers stop doing business with a company. 
This dashboard leverages advanced Machine Learning to identify high-risk customers, uncover key churn drivers, 
and provide actionable retention strategies to maximize customer lifetime value.
""")

st.divider()

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
