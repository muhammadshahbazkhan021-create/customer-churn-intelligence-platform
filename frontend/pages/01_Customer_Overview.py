import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. Page Configuration
st.set_page_config(page_title="Customer Overview", page_icon="📈", layout="wide")

st.title("📈 Customer Overview & Churn Trends")
st.markdown("Analyze historical trends, customer behavior, and key drivers impacting customer retention.")

# 2. Data Loading with Caching
@st.cache_data
def load_clean_data():
    # Dynamic path handling for multi-page environment
    file_path = os.path.join("data", "interim", "customer_churn_clean.csv")
    
    # Mock data generation if file doesn't exist yet (for immediate testing)
    if not os.path.exists(file_path):
        import numpy as np
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            "CustomerID": [f"CUST-{i:04d}" for i in range(n)],
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.5, 0.25, 0.25]),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.4, 0.4, 0.2]),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
            "tenure": np.random.randint(1, 72, n),
            "MonthlyCharges": np.random.uniform(20, 120, n),
            "TotalCharges": np.random.uniform(100, 5000, n),
            "Churn": np.random.choice(["Yes", "No"], n, p=[0.265, 0.735])
        })
    return pd.read_csv(file_path)

df = load_clean_data()

# 3. Top Filter Sidebar
st.sidebar.header("🎯 Filter Dataset")

# Contract Type Filter
all_contracts = df["Contract"].unique().tolist()
selected_contracts = st.sidebar.multiselect(
    "Select Contract Type", 
    options=all_contracts, 
    default=all_contracts
)

# Internet Service Filter
all_internet = df["InternetService"].unique().tolist()
selected_internet = st.sidebar.multiselect(
    "Select Internet Service", 
    options=all_internet, 
    default=all_internet
)

# Apply Filters dynamically
filtered_df = df[
    (df["Contract"].isin(selected_contracts)) & 
    (df["InternetService"].isin(selected_internet))
]

# Safeguard against empty filtered data
if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters. Please adjust your sidebar choices.")
    st.stop()

# 4. Interactive Charts (Plotly Layout)
st.subheader("📊 Analytical Deep Dive")

# Row 1: Churn Distribution & Tenure Distribution
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown("#### **Overall Churn Breakdown**")
    churn_counts = filtered_df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Status", "Count"]
    
    # Donut Chart for Churn distribution
    fig_donut = px.pie(
        churn_counts, names="Status", values="Count", 
        hole=0.4, color="Status",
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"}
    )
    fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
    st.plotly_chart(fig_donut, use_container_width=True)

with row1_col2:
    st.markdown("#### **Customer Longevity (Tenure vs Churn)**")
    # Histogram showcasing how quickly new customers drop off
    if df["Churn"].dtype in [int, "int64"]:
        df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})
    fig_hist = px.histogram(
        filtered_df, x="tenure", color="Churn", 
        barmode="overlay", marginal="box",
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
        labels={"tenure": "Tenure (Months)"}
    )
    fig_hist.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
    st.plotly_chart(fig_hist, use_container_width=True)


st.divider()

# Row 2: Contract/Payment Impact & Financial Scatter Matrix
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown("#### **Contract Type Churn Risk Profile**")
    # Grouping to check which contract structure brings highest risk
    contract_churn = filtered_df.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
    fig_bar = px.bar(
        contract_churn, x="Contract", y="Count", color="Churn", 
        barmode="group", color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"}
    )
    fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
    st.plotly_chart(fig_bar, use_container_width=True)

with row2_col2:
    st.markdown("#### **Financial Analysis: Charges vs Churn**")
    # Scatter plot correlating monthly billing, total billing, and target churn
    fig_scatter = px.scatter(
        filtered_df, x="MonthlyCharges", y="TotalCharges", color="Churn",
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
        opacity=0.6, labels={"MonthlyCharges": "Monthly Charges ($)", "TotalCharges": "Total Charges ($)"}
    )
    fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# 5. Summary Data Table
st.subheader("📋 Filtered Dataset Snippet")
st.markdown("Explore the detailed metrics for individual records. Use the built-in search and sort tools below:")

# Interactive data frame showcasing the top 15 records
st.dataframe(
    filtered_df.head(15), 
    use_container_width=True,
    hide_index=True
)
