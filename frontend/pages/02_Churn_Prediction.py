import io
import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Prediction Engine", page_icon="🔮", layout="wide")

st.title("🔮 Customer Churn Prediction Engine")
st.markdown("Assess customer churn risks in real-time using either individual profiling or bulk file processing.")

BASE_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")

tab1, tab2 = st.tabs(["👤 Single Customer Prediction", "📁 Batch CSV Upload"])

# =========================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# =========================================================================
with tab1:
    st.subheader("Analyze Individual Customer Risk")
    
    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### **Account & Financials**")
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
            monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
            total_charges = st.slider("Total Charges ($)", min_value=0.0, max_value=9000.0, value=780.0)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        with col2:
            st.markdown("##### **Connectivity & Services**")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            # Exact literal matches expected by backend schema:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col3:
            st.markdown("##### **Add-on Subscriptions**")
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submit_btn = st.form_submit_button("Predict Churn Risk")

    if submit_btn:
        payload = {
            "customerID": "WEB-PRED-001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges)
        }
        
        try:
            with st.spinner("Evaluating inference profile..."):
                response = requests.post(f"{BASE_API_URL}/predict", json=payload, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                prob = result.get("churn_probability", 0.0)
                risk_tier = result.get("risk_segment", "Low")
                
                st.markdown("---")
                st.subheader("🎯 Risk Assessment Output")
                
                metric_col, badge_col = st.columns(2)
                with metric_col:
                    st.metric(label="Churn Probability", value=f"{prob * 100:.1f}%")
                    st.progress(float(prob))
                
                with badge_col:
                    if risk_tier == "Low":
                        st.markdown("### 🟢 **Risk Level:** Low Risk")
                        st.success("Customer shows solid retention signals. Maintain standard relationship.")
                    elif risk_tier == "Medium":
                        st.markdown("### 🟡 **Risk Level:** Medium Risk")
                        st.warning("Customer engagement dropping. Consider contract incentives or service review.")
                    else:
                        st.markdown("### 🔴 **Risk Level:** High Risk")
                        st.error("Critical flight risk! Proactive retention contact recommended immediately.")
            else:
                err_detail = response.json().get("detail", response.text)
                st.error(f"Backend API Error ({response.status_code}): {err_detail}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Failed: Ensure backend is running on {BASE_API_URL}. Details: {e}")

# =========================================================================
# TAB 2: BATCH CSV UPLOAD
# =========================================================================
with tab2:
    st.subheader("Bulk File Processing Portal")
    st.markdown("Upload a CSV file containing customer portfolios to generate bulk churn scores.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write(f"📁 **File Uploaded:** {len(input_df)} records identified.")
            st.dataframe(input_df.head(5), use_container_width=True)
            
            if st.button("Process Batch Prediction"):
                # Clean missing fields and prepare JSON payload matching BatchCustomerInputSchema
                records = input_df.to_dict(orient="records")
                batch_payload = {"customers": records}
                
                try:
                    with st.spinner("Processing batch pipeline through API..."):
                        response = requests.post(f"{BASE_API_URL}/predict/batch", json=batch_payload, timeout=60)
                    
                    if response.status_code == 200:
                        batch_res = response.json()
                        predictions_list = batch_res.get("predictions", [])
                        pred_df = pd.DataFrame(predictions_list)
                        
                        scored_df = pd.concat([input_df.reset_index(drop=True), pred_df.drop(columns=["customer_id"], errors="ignore")], axis=1)
                        
                        st.success(f"🎉 Processed {len(scored_df)} customer records successfully!")
                        st.dataframe(scored_df.head(10), use_container_width=True)
                        
                        csv_buffer = io.StringIO()
                        scored_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="📥 Download Scored Churn Report (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name="customer_churn_scored.csv",
                            mime="text/csv"
                        )
                    else:
                        err_detail = response.json().get("detail", response.text)
                        st.error(f"Batch Processing Error ({response.status_code}): {err_detail}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to communicate with Batch API: {e}")
                    
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")