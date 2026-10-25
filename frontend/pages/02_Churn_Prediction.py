import io
import os
import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Churn Prediction Engine", page_icon="🔮", layout="wide")

st.title("🔮 Customer Churn Prediction Engine")
st.markdown("Assess customer churn risks in real-time using either individual profiling or bulk file processing.")

BASE_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")


# Direct fallback inference function for Streamlit Cloud
@st.cache_resource
def load_local_pipeline():
    model_path = os.path.join("artifacts", "models", "best_model.joblib")
    prep_path = os.path.join("artifacts", "preprocessors", "preprocessor.joblib")
    if os.path.exists(model_path) and os.path.exists(prep_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        return model, preprocessor
    return None, None


local_model, local_preprocessor = load_local_pipeline()


def predict_locally(records_df):
    from src.churn_project.data.clean_data import clean_data
    from src.churn_project.features.build_features import build_features

    cleaned_df = clean_data(records_df, is_training=False)
    featured_df = build_features(cleaned_df, is_training=False)
    transformed_data = local_preprocessor.transform(featured_df)
    probabilities = local_model.predict_proba(transformed_data)[:, 1]

    results = []
    threshold = 0.42
    for prob in probabilities:
        prob = float(prob)
        is_churn = prob >= threshold
        if prob < 0.30:
            segment = "Low"
        elif prob < 0.60:
            segment = "Medium"
        else:
            segment = "High"
        results.append(
            {
                "churn_prediction": int(is_churn),
                "churn_probability": round(prob, 4),
                "risk_segment": segment,
            }
        )
    return results


tab1, tab2 = st.tabs(["👤 Single Customer Prediction", "📁 Batch CSV Upload"])

with tab1:
    st.subheader("Analyze Individual Customer Risk")
    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### **Account & Financials**")
            tenure = st.slider(
                "Tenure (Months)", min_value=0, max_value=72, value=12
            )
            monthly_charges = st.slider(
                "Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0
            )
            total_charges = st.slider(
                "Total Charges ($)", min_value=0.0, max_value=9000.0, value=780.0
            )
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        with col2:
            st.markdown("##### **Connectivity & Services**")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col3:
            st.markdown("##### **Add-on Subscriptions**")
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )

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
            "TotalCharges": float(total_charges),
        }

        prob, risk_tier = None, None

        # Try API first, fallback to direct artifacts if offline
        try:
            resp = requests.post(
                f"{BASE_API_URL}/predict", json=payload, timeout=3
            )
            if resp.status_code == 200:
                res = resp.json()
                prob = res.get("churn_probability")
                risk_tier = res.get("risk_segment")
        except Exception:
            pass

        if prob is None and local_model is not None:
            df_input = pd.DataFrame([payload])
            local_res = predict_locally(df_input)[0]
            prob = local_res["churn_probability"]
            risk_tier = local_res["risk_segment"]

        if prob is not None:
            st.markdown("---")
            st.subheader("🎯 Risk Assessment Output")
            metric_col, badge_col = st.columns(2)
            with metric_col:
                st.metric(label="Churn Probability", value=f"{prob * 100:.1f}%")
                st.progress(float(prob))
            with badge_col:
                if risk_tier == "Low":
                    st.markdown("### 🟢 **Risk Level:** Low Risk")
                    st.success("Customer shows solid retention signals.")
                elif risk_tier == "Medium":
                    st.markdown("### 🟡 **Risk Level:** Medium Risk")
                    st.warning("Customer engagement dropping.")
                else:
                    st.markdown("### 🔴 **Risk Level:** High Risk")
                    st.error("Critical flight risk! Proactive retention needed.")
        else:
            st.error(
                "Unable to evaluate prediction. Model artifacts not accessible."
            )

with tab2:
    st.subheader("Bulk File Processing Portal")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write(f"📁 **File Uploaded:** {len(input_df)} records.")
        st.dataframe(input_df.head(5), use_container_width=True)

        if st.button("Process Batch Prediction"):
            predictions_list = None

            # Try API first
            try:
                records = input_df.to_dict(orient="records")
                batch_payload = {"customers": records}
                resp = requests.post(
                    f"{BASE_API_URL}/predict/batch",
                    json=batch_payload,
                    timeout=10,
                )
                if resp.status_code == 200:
                    predictions_list = resp.json().get("predictions", [])
            except Exception:
                pass

            # Fallback to local
            if predictions_list is None and local_model is not None:
                predictions_list = predict_locally(input_df)

            if predictions_list:
                pred_df = pd.DataFrame(predictions_list)
                scored_df = pd.concat(
                    [
                        input_df.reset_index(drop=True),
                        pred_df.drop(columns=["customer_id"], errors="ignore"),
                    ],
                    axis=1,
                )

                st.success(
                    f"🎉 Processed {len(scored_df)} customer records"
                    " successfully!"
                )
                st.dataframe(scored_df.head(10), use_container_width=True)

                csv_buffer = io.StringIO()
                scored_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Scored Churn Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="customer_churn_scored.csv",
                    mime="text/csv",
                )
            else:
                st.error("Failed to process batch.")