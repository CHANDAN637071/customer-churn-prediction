import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

MODEL_PATH = r"C:\Users\pradh\OneDrive\Desktop\Customer Churn Prediction and Retention Analysis\churn-project\models\lgbm_churn_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
explainer = shap.TreeExplainer(model)

st.title("Customer Churn Prediction & Retention Dashboard")

tab1, tab2 = st.tabs(["Single Customer Lookup", "Batch Upload"])

# ---------------- TAB 1: Single customer ----------------
with tab1:
    st.subheader("Enter customer details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col3:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    col4, col5, col6 = st.columns(3)
    with col4:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col5:
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with col6:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

    total_charges = monthly_charges * tenure if tenure > 0 else monthly_charges

    if st.button("Predict Churn Risk"):
        input_dict = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
        }
        input_df = pd.DataFrame([input_dict])

        categorical_cols = input_df.select_dtypes(include="object").columns.tolist()
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

        # Align columns with training data
        input_encoded = input_encoded.reindex(columns=model.feature_name_, fill_value=0)

        proba = model.predict_proba(input_encoded)[0, 1]

        st.metric("Churn Probability", f"{proba:.1%}")
        if proba >= 0.7:
            st.error("High Risk — recommend immediate retention action")
        elif proba >= 0.4:
            st.warning("Medium Risk — monitor and consider proactive offer")
        else:
            st.success("Low Risk")

        # SHAP explanation
        st.subheader("Why this prediction?")
        shap_values = explainer.shap_values(input_encoded)
        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

        fig, ax = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                data=input_encoded.iloc[0],
                feature_names=input_encoded.columns.tolist()
            ),
            show=False
        )
        st.pyplot(fig)

# ---------------- TAB 2: Batch upload ----------------
with tab2:
    st.subheader("Upload a CSV of customers")
    uploaded_file = st.file_uploader("CSV file", type="csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview:", batch_df.head())

        if "TotalCharges" in batch_df.columns:
            batch_df["TotalCharges"] = pd.to_numeric(batch_df["TotalCharges"], errors="coerce").fillna(0)

        drop_cols = [c for c in ["customerID", "Churn"] if c in batch_df.columns]
        features_df = batch_df.drop(columns=drop_cols)

        categorical_cols = features_df.select_dtypes(include="object").columns.tolist()
        features_encoded = pd.get_dummies(features_df, columns=categorical_cols)
        features_encoded = features_encoded.reindex(columns=model.feature_name_, fill_value=0)

        batch_df["churn_probability"] = model.predict_proba(features_encoded)[:, 1]
        batch_df["risk_segment"] = pd.cut(
            batch_df["churn_probability"], bins=[-0.01, 0.4, 0.7, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        )

        st.write("Predictions:", batch_df)

        csv_out = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions as CSV", csv_out, "churn_predictions.csv", "text/csv")