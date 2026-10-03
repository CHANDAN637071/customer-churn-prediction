import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_churn_model.pkl")

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load(MODEL_PATH)

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: Customer):
    input_df = pd.DataFrame([customer.model_dump()])

    categorical_cols = input_df.select_dtypes(include="object").columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    input_encoded = input_encoded.reindex(columns=model.feature_name_, fill_value=0)

    proba = float(model.predict_proba(input_encoded)[0, 1])

    if proba >= 0.7:
        risk = "High Risk"
    elif proba >= 0.4:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "churn_probability": round(proba, 4),
        "risk_segment": risk
    }