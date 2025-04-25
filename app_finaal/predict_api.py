from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Load model and scaler
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI(title="Fraud Detection API")

# Example feature order reference (replace with actual column order used after preprocessing)
FEATURE_ORDER = [
    "Age", "Gender", "State", "City", "Bank_Branch", "Account_Type",
    "Transaction_Amount", "Transaction_Type", "Merchant_Category",
    "Transaction_Device", "Transaction_Location", "Device_Type",
    "Transaction_Currency", "Transaction_Description", "Account_Balance"
]

class Transaction(BaseModel):
    features: list[float]  # List of numerical input values in proper order

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        if len(transaction.features) != len(FEATURE_ORDER):
            raise ValueError(f"Expected {len(FEATURE_ORDER)} features: {FEATURE_ORDER}")

        input_data = np.array(transaction.features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        return {"fraud_prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("predict_api:app", host="localhost", port=8000, reload=True)