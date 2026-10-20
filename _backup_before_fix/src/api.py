"""
Part 4: FastAPI backend — real-time heartbeat scoring service.

Run with:
    uvicorn api:app --reload --port 8000

Then send it a heartbeat's features and it returns a prediction instantly.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(title="Real-Time Abnormal Heartbeat Detection API")

# Load the trained model and scaler once, when the server starts
model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


class HeartbeatFeatures(BaseModel):
    rr_prev: float
    rr_next: float
    amplitude: float
    local_mean: float
    local_std: float


@app.get("/")
def root():
    return {"message": "Real-Time Abnormal Heartbeat Detection API is running."}


@app.post("/predict")
def predict(features: HeartbeatFeatures):
    """
    Accepts one heartbeat's features and returns whether it's
    normal or abnormal, plus a risk probability.
    """
    X = np.array([[
        features.rr_prev,
        features.rr_next,
        features.amplitude,
        features.local_mean,
        features.local_std
    ]])

    # Some models (tree-based) were trained on unscaled data in train_models.py,
    # but scaling unscaled inputs doesn't hurt tree models much if you want one
    # unified path. For strict correctness, keep this consistent with how your
    # chosen best_model was trained (scaled vs unscaled).
    try:
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]
    except Exception:
        prob = model.predict_proba(X)[0][1]

    prediction = "Abnormal" if prob >= 0.5 else "Normal"

    return {
        "prediction": prediction,
        "abnormal_probability": round(float(prob), 4)
    }
