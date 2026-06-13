from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Initialize the API
app = FastAPI()

# --- THE FIX: Explicit CORS Whitelist ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://anirudhrao-24.github.io",  # Your live GitHub Pages frontend
        "http://localhost:8000",            # Your local testing environment
        "*"                                 # General fallback
    ], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dynamic Absolute Paths ---
# Gets the directory where api.py lives, then steps back to /models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../models')

print("Loading models into memory...")
scaler = joblib.load(os.path.join(MODELS_DIR, 'standard_scaler.pkl'))
meta_model = joblib.load(os.path.join(MODELS_DIR, 'meta_stacked_model.pkl'))
rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))

# Define the expected incoming data structure
class CVRequest(BaseModel):
    features: list

# --- The "Welcome" Route ---
@app.get("/")
def home():
    return {
        "status": "Online",
        "project": "BiFeO3 ML-Supercapacitor Pipeline",
        "message": "API is active. Send POST requests to /predict to execute the model."
    }

# --- The Core Prediction Engine ---
@app.post("/predict")
def predict_current(request: CVRequest):
    # 1. Scale the incoming data
    input_data = np.array(request.features)
    scaled_data = scaler.transform(input_data)
    
    # 2. Run the prediction
    prediction = meta_model.predict(scaled_data)
    
    # 3. Extract Feature Importance for the XAI visualizer
    feature_weights = rf_model.feature_importances_.tolist()
    
    # 4. Return the data to the JavaScript frontend
    return {
        "predicted_current": prediction.tolist(),
        "feature_importance": feature_weights
    }
