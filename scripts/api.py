from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Initialize the API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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

@app.post("/predict")
def predict_current(request: CVRequest):
    input_data = np.array(request.features)
    scaled_data = scaler.transform(input_data)
    
    prediction = meta_model.predict(scaled_data)
    feature_weights = rf_model.feature_importances_.tolist()
    
    return {
        "predicted_current": prediction.tolist(),
        "feature_importance": feature_weights
    }
