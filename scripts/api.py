from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the API
app = FastAPI()

# Allow your local HTML file to communicate with this local API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your pre-trained models
print("Loading models into memory...")
scaler = joblib.load('../models/standard_scaler.pkl')
meta_model = joblib.load('../models/meta_stacked_model.pkl')
rf_model = joblib.load('../models/rf_model.pkl') # For feature importance

# Define the expected incoming data structure
class CVRequest(BaseModel):
    features: list  # e.g., [[potential, scan_rate, redox_state, dopant_1, dopant_2, dopant_3]]

@app.post("/predict")
def predict_current(request: CVRequest):
    # 1. Scale the incoming data
    input_data = np.array(request.features)
    scaled_data = scaler.transform(input_data)
    
    # 2. Run the prediction (Assuming meta_model directly accepts the scaled features for this example)
    # If your meta-model requires base-model predictions first, you would generate those here.
    prediction = meta_model.predict(scaled_data)
    
    # 3. Extract Feature Importance for the XAI visualizer
    feature_weights = rf_model.feature_importances_.tolist()
    
    # 4. Return the data to the JavaScript frontend
    return {
        "predicted_current": prediction.tolist(),
        "feature_importance": feature_weights
    }