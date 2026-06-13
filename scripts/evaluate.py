import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import load_model

print("Loading completely unseen 60mV Dataset...")
# Adjust filename to match exactly what is in your /data folder
unseen_file_path = '../data/60MV_CV (1).xlsx' 
df_unseen = pd.read_excel(unseen_file_path)

target_col = 'Current'
y_unseen = df_unseen[target_col]
X_unseen = df_unseen.drop(target_col, axis=1)

print("Loading Models and Scaler...")
scaler = joblib.load('../models/standard_scaler.pkl')
ann_model = load_model('../models/ann_model.h5')
rf_model = joblib.load('../models/rf_model.pkl')
xgb_model = joblib.load('../models/xgb_model.pkl')
meta_model = joblib.load('../models/meta_stacked_model.pkl')

print("Scaling Unseen Data...")
X_unseen_scaled = scaler.transform(X_unseen)

print("Generating Base Model Predictions...")
unseen_pred_ann = ann_model.predict(X_unseen_scaled, verbose=0).flatten()
unseen_pred_rf = rf_model.predict(X_unseen_scaled)
unseen_pred_xgb = xgb_model.predict(X_unseen_scaled)

X_meta_unseen = pd.DataFrame({
    'ANN': unseen_pred_ann,
    'RF': unseen_pred_rf,
    'XGB': unseen_pred_xgb
})

print("Generating Final Meta-Model Prediction...")
final_predictions = meta_model.predict(X_meta_unseen)

final_r2 = r2_score(y_unseen, final_predictions)
final_rmse = np.sqrt(mean_squared_error(y_unseen, final_predictions))

print("\n" + "="*40)
print("       REAL WORLD TEST RESULTS")
print("="*40)
print(f"Unseen Data R² Score: {final_r2:.4f}")
print(f"Unseen Data RMSE:     {final_rmse:.6f}")
print("="*40 + "\n")