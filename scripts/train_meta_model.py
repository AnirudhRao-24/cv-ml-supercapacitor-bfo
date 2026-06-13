import joblib
import pandas as pd
from sklearn.linear_model import RidgeCV
from tensorflow.keras.models import load_model

print("Loading validation data and base models...")
_, X_test_scaled, _, y_test = joblib.load('../data/processed_data.pkl')

ann_model = load_model('../models/ann_model.h5')
rf_model = joblib.load('../models/rf_model.pkl')
xgb_model = joblib.load('../models/xgb_model.pkl')

print("Generating base model predictions for Meta-Model stacking...")
pred_ann = ann_model.predict(X_test_scaled, verbose=0).flatten()
pred_rf = rf_model.predict(X_test_scaled)
pred_xgb = xgb_model.predict(X_test_scaled)

# Create the stacked feature matrix
stacked_features = pd.DataFrame({
    'ANN': pred_ann,
    'RF': pred_rf,
    'XGB': pred_xgb
})

print("Training RidgeCV Meta-Model...")
meta_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
meta_model.fit(stacked_features, y_test)

print("Saving Meta-Model to /models directory...")
joblib.dump(meta_model, '../models/meta_stacked_model.pkl')
print(f"✅ Meta-Model Training complete. (Optimal Alpha chosen: {meta_model.alpha_})")