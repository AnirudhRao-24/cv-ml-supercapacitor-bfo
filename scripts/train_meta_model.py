import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error

def main():
    data_dir = 'data/'
    models_dir = 'models/'

    # 1. Load Preprocessed Data
    print("Loading preprocessed data...")
    X_train_scaled = np.load(os.path.join(data_dir, 'X_train_scaled.npy'))
    X_test_scaled = np.load(os.path.join(data_dir, 'X_test_scaled.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # 2. Load Base Models
    print("Loading Base Models...")
    ann_model = load_model(os.path.join(models_dir, 'ann_model.keras'))
    rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
    xgb_model = joblib.load(os.path.join(models_dir, 'xgb_model.pkl'))

    print("Building the Stacked Meta-Model...")

    # 3. Generate Predictions from the Training Data
    train_pred_ann = ann_model.predict(X_train_scaled, verbose=0).flatten()
    train_pred_rf = rf_model.predict(X_train_scaled)
    train_pred_xgb = xgb_model.predict(X_train_scaled)

    X_meta_train = pd.DataFrame({
        'ANN': train_pred_ann,
        'RF': train_pred_rf,
        'XGB': train_pred_xgb
    })

    # 4. Generate Predictions from the Testing Data
    test_pred_ann = ann_model.predict(X_test_scaled, verbose=0).flatten()
    test_pred_rf = rf_model.predict(X_test_scaled)
    test_pred_xgb = xgb_model.predict(X_test_scaled)

    X_meta_test = pd.DataFrame({
        'ANN': test_pred_ann,
        'RF': test_pred_rf,
        'XGB': test_pred_xgb
    })

    # 5. Train the Meta-Model (Ridge Regression)
    meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    meta_model.fit(X_meta_train, y_train)

    # 6. Final Evaluation
    y_pred_meta = meta_model.predict(X_meta_test)
    r2_meta = r2_score(y_test, y_pred_meta)
    rmse_meta = np.sqrt(mean_squared_error(y_test, y_pred_meta))

    print("\n✅ Meta-Model Training Complete!")
    print("--- FINAL ACCURACY ---")
    print(f"Meta-Model R² Score: {r2_meta:.4f}")
    print(f"Meta-Model RMSE:     {rmse_meta:.6f}")
    print(f"Ridge Alpha chosen:  {meta_model.alpha_}")

    # 7. Save Meta-Model
    joblib.dump(meta_model, os.path.join(models_dir, 'meta_stacked_model.pkl'))
    print(f"Saved Meta-Model to {models_dir}meta_stacked_model.pkl")

if __name__ == "__main__":
    main()