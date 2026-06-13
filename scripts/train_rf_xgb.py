import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
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

    # ==========================================
    # Random Forest Model
    # ==========================================
    print("\nBuilding Random Forest (RF) Model...")
    rf_model = RandomForestRegressor(
        n_estimators=40,
        max_depth=11,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training RF started...")
    rf_model.fit(X_train_scaled, y_train)

    y_pred_rf = rf_model.predict(X_test_scaled)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    print("✅ Random Forest Training Complete!")
    print(f"RF R² Score: {r2_rf:.4f}")
    print(f"RF RMSE:     {rmse_rf:.6f}")
    
    joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))

    # ==========================================
    # XGBoost Model
    # ==========================================
    print("\nBuilding XGBoost (XGB) Model...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    print("Training XGB started...")
    xgb_model.fit(X_train_scaled, y_train)

    y_pred_xgb = xgb_model.predict(X_test_scaled)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

    print("✅ XGBoost Training Complete!")
    print(f"XGB R² Score: {r2_xgb:.4f}")
    print(f"XGB RMSE:     {rmse_xgb:.6f}")
    
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.pkl'))
    print(f"\nSaved RF and XGB models to {models_dir}")

if __name__ == "__main__":
    main()