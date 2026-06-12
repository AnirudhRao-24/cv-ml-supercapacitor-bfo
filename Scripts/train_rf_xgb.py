import joblib
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

print("Loading processed training data...")
X_train_scaled, X_test_scaled, y_train, y_test = joblib.load('../data/processed_data.pkl')

print("Training Random Forest Ensembles...")
rf_model = RandomForestRegressor(n_estimators=40, max_depth=11, random_state=42)
rf_model.fit(X_train_scaled, y_train)
joblib.dump(rf_model, '../models/rf_model.pkl')
print("Saved Random Forest.")

print("Training XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
joblib.dump(xgb_model, '../models/xgb_model.pkl')
print("Saved XGBoost.")

print("✅ Traditional Models Training complete.")