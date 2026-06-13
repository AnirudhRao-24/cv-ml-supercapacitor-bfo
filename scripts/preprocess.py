import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Setup relative paths based on your repository structure
os.makedirs('../models', exist_ok=True)
DATA_PATH = '../data/CV_DATASET.xlsx' # Update extension to .csv if applicable

print("Loading raw dataset...")
df = pd.read_excel(DATA_PATH)

# Separate features and target
target_col = 'Current'
y = df[target_col]
X = df.drop(target_col, axis=1)

print("Splitting and Scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the scaler ONLY on the training data to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Saving processed data and scaler...")
joblib.dump(scaler, '../models/standard_scaler.pkl')

# Save the split data temporarily so the training scripts can use it
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), '../data/processed_data.pkl')
print("✅ Preprocessing complete.")