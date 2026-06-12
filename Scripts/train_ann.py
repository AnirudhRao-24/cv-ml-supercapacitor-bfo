import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("Loading processed training data...")
X_train_scaled, X_test_scaled, y_train, y_test = joblib.load('../data/processed_data.pkl')

print("Building and training Artificial Neural Network...")
ann_model = Sequential([
    Dense(100, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(80, activation='relu'),
    Dense(1, activation='linear')
])

ann_model.compile(optimizer='adam', loss='mse')

# Train the model
ann_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

print("Saving ANN model to /models directory...")
ann_model.save('../models/ann_model.h5')
print("✅ ANN Training complete.")