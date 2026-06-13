import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

    # 2. Build Artificial Neural Network (ANN)
    print("Building Artificial Neural Network (ANN)...")
    ann_model = Sequential()
    ann_model.add(Dense(100, activation='relu', input_dim=X_train_scaled.shape[1]))
    ann_model.add(Dense(80, activation='relu'))
    ann_model.add(Dense(1, activation='linear'))

    ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 3. Train the Model
    print("Training started. This may take a minute or two...\n")
    ann_model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # 4. Evaluate the Model
    y_pred_ann = ann_model.predict(X_test_scaled)
    r2_ann = r2_score(y_test, y_pred_ann)
    rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))

    print("\n✅ ANN Training Complete!")
    print(f"ANN R² Score: {r2_ann:.4f} (Aiming for >0.97)")
    print(f"ANN RMSE:     {rmse_ann:.6f}")

    # 5. Export Model
    ann_path = os.path.join(models_dir, 'ann_model.keras')
    ann_model.save(ann_path)
    print(f"Saved ANN to: {ann_path}")

if __name__ == "__main__":
    main()