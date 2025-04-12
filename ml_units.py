import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

def predict_units(X_units, y_units, preprocessor):
    X1_train, X1_test, y1_train, y1_test = train_test_split(X_units, y_units, test_size=0.2, random_state=42)

    X1_train_proc = preprocessor.fit_transform(X1_train)
    X1_test_proc = preprocessor.transform(X1_test)

    model_units = Sequential([
        Dense(64, activation='relu', input_shape=(X1_train_proc.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model_units.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model_units.fit(X1_train_proc, y1_train, epochs=50, batch_size=32, validation_split=0.2)

def predict_units_manually(X_units, y_units):
    # Train Test Split
    X1_train, X1_test, y1_train, y1_test = train_test_split(X_units, y_units, test_size=0.2, random_state=42)

    # Convert to numpy
    #X1_train = X1_train.to_numpy()
    y1_train = y1_train.reshape(-1, 1)

    # Hyperparameters
    lr = 0.001
    epochs = 500
    n, m = X1_train.shape

    # Initialize weights + bias
    W = np.random.randn(m, 1)
    b = np.zeros((1,))

    for epoch in range(epochs):

        # Forward pass
        y_pred = X1_train @ W + b

        # Loss: MSE
        loss = np.mean((y1_train - y_pred) ** 2)

        # Gradients
        dW = (-2 / n) * X1_train.T @ (y1_train - y_pred)
        db = (-2 / n) * np.sum(y1_train - y_pred)

        # Update weights
        W -= lr * dW
        b -= lr * db

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return W, b