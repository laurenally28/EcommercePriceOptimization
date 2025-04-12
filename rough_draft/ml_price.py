import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

def predict_price(X_price, y_price, preprocessor):
    X2_train, X2_test, y2_train, y2_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

    X2_train_proc = preprocessor.fit_transform(X2_train)
    X2_test_proc = preprocessor.transform(X2_test)

    model_price = Sequential([
        Dense(64, activation='relu', input_shape=(X2_train_proc.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model_price.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model_price.fit(X2_train_proc, y2_train, epochs=50, batch_size=32, validation_split=0.2)

def predict_price_manually(X_price, y_price):
    # Train Test Split
    X2_train, X2_test, y2_train, y2_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

    # Convert to numpy
    #X2_train = X2_train.to_numpy()
    y2_train = y2_train.reshape(-1, 1)

    # Hyperparameters
    lr = 0.001
    epochs = 500
    n, m = X2_train.shape

    # Initialize weights + bias
    W = np.random.randn(m, 1)
    b = np.zeros((1,))

    for epoch in range(epochs):
        # Forward pass
        y_pred = X2_train @ W + b

        # Loss: MSE
        loss = np.mean((y2_train - y_pred) ** 2)

        # Gradients
        dW = (-2 / n) * X2_train.T @ (y2_train - y_pred)
        db = (-2 / n) * np.sum(y2_train - y_pred)

        # Update weights
        W -= lr * dW
        b -= lr * db

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return W, b
