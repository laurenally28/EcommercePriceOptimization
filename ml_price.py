import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def predict_price(X_price, y_price):
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