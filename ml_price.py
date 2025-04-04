import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def predict_units(X_units, y_units):
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
