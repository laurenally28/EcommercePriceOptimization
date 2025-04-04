import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from ml_units import predict_units
from ml_price import predict_price

# Read in data
df = pd.read_csv('amz_us_price_prediction_dataset.csv')  # Update path if needed
df = df.dropna()
df['isBestSeller'] = df['isBestSeller'].astype(bool)
features = ['stars', 'reviews', 'price', 'category', 'isBestSeller']

# Inputs for predict units sold
X_units = df[features]
y_units = df['boughtInLastMonth']

# Input for predict optimal price
X_price = df[features].copy()
X_price['boughtInLastMonth'] = df['boughtInLastMonth']
y_price = df['price']

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['stars', 'reviews', 'price', 'boughtInLastMonth']),
    ('cat', OneHotEncoder(), ['category', 'isBestSeller'])
], remainder='drop')

predict_units(X_units, y_units)
predict_price(X_price, y_price)