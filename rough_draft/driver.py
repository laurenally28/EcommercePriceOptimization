import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from ml_units import predict_units, predict_units_manually
from ml_price import predict_price, predict_price_manually

# Read data
df = pd.read_csv('/Users/seanmerrkle/Desktop/Courses/DS4420/amz_us_price_prediction_dataset.csv')
df = df.dropna()
df['isBestSeller'] = df['isBestSeller'].astype(bool)
print(df.shape)
df = df.sample(n=1000)

# One-hot encode category
ohe = OneHotEncoder(sparse_output=False)
category_encoded = ohe.fit_transform(df[['category']])

# TF-IDF on titles
tfidf = TfidfVectorizer(max_features=20)
title_tfidf = tfidf.fit_transform(df['title']).toarray()

# Create units features
X_units = np.hstack((
    df[['stars', 'reviews', 'price', 'isBestSeller']].to_numpy().astype(float),
    category_encoded,
    title_tfidf
))
y_units = df['boughtInLastMonth'].to_numpy().astype(float)

# Create price features
X_price = np.hstack((
    df[['stars', 'reviews', 'isBestSeller', 'boughtInLastMonth']].to_numpy().astype(float),
    category_encoded,
    title_tfidf
))
y_price = df['price'].to_numpy().astype(float)

# Manual Units Model
units_W, units_b = predict_units_manually(X_units, y_units)

X_new_units = np.hstack((
    np.array([[4.5, 250, 19.99, 1]]),       # stars, reviews, price, isBestSeller
    np.array([[0, 0, 1, 0, 0]]) ,           # one-hot category example (needs to match your ohe.categories_)
    np.random.rand(1, 20)                   # fake TF-IDF for new title (replace with real tfidf.transform(['some title']).toarray())
))
y_new_units = X_new_units @ units_W + units_b
print(f"Predicted Units Sold: {y_new_units.flatten()[0]}")

# Manual Price Model
price_W, price_b = predict_price_manually(X_price, y_price)

X_new_price = np.hstack((
    np.array([[4.3, 150, 1, 300]]),         # stars, reviews, isBestSeller, boughtInLastMonth
    np.array([[0, 1, 0, 0, 0]]),            # one-hot category example
    np.random.rand(1, 20)                   # fake TF-IDF for new title
))
y_new_price = X_new_price @ price_W + price_b
print(f"Predicted Optimal Price: ${y_new_price.flatten()[0]:.2f}")