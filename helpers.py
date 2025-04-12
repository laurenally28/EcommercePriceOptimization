import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def load_data(filename):
    """
    Load a CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(filename)
    return df

def relu(x):
    """
    ReLU activation function
    """
    return np.maximum(0, x)

def drelu(x):
    """
    Derivative of the ReLU activation function.
    """
    return (x > 0).astype(float)

def tanh(x):
    """
    Hyperbolic tangent activation function
    """
    return np.tanh(x)

def dtanh(x):
    """
    Derivative of the tanh activation function
    """
    return 1 - np.tanh(x) ** 2

def prepare_data(df):
    """
    Prepare input features and target values from the DataFrame.
    Converts columns to appropriate data types and returns NumPy arrays.
    """
    # Convert numeric columns to float to be sure of correct type
    df['stars'] = df['stars'].astype(float)
    df['reviews'] = df['reviews'].astype(float)
    df['price'] = df['price'].astype(float)
    df['isBestSeller'] = df['isBestSeller'].astype(bool)
    
    # Convert target column to float
    df['boughtInLastMonth'] = df['boughtInLastMonth'].astype(float)
    
    # One-hot encode the 'category' column.
    ohe = OneHotEncoder(sparse_output=False)
    category_encoded = ohe.fit_transform(df[['category']])
    
    # Extract numeric features.
    numeric_features = df[['stars', 'reviews', 'price', 'isBestSeller']].to_numpy().astype(np.float64)
    
    # Combine numeric features and one-hot encoded category features.
    X = np.hstack((numeric_features, category_encoded)).astype(np.float64)
    y = df['boughtInLastMonth'].to_numpy().astype(np.float64)
    
    return X, y