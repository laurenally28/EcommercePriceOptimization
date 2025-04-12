import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def evaluate_model_all(y_true, y_pred):
    """
    Evaluate the regression model using multiple metrics: 
        Mean Squared Error (MSE), Root Mean Squared Error (RMSE), 
        Mean Absolute Error (MAE), and R squared score
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plot actual vs. predicted values
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    # Plot a diagonal line for reference
    min_val = np.min([np.min(y_true), np.min(y_pred)])
    max_val = np.max([np.max(y_true), np.max(y_pred)])
    plt.plot([min_val, max_val], [min_val, max_val], color='red', lw=2)
    plt.show()

def plot_residuals(y_true, y_pred):
    """
    Plot the residuals (differences between actual and predicted values) against predicted values
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.axhline(0, color='red', linestyle='--')
    plt.show()

def plot_error_histogram(y_true, y_pred):
    """
    Plot a histogram of the residuals (error distribution)
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.show()
