import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model_all(y_true, y_pred):
    """
    Evaluate the regression model using multiple metrics
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
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    
    # Plot diagonal line for reference
    min_val = np.min([np.min(y_true), np.min(y_pred)])
    max_val = np.max([np.max(y_true), np.max(y_pred)])
    plt.plot([min_val, max_val], [min_val, max_val], color='red', lw=2)
    filename = plt.gca().get_title().replace(" ", "_") + ".png"
    plt.savefig(filename)
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
    filename = plt.gca().get_title().replace(" ", "_") + ".png"
    plt.savefig(filename)
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
    filename = plt.gca().get_title().replace(" ", "_") + ".png"
    plt.savefig(filename)
    plt.show()

def plot_learning_curves(train_losses, val_losses):
    """
    Plot training and validation loss curves over epochs
    """
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    filename = plt.gca().get_title().replace(" ", "_") + ".png"
    plt.savefig(filename)
    plt.show()

def sensitivity_analysis(model, X, feature_names, epsilon=1e-4):
    """
    Conduct a simple sensitivity analysis by perturbing each feature
    For each feature, the function perturbs that feature by a small amount (epsilon) and evaluates the change in model predictions
    """
    base_preds = model.predict(X)
    sensitivities = {}
    
    for i, feature in enumerate(feature_names):
        X_perturbed = X.copy()
        X_perturbed[:, i] += epsilon
        perturbed_preds = model.predict(X_perturbed)
        # Estimate sensitivity: average change in prediction per unit change in the feature.
        sensitivity = np.mean((perturbed_preds - base_preds) / epsilon)
        sensitivities[feature] = sensitivity
        
    return sensitivities


def plot_sensitivity(sensitivities):
    """
    Plot a bar chart of sensitivities for each feature
    """
    features = list(sensitivities.keys())
    values = list(sensitivities.values())
    
    plt.figure(figsize=(8,6))
    plt.bar(features, values, color='skyblue')
    
    # Draw a horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='--')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    
    # Add labels and title
    plt.xlabel("Feature")
    plt.ylabel("Average Sensitivity")
    plt.title("Feature Sensitivity Analysis")
    plt.tight_layout()
    
    filename = plt.gca().get_title().replace(" ", "_") + ".png"
    plt.savefig(filename)
    plt.show()
