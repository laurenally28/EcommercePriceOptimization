from helpers import load_data, prepare_data, relu, drelu, tanh, dtanh
from neural_net import NeuralNetwork
from analysis import evaluate_model_all, plot_actual_vs_predicted, plot_residuals, plot_error_histogram, plot_learning_curves, plot_sensitivity, sensitivity_analysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def main():
    #filename = '/Users/laurenally/Desktop/ML 2/amz_us_price_prediction_dataset.csv'
    filename = '/Users/seanmerrkle/Desktop/Courses/DS4420/amz_us_price_prediction_dataset.csv'
    
    df = load_data(filename)
    
    # Randomly sample from very large dataset
    df = df.sample(n=16000, random_state=42)

    # Prepare the data
    objective = "P" 
    X, y = prepare_data(df, objective)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale feature data
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    # Scale target data
    # Reshape y to 2D array, then flatten back later
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    input_size = X_train.shape[1]
    hidden_size1 = 30
    hidden_size2 = 15
    output_size = 1
    learning_rate = 0.01
    epochs = 10000
    
    nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size, relu, drelu)
    loss_results = nn.train(X_train, y_train, epochs, learning_rate, X_val=X_test, y_val=y_test)
    predictions = nn.predict(X_test)
    
    # Inverse-transform the predictions back to original scale for comparison
    predictions_unscaled = scaler_y.inverse_transform(predictions)
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    print("DONE -- Predictions generated, Analyzing results...\n")

    # Evaluate model
    extended_metrics = evaluate_model_all(y_test_unscaled, predictions_unscaled)
    print("Extended Metrics:\n", extended_metrics)
    
    # Plot Actual vs Predicted, Residual Plot, and Error Histogram
    plot_actual_vs_predicted(y_test_unscaled, predictions_unscaled)
    plot_residuals(y_test_unscaled, predictions_unscaled)
    plot_error_histogram(y_test_unscaled, predictions_unscaled)
    plot_learning_curves(loss_results['train_loss'], loss_results.get('val_loss'))
    feature_names = ["stars", "reviews", "price", "isBestSeller", "cateogory_encoded", "title_sbert", "title_tfidf"]
    sens = sensitivity_analysis(nn, X_test, feature_names)
    print("Feature Sensitivities:", sens)
    plot_sensitivity(sens)

if __name__ == "__main__":
    main()

