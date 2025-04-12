import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation, activation_deriv):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
        self.b1 = np.random.uniform(-0.5, 0.5, (1, hidden_size))
        self.W2 = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, output_size))
        
        self.activation = activation
        self.activation_deriv = activation_deriv
        
    def forward(self, X):
        """
        Perform forward propagation
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Output layer --> linear
        return self.z2
        
    def backward(self, X, y, output, learning_rate):
        """
        Perform backpropagation and update weights
        """
        m = X.shape[0]
        y = y.reshape(-1, 1)
        
        # Compute gradient of mean squared error loss
        dLoss_output = (output - y) / m
        
        # Gradients for second layer
        dW2 = np.dot(self.a1.T, dLoss_output)
        db2 = np.sum(dLoss_output, axis=0, keepdims=True)
        
        # Backpropagate into hidden layer
        dA1 = np.dot(dLoss_output, self.W2.T)
        dZ1 = dA1 * self.activation_deriv(self.z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
    def train(self, X, y, epochs, learning_rate):
        """
        Train the neural network
        """
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean(0.5 * (output - y.reshape(-1, 1))**2)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss:.4f}")
                
    def predict(self, X):
        """
        Generate predictions for given inputs
        """
        return self.forward(X)
