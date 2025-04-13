import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation, activation_deriv, lambda_reg=0.001):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size1))
        self.b1 = np.random.uniform(-0.5, 0.5, (1, hidden_size1))
        
        self.W2 = np.random.uniform(-0.5, 0.5, (hidden_size1, hidden_size2))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, hidden_size2))
        
        self.W3 = np.random.uniform(-0.5, 0.5, (hidden_size2, output_size))
        self.b3 = np.random.uniform(-0.5, 0.5, (1, output_size))
        
        self.activation = activation
        self.activation_deriv = activation_deriv
        
        # Regularization parameter
        self.lambda_reg = lambda_reg

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        output = self.z3
        return output

    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        dLoss_output = (output - y) / m
        
        # Gradients for output layer
        dW3 = np.dot(self.a2.T, dLoss_output)
        db3 = np.sum(dLoss_output, axis=0, keepdims=True)

        # Add L2 regularization term for W3
        dW3 += self.lambda_reg * self.W3
        
        # Backpropagate into hidden layer 2
        dA2 = np.dot(dLoss_output, self.W3.T)
        dZ2 = dA2 * self.activation_deriv(self.z2)
        dW2 = np.dot(self.a1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Add L2 regularization term for W2
        dW2 += self.lambda_reg * self.W2
        
        # Backpropagate into hidden layer 1
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_deriv(self.z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Add L2 regularization term for W1
        dW1 += self.lambda_reg * self.W1
        
        # Update weights and biases
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate, X_val=None, y_val=None):
        """
        Train the neural network over a given number of epochs
        If validation data (X_val and y_val) is provided, the method also logs validation loss
        """
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean(0.5 * (output - y.reshape(-1, 1)) ** 2)
            train_losses.append(loss)
            self.backward(X, y, output, learning_rate)
            
            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val)
                val_loss = np.mean(0.5 * (val_output - y_val.reshape(-1, 1)) ** 2)
                val_losses.append(val_loss)
            
            if epoch % 100 == 0:
                if X_val is not None and y_val is not None:
                    print(f"Epoch: {epoch}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch: {epoch}, Train Loss: {loss:.4f}")
                    
        results = {'train_loss': train_losses}
        if X_val is not None and y_val is not None:
            results['val_loss'] = val_losses
        return results


    def predict(self, X):
        return self.forward(X)
