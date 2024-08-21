import numpy as np
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MLP:
    def __init__(self, config):
        self.layers = config['layers']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.weights = []
        self.biases = []
        self.activations = []
        self.initialize_network()

    def initialize_network(self):
        for i in range(len(self.layers) - 1):
            input_neurons = self.layers[i]['neurons']
            output_neurons = self.layers[i + 1]['neurons']
            weight_matrix = np.random.rand(input_neurons, output_neurons) - 0.5
            bias_vector = np.random.rand(1, output_neurons) - 0.5
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

            activation = self.layers[i + 1].get('activation', 'sigmoid')
            self.activations.append(activation)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward_propagation(self, X):
        self.a = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            if self.activations[i] == 'sigmoid':
                a = self.sigmoid(z)
            elif self.activations[i] == 'relu':
                a = self.relu(z)
            elif self.activations[i] == 'tanh':
                a = self.tanh(z)
            elif self.activations[i] == 'softmax':
                a = self.softmax(z)
            self.a.append(a)
        return self.a[-1]

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward_propagation(self, X, y):
        m = y.shape[0]
        deltas = []
        output_activation = self.a[-1]
        
        if self.activations[-1] == 'sigmoid':
            delta = (output_activation - y) * self.sigmoid_derivative(output_activation)
        elif self.activations[-1] == 'relu':
            delta = (output_activation - y) * self.relu_derivative(output_activation)
        elif self.activations[-1] == 'tanh':
            delta = (output_activation - y) * self.tanh_derivative(output_activation)
        elif self.activations[-1] == 'softmax':
            delta = output_activation - y
        
        deltas.append(delta)

        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[-1], self.weights[i + 1].T) * self.sigmoid_derivative(self.a[i + 1])
            deltas.append(delta)

        deltas.reverse()
        return deltas

    def update_weights(self, X, y):
        m = y.shape[0]
        deltas = self.backward_propagation(X, y)
        
        for i in range(len(self.weights)):
            self.weights[i] -= (1 / m) * self.learning_rate * np.dot(self.a[i].T, deltas[i])
            self.biases[i] -= (1 / m) * self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, X, y):
        for epoch in range(self.epochs):
            self.forward_propagation(X)
            loss = self.mean_squared_error(y, self.a[-1])
            self.update_weights(X, y)
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        output = self.forward_propagation(X)
        return np.argmax(output, axis=1)

def parse_config(json_string):
    return json.loads(json_string)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Example usage
if __name__ == "__main__":
    config_json = '''
    {
        "layers": [
            {
                "type": "input",
                "neurons": 4
            },
            {
                "type": "hidden",
                "neurons": 4,
                "activation": "sigmoid"
            },
            {
                "type": "output",
                "neurons": 2,
                "activation": "softmax"
            }
        ],
        "learning_rate": 0.1,
        "epochs": 10000,
        "batch_size": 32
    }
    '''
    config = parse_config(config_json)
    mlp = MLP(config)

    # Dummy data for training
    X_train = np.random.rand(100, 4)  # 100 samples, 4 features
    y_train = np.random.randint(0, 2, (100, 2))  # 100 samples, 2 classes (one-hot encoded)

    mlp.train(X_train, y_train)
    predictions = mlp.predict(X_train)

    print("Predictions:", predictions)

    # Plot confusion matrix
    y_true = np.argmax(y_train, axis=1)  # Convert one-hot to class labels
    plot_confusion_matrix(y_true, predictions)
