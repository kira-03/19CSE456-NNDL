import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Activation Functions
def step_function(x):
    return np.where(x > 0, 1, 0)

def linear_function(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softplus(x):
    return np.log1p(np.exp(x))

def swish(x):
    return x * sigmoid(x)

def softsign(x):
    return x / (1 + np.abs(x))

# Derivative Functions
def step_function_derivative(x):
    return np.zeros_like(x)

def linear_function_derivative(x):
    return np.ones_like(x)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def softplus_derivative(x):
    return sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def softsign_derivative(x):
    return 1 / (1 + np.abs(x))**2

x_values = np.linspace(-10, 10, 400)

# Apply functions and compute derivatives
def apply_functions(x):
    return {
        'Step': step_function(x),
        'Linear': linear_function(x),
        'Sigmoid': sigmoid(x),
        'Tanh': tanh(x),
        'ReLU': relu(x),
        'Leaky ReLU': leaky_relu(x),
        'ELU': elu(x),
        'Softplus': softplus(x),
        'Swish': swish(x),
        'Softsign': softsign(x)
    }

def compute_derivatives(x):
    return {
        'Step': step_function_derivative(x),
        'Linear': linear_function_derivative(x),
        'Sigmoid': sigmoid_derivative(x),
        'Tanh': tanh_derivative(x),
        'ReLU': relu_derivative(x),
        'Leaky ReLU': leaky_relu_derivative(x),
        'ELU': elu_derivative(x),
        'Softplus': softplus_derivative(x),
        'Swish': swish_derivative(x),
        'Softsign': softsign_derivative(x)
    }

activation_outputs = apply_functions(x_values)
activation_derivatives = compute_derivatives(x_values)

# Create subplots with overlapping activation functions and derivatives
def plot_activation_and_derivative(activation_outputs, activation_derivatives):
    names = list(activation_outputs.keys())
    num_plots = len(names)
    
    for i in range(num_plots):
        fig = go.Figure()
        
        # Plot activation function
        fig.add_trace(
            go.Scatter(x=x_values, y=activation_outputs[names[i]], mode='lines', name=f'{names[i]} Function', line=dict(color='blue'))
        )
        
        # Plot derivative
        fig.add_trace(
            go.Scatter(x=x_values, y=activation_derivatives[names[i]], mode='lines', name=f'{names[i]} Derivative', line=dict(color='red', dash='dash', width=4))
        )
        
        fig.update_layout(
            title=f'{names[i]} Function and Derivative',
            xaxis_title='x',
            yaxis_title='y',
            legend_title='Legend',
            showlegend=True
        )
        
        fig.show()

# Plot activation functions and their derivatives
plot_activation_and_derivative(activation_outputs, activation_derivatives)
