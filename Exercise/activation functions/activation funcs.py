import numpy as np
import matplotlib.pyplot as plt

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



def plot_in_batches(data_dict, title_prefix, batch_size=4):
    num_plots = len(data_dict)
    for start in range(0, num_plots, batch_size):
        end = min(start + batch_size, num_plots)
        batch_names = list(data_dict.keys())[start:end]
        
        fig, axs = plt.subplots(len(batch_names), 1, figsize=(10, 10))
        if len(batch_names) == 1:
            axs = [axs]
        
        for i, name in enumerate(batch_names):
            axs[i].plot(x_values, data_dict[name])
            axs[i].set_title(f'{title_prefix} - {name}')
            axs[i].grid(True)
        
        plt.tight_layout()
        plt.show()

# Plot activation functions in batches of 4
plot_in_batches(activation_outputs, 'Activation Function')

# Plot derivatives in batches of 4
plot_in_batches(activation_derivatives, 'Derivative')