import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - tanh(x)**2

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

# Define the range for the input values
x_range = np.linspace(-2.5, 2.5, 200)

# Initialize the plot with a professional style
sns.set(style="whitegrid", context='talk', palette='muted', color_codes=True, rc={'figure.figsize': (10, 6)})
plt.figure()

# Plot the Sigmoid function and its derivative
plt.plot(x_range, sigmoid(x_range), label='Sigmoid', color='b', linewidth=2)
plt.plot(x_range, sigmoid_derivative(x_range), label='Sigmoid Derivative', color='b', linestyle='--', linewidth=2)

# Plot the ReLU function and its derivative
plt.plot(x_range, relu(x_range), label='ReLU', color='r', linewidth=2)
plt.plot(x_range, relu_derivative(x_range), label='ReLU Derivative', color='r', linestyle='--', linewidth=2)

# Plot the Tanh function and its derivative
plt.plot(x_range, tanh(x_range), label='Tanh', color='g', linewidth=2)
plt.plot(x_range, tanh_derivative(x_range), label='Tanh Derivative', color='g', linestyle='--', linewidth=2)

# Plot the Softmax function (simplified for demonstration purposes)
plt.plot(x_range, [softmax(np.array([i]))[0] for i in x_range], label='Softmax', color='m', linewidth=2)
plt.plot(x_range, [softmax_derivative(np.array([i]))[0] for i in x_range], label='Softmax Derivative', color='m', linestyle='--', linewidth=2)

# Customize the plot with labels, title, and legend
plt.title('Activation Functions and Their Derivatives')
plt.xlabel('Input value (x)')
plt.ylabel('Output value')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
