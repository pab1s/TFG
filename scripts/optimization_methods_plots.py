import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Rosenbrock function
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b*(y - x**2)**2

# Define the gradient of the Rosenbrock function
def rosenbrock_grad(x, y, a=1, b=100):
    grad_x = -2*(a - x) - 4*b*x*(y - x**2)
    grad_y = 2*b*(y - x**2)
    return np.array([grad_x, grad_y], dtype=float)

# Create a meshgrid for the 3D surface plot
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# Initialize the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Set the boundaries of the plot to match the domain exactly
ax.set_xlim([x.min(), x.max()])
ax.set_ylim([y.min(), y.max()])
ax.set_zlim([Z.min(), Z.max()])

# Initialize the starting point for the optimizations
initial_position = np.array([0, 2.5], dtype=float)

# Set optimization parameters
steps = 1000
alpha = 0.001
momentum_value = 0.2
velocity = np.zeros(2, dtype=float)
beta1, beta2 = 0.9, 0.999
m, v = np.zeros(2, dtype=float), np.zeros(2, dtype=float)
epsilon = 1e-8

# Lists to hold paths of each optimizer
path_sgd = [initial_position.copy()]
path_momentum = [initial_position.copy()]
path_adam = [initial_position.copy()]

# Simulation of optimization paths
for i in range(steps):
    current_position = path_sgd[-1]
    grad_current = rosenbrock_grad(current_position[0], current_position[1])
    
    # Update positions using SGD
    new_position = current_position - alpha * grad_current
    path_sgd.append(new_position.copy())
    
    # Update positions using Momentum
    current_position = path_momentum[-1]
    velocity = momentum_value * velocity + alpha * grad_current
    new_position = current_position - velocity
    path_momentum.append(new_position.copy())
    
    # Update positions using Adam
    current_position = path_adam[-1]
    m = beta1 * m + (1 - beta1) * grad_current
    v = beta2 * v + (1 - beta2) * (grad_current**2)
    m_hat = m / (1 - beta1**(i + 1))
    v_hat = v / (1 - beta2**(i + 1))
    new_position = current_position - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    path_adam.append(new_position.copy())

# Convert path lists to arrays for plotting
path_sgd = np.array(path_sgd)
path_momentum = np.array(path_momentum)
path_adam = np.array(path_adam)

# Plot the paths on the 3D surface
ax.plot(path_sgd[:, 0], path_sgd[:, 1], rosenbrock(path_sgd[:, 0], path_sgd[:, 1]), 'r.-', label='SGD')
ax.plot(path_momentum[:, 0], path_momentum[:, 1], rosenbrock(path_momentum[:, 0], path_momentum[:, 1]), 'b.-', label='Momentum')
ax.plot(path_adam[:, 0], path_adam[:, 1], rosenbrock(path_adam[:, 0], path_adam[:, 1]), 'g.-', label='Adam')

# Setting labels and showing the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
