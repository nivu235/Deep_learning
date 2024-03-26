import numpy as np
import matplotlib.pyplot as plt

# Generate some modified data for demonstration
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X**2 + np.random.randn(100, 1)

# Add bias term to input features
X_b = np.c_[np.ones((100, 1)), X]

# Function to compute mean squared error (MSE)
def compute_mse(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    mse = (1 / m) * np.sum((predictions - y) ** 2)
    return mse

# Function to perform gradient descent
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.random.randn(2, 1)  # Random initialization of parameters

    mse_values = []  # To store the MSE at each iteration

    for iteration in range(n_iterations):
        gradients = 2 / m * X.T.dot(X.dot(theta) - y)  # Compute gradients
        theta = theta - learning_rate * gradients  # Update parameters
        mse = compute_mse(theta, X, y)  # Compute MSE
        mse_values.append(mse)

    return theta, mse_values

# Perform gradient descent
theta, mse_values = gradient_descent(X_b, y)

# Print the optimized parameters
print("Optimized Parameters (theta):", theta)

# Plot the learning curve (MSE vs. Iterations)
plt.plot(range(len(mse_values)), mse_values)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Gradient Descent Learning Curve')
plt.show()
