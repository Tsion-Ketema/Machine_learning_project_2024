import numpy as np

# ========================== Activation Functions & Derivatives ==========================


def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def leaky_relu_derivative(x, alpha=0.01): return np.where(x > 0, 1, alpha)


def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x) ** 2


def sigmoid(x):
    """Numerically stable sigmoid function to avoid overflow."""
    x = np.clip(x, -50, 50)  # Clip values to prevent overflow
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def identity(x): return x
def identity_derivative(x): return np.ones_like(x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)  # Simplified version (not the full Jacobian)


# Activation mapping dictionary for easy retrieval
activation_map = {
    'relu': (relu, relu_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'identity': (identity, identity_derivative),
    'softmax': (softmax, softmax_derivative),
}

# ========================== Loss Functions & Derivatives ==========================


def mse_loss(y_true, y_pred):
    """Mean Squared Error Loss."""
    return np.mean((y_true - y_pred) ** 2)


def mse_loss_derivative(y_true, y_pred):
    """Derivative of MSE Loss."""
    return 2 * (y_pred - y_true) / y_true.size
