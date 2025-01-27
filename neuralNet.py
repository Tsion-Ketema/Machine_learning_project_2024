import numpy as np
import os
from functions import activation_map


class NeuralNetwork:
    def __init__(self, hidden_layer_sizes, learning_rate, epochs, momentum,
                 weight_initialization, regularization=None, activations=None, task_type='classification'):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum
        self.weight_initialization = weight_initialization
        self.task_type = task_type
        self.regularization = regularization or (
            'none', 0)  # Default no regularization

        self.activations = self._initialize_activations(activations)
        self.activation_functions, self.derivative_functions = self._get_activation_functions()

        self._initialize_parameters()
        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = ([] if self.task_type == 'classification' else None,
                                                      [] if self.task_type == 'classification' else None)

    def _initialize_activations(self, activations):
        """Initialize activations based on task type if not provided."""
        if activations is None:
            default_activation = ['relu'] * (len(self.hidden_layer_sizes) - 2)
            return default_activation + ['sigmoid' if self.task_type == 'classification' else 'identity']
        if len(activations) != len(self.hidden_layer_sizes) - 1:
            raise ValueError(
                f"Invalid number of activations: {len(activations)} expected {len(self.hidden_layer_sizes) - 1}.")
        return activations

    def _get_activation_functions(self):
        """Retrieve activation and derivative functions."""
        activation_funcs, derivative_funcs = [], []
        for act in self.activations:
            act_func, dact_func = activation_map.get(act, (None, None))
            if act_func is None:
                raise ValueError(f"Unknown activation function: {act}")
            activation_funcs.append(act_func)
            derivative_funcs.append(dact_func)
        return activation_funcs, derivative_funcs

    def _initialize_parameters(self):
        """Initialize weights and biases using the selected method."""
        np.random.seed(42)  # Set a fixed seed for reproducibility
        self.weights, self.biases, self.velocity_w, self.velocity_b = [], [], [], []
        for i in range(len(self.hidden_layer_sizes) - 1):
            fan_in, fan_out = self.hidden_layer_sizes[i], self.hidden_layer_sizes[i + 1]
            W = np.random.randn(fan_in, fan_out) * (np.sqrt(2.0 / fan_in)
                                                    if self.weight_initialization == 'he' else np.sqrt(1.0 / fan_in))
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)
            self.velocity_w.append(np.zeros_like(W))
            self.velocity_b.append(np.zeros_like(b))

    def forward(self, X):
        """Perform forward propagation."""
        self.a_cache, self.z_cache = [X], []
        for W, b, act in zip(self.weights, self.biases, self.activation_functions):
            z = np.dot(self.a_cache[-1], W) + b
            self.z_cache.append(z)
            self.a_cache.append(act(z))
        return self.a_cache[-1]

    def compute_loss(self, y_true, y_pred):
        """Compute loss with optional L1 or L2 regularization."""
        diff = y_true - y_pred
        loss = np.mean(np.sum(diff ** 2, axis=1))  # MSE

        if self.regularization[0] == 'L1':
            reg_term = self.regularization[1] * \
                sum(np.sum(np.abs(w)) for w in self.weights)
        elif self.regularization[0] == 'L2':
            reg_term = self.regularization[1] * \
                sum(np.sum(w ** 2) for w in self.weights)
        else:
            reg_term = 0

        return loss + reg_term

    def backward(self, X, y_true):
        """Compute gradients including L1 or L2 regularization if applicable."""
        m = X.shape[0]
        y_pred = self.forward(X)
        diff = y_pred - y_true
        dLoss_dYpred = (2 * diff / m)

        grads_w, grads_b = [], []
        delta = dLoss_dYpred
        for i in reversed(range(len(self.weights))):
            reg_term = 0
            if self.regularization[0] == 'L1':
                reg_term = self.regularization[1] * np.sign(self.weights[i])
            elif self.regularization[0] == 'L2':
                reg_term = 2 * self.regularization[1] * self.weights[i]

            grads_w.insert(0, np.dot(self.a_cache[i].T, delta) + reg_term)
            grads_b.insert(0, np.sum(delta, axis=0, keepdims=True))
            if i > 0:
                delta = np.dot(
                    delta, self.weights[i].T) * self.derivative_functions[i - 1](self.z_cache[i - 1])

        return grads_w, grads_b

    def train(self, X, y, X_val=None, y_val=None):
        """Train the neural network with the provided data."""
        for epoch in range(self.epochs):
            self.forward(X)
            grads_w, grads_b = self.backward(X, y)

            for i in range(len(self.weights)):
                self.velocity_w[i] = self.momentum * \
                    self.velocity_w[i] - self.learning_rate * grads_w[i]
                self.velocity_b[i] = self.momentum * \
                    self.velocity_b[i] - self.learning_rate * grads_b[i]
                self.weights[i] += self.velocity_w[i]
                self.biases[i] += self.velocity_b[i]

            # Store train loss and accuracy
            train_loss = self.compute_loss(y, self.forward(X))
            self.train_losses.append(train_loss)

            if self.task_type == 'classification':
                train_acc = self._compute_accuracy(y, self.forward(X))
                self.train_accuracies.append(train_acc)

            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(y_val, self.forward(X_val))
                self.val_losses.append(val_loss)

                if self.task_type == 'classification':
                    val_acc = self._compute_accuracy(
                        y_val, self.forward(X_val))
                    if self.val_accuracies is not None:
                        self.val_accuracies.append(val_acc)

    def _compute_accuracy(self, y_true, y_pred):
        """Compute classification accuracy safely."""
        if y_true.shape[1] > 1:  # If one-hot encoded, convert to class labels
            predicted_labels = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_true, axis=1)
        else:
            # Binary classification thresholding
            predicted_labels = (y_pred >= 0.5).astype(int)
            true_labels = y_true.astype(int)

        return np.mean(predicted_labels == true_labels) * 100

    def evaluate(self, X, y):
        """Evaluate the model on test data."""
        return self.compute_loss(y, self.forward(X))

    def save_model(self, file_name):
        """Save the neural network model (weights, biases, and essential attributes) to an .npz file."""
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        np.savez(file_name,
                 hidden_layer_sizes=self.hidden_layer_sizes,
                 activations=self.activations,
                 learning_rate=self.learning_rate,
                 epochs=self.epochs,
                 momentum=self.momentum,
                 weight_initialization=self.weight_initialization,
                 regularization=self.regularization,
                 **{f"weight_{i}": W for i, W in enumerate(self.weights)},
                 **{f"bias_{i}": b for i, b in enumerate(self.biases)})

        print(f"[INFO] Model saved to {file_name}")

    @classmethod
    def load_model(cls, file_name):
        """Load the neural network model (weights, biases, and attributes) from an .npz file."""
        with np.load(file_name, allow_pickle=True) as data:
            hidden_layer_sizes = data['hidden_layer_sizes'].tolist()
            activations = data['activations'].tolist()
            learning_rate = float(data['learning_rate'])
            epochs = int(data['epochs'])
            momentum = float(data['momentum'])
            weight_initialization = str(data['weight_initialization'])
            regularization = float(data['regularization'])

            model = cls(hidden_layer_sizes, learning_rate, epochs, momentum,
                        weight_initialization, regularization, activations)

            model.weights = [data[f"weight_{i}"]
                             for i in range(len(hidden_layer_sizes) - 1)]
            model.biases = [data[f"bias_{i}"]
                            for i in range(len(hidden_layer_sizes) - 1)]

            print(f"[INFO] Model loaded from {file_name}")

        return model
