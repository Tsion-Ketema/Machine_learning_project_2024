import json
import os
import random
from itertools import product


class HyperparameterManager:
    """Manages generation, saving, and loading of hyperparameter combinations."""

    def __init__(self):
        # Define hyperparameters for CUP and MONK datasets
        self.hyperparameters = {
            'monks-1': {
                'hidden_layer_sizes': [
                    (17, 8, 1),
                    (17, 12, 6, 1),
                    (17, 16, 8, 1)
                ],
                'activations': [
                    ('relu', 'sigmoid'),
                    ('tanh', 'sigmoid')
                ],
                'learning_rates': [
                    0.1, 0.01, 0.001
                ],
                'weight_initialization': [
                    'xavier', 'he'
                ],
                'momentum': [
                    0.9, 0.95, 0.99
                ],
                'epochs': [
                    1000
                ],
                'weight_decay': [
                    0,
                    0.00001,
                ],
            },

            'monks-2': {
                'hidden_layer_sizes': [
                    (17, 4, 1),  # One hidden layer with 4 neurons
                    (17, 8, 4, 1),  # Two hidden layers with decreasing size
                    (17, 12, 6, 1)  # Three hidden layers for deeper network
                ],
                'activations': [
                    # ReLU for hidden layers, Sigmoid for output
                    ('relu', 'sigmoid'),
                    ('tanh', 'sigmoid')
                ],
                'learning_rates': [
                    0.01,  # Standard learning rate for stable convergence
                    0.005,  # Slightly lower learning rate to avoid overshooting
                    0.001  # For finer adjustments
                ],
                'weight_initialization': [
                    'xavier',  # Xavier initialization for better weight scaling
                    'he'  # He initialization for ReLU activations
                ],
                'momentum': [
                    0.85,  # Balanced momentum for smoothing updates
                    0.9,   # Higher momentum for faster convergence
                    0.95   # Aggressive momentum
                ],
                'epochs': [
                    1000  # For experimentation
                ],
                'weight_decay': [
                    0,  # No regularization for baseline
                    0.00001,  # Minimal L2 weight decay to prevent overfitting
                    0.0001  # Slightly stronger regularization
                ],

            },
            'monks-3-no-reg': {
                'hidden_layer_sizes': [
                    (17, 15, 1),  # Simplest architecture
                    (17, 12, 6, 1),  # Slightly deeper but still simple
                    (17, 15, 10, 1)  # Moderate complexity
                ],
                'activations': [
                    ('relu', 'sigmoid'),  # Common and effective
                    ('tanh', 'sigmoid'),  # Balanced non-linearity
                    ('tanh', 'relu', 'sigmoid')  # Slightly more depth
                ],
                'learning_rates': [
                    0.001,  # For stable convergence
                    0.002,
                    0.005  # Slightly faster convergence
                ],
                'weight_initialization': [
                    'xavier',  # Balanced initialization
                    'he'  # For better handling of ReLU
                ],
                'momentum': [
                    0.85,  # Balanced momentum
                    0.9
                ],
                'epochs': [1000],  # Ample epochs for full convergence
                'weight_decay': [
                    0  # No regularization
                ],
            },
            'monks-3-l2': {
                'hidden_layer_sizes': [
                    (17, 12, 6, 1),
                    (17, 16, 8, 1),
                    (17, 24, 12, 6, 1),

                ],
                'activations': [
                    ('relu', 'sigmoid'),
                    ('tanh', 'sigmoid'),
                    ('relu', 'leaky_relu', 'sigmoid')
                ],
                'learning_rates': [
                    0.01,   # Faster learning
                    0.005,  # Balanced convergence
                    0.002,  # Lower rate for stable optimization
                    0.001,  # Fine-tuned learning
                    0.0005  # Very fine-tuned adjustments
                ],
                'weight_initialization': [
                    'he',     # Optimized for ReLU
                    'xavier'  # Balanced for tanh/sigmoid
                ],
                'momentum': [
                    0.85,  # Standard momentum for stability
                    0.9,   # High momentum for faster convergence
                    0.95   # Aggressive smoothing
                ],
                'epochs': [
                    1500   # More training for deeper models
                ],
                'regularization': [
                    ('L2', 0.0001),  # Light regularization
                    ('L2', 0.0005),  # Moderate
                    ('L2', 0.001),   # Stronger
                    ('L2', 0.0023),  # Higher
                    ('L2', 0.005)    # Strong regularization for complex models
                ],
            },
            'cup': {
                'hidden_layer_sizes': [
                    (12, 6, 3),         # Shallower architecture
                    (12, 8, 4, 3),      # Moderate depth
                ],
                'activations': [
                    ('relu', 'identity'),          # Common activation
                    ('tanh', 'identity'),          # Diversity with tanh
                ],
                # Most impactful learning rates
                'learning_rates': [0.0005, 0.001],
                # Retain both initializations
                'weight_initialization': ['he', 'xavier'],
                'momentum': [0.5, 0.9],           # Moderate and high momentum
                # Most effective regularizations
                'regularization': [0.0001, 0.0005],
                'dropout_rate': [0.1, 0.2],       # Two dropout levels
                # Focus on higher epoch counts
                'epochs': [200, 500],
            }

        }
        self.dataset_context = None

    def set_dataset_context(self, train_file):
        """Set the dataset context based on the training file name."""
        dataset_name = os.path.basename(train_file).split('.')[0]
        if dataset_name not in self.hyperparameters:
            raise ValueError(
                f"Invalid dataset specified. Use {list(self.hyperparameters.keys())}")
        self.dataset_context = dataset_name
        print(f"[INFO] Dataset context set to: {self.dataset_context}")

    def generate_combinations_grid(self):
        """Generate all valid hyperparameter combinations for the selected dataset."""
        if not self.dataset_context:
            raise ValueError(
                "Dataset context not set. Use set_dataset_context() first.")

        params = self.hyperparameters[self.dataset_context]
        keys, values = zip(*params.items())
        all_combinations = [dict(zip(keys, combo))
                            for combo in product(*values)]

        # Ensure valid activation layers match the number of hidden layers - 1
        valid_combinations = [combo for combo in all_combinations
                              if len(combo['activations']) == len(combo['hidden_layer_sizes']) - 1]

        random.shuffle(valid_combinations)
        return valid_combinations


def save_combinations(self, combinations, dataset_name, folder="cup_hyperparameters/generated"):
    """Save hyperparameter combinations to a JSON file."""
    folder = os.path.join(
        folder, dataset_name if "monk" in dataset_name else "")
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(
        folder, "generated_all.json" if "monk" in dataset_name else f"{dataset_name}_all.json")
    with open(file_path, 'w') as f:
        json.dump(combinations, f, indent=4)
    print(f"[INFO] Saved hyperparameters for {dataset_name} to {file_path}")


def save_best_hyperparameters(self, best_hyperparams, dataset_name, folder="cup_hyperparameters/chosen"):
    """Save the best hyperparameters after model selection."""
    folder = os.path.join(
        folder, dataset_name if "monk" in dataset_name else "")
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(
        folder, "selected_one.json" if "monk" in dataset_name else f"cup_json_{len(best_hyperparams)}.json")
    with open(file_path, 'w') as f:
        json.dump(best_hyperparams, f, indent=4)
    print(f"[INFO] Best hyperparameters saved to {file_path}")
