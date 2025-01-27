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
                    (17, 4, 1), (17, 8, 4, 1), (17, 12, 6, 1), (17,
                                                                16, 8, 1)  # Simple architectures to complex
                ],
                'activations': [
                    ('relu', 'sigmoid'),
                    ('tanh', 'sigmoid')
                ],
                'learning_rates': [
                    0.76, 0.8, 0.83, 0.001, 0.002  # Based on the report's eta values
                ],
                'weight_initialization': [
                    # Both Glorot (Xavier) and He initialization
                    'xavier', 'he'
                ],
                'momentum': [
                    0.8, 0.83, 0.85, 0.9  # Fine-tuned momentum values to smooth convergence
                ],
                'epochs': [
                    1000  # Report suggests convergence by 500 epochs
                ],
                'weight_decay': [
                    0  # No regularization as indicated in the report
                ]
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

            'monks-3-no-reg': {  # Monk-3 without regularization
                'hidden_layer_sizes': [(17, 8, 1), (17, 16, 1), (17, 8, 8, 1)],
                'activations': [('relu', 'sigmoid'), ('tanh', 'sigmoid')],
                'learning_rates': [0.001, 0.005, 0.01],
                'weight_initialization': ['xavier', 'he'],
                'momentum': [0.9, 0.95, 0.99],
                'epochs': [500],
                'regularization': [('none', 0), ('L1', 0.0001), ('L2', 0.0005)],
            },
            'monks-3-l1': {  # Monk-3 with L1 regularization
                'hidden_layer_sizes': [(17, 15, 1)],
                'activations': [('relu', 'sigmoid')],
                'learning_rates': [0.001],
                'weight_initialization': ['xavier'],
                'momentum': [0.9],
                'epochs': [500],
                'regularization': [('L1', 0.0012)],  # L1 regularization
            },
            'monks-3-l2': {  # Monk-3 with L2 regularization
                'hidden_layer_sizes': [(17, 15, 1)],
                'activations': [('relu', 'sigmoid')],
                'learning_rates': [0.001],
                'weight_initialization': ['xavier'],
                'momentum': [0.9],
                'epochs': [500],
                'regularization': [('L2', 0.0023)],  # L2 regularization
            },
            'cup': {
                'hidden_layer_sizes': [(12, 8, 3), (12, 16, 8, 3)],
                'activations': [('relu', 'identity'), ('leaky_relu', 'leaky_relu', 'identity')],
                'learning_rates': [0.0001, 0.001],
                'batch_sizes': [8, 16],
                'weight_initialization': ['he', 'xavier'],
                'momentum': [0.2, 0.8],
                'regularization': [('L1', 0.0001), ('L2', 0.001)],
                'epochs': [1000],
                'dropout_rate': [0.1, 0.2],
            },
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
