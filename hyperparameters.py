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
                    (17, 4, 1),
                    (17, 8, 4, 1),
                    (17, 12, 6, 1)
                ],
                'activations': [

                    ('relu', 'sigmoid'),
                    ('tanh', 'sigmoid')
                ],
                'learning_rates': [
                    0.01,
                    0.005,
                    0.001
                ],
                'weight_initialization': [
                    'xavier',
                    'he'
                ],
                'momentum': [
                    0.85,
                    0.9,
                    0.95
                ],
                'epochs': [
                    1000
                ],
                'weight_decay': [
                    0,
                    0.00001,
                    0.0001
                ],

            },
            'monks-3-no-reg': {
                'hidden_layer_sizes': [
                    (17, 15, 1),
                    (17, 12, 6, 1),
                    (17, 15, 10, 1)
                ],
                'activations': [
                    ('relu', 'sigmoid'),
                    ('tanh', 'sigmoid'),
                    ('tanh', 'relu', 'sigmoid')
                ],
                'learning_rates': [
                    0.001,
                    0.002,
                    0.005
                ],
                'weight_initialization': [
                    'xavier',
                    'he'
                ],
                'momentum': [
                    0.85,
                    0.9
                ],
                'epochs': [1000],
                'weight_decay': [
                    0
                ],
            },

            'monks-3-l2': {
                'hidden_layer_sizes': [(17, 12, 6, 1), (17, 8, 4, 1)],
                'activations': [('relu', 'relu', 'sigmoid')],
                'learning_rates': [0.001, 0.0007, 0.0003],
                'batch_sizes': [8, 16],
                'weight_initialization': ['he'],
                'momentum': [0.8, 0.9],
                'regularization': [('L2', 0.0001), ('L2', 0.0003)],
                'epochs': [2000, 2500]
            },

            'cup': {
                'hidden_layer_sizes': [
                    (12, 6, 3),        # Shallow network
                    (12, 10, 6, 3)     # Slightly deeper for feature extraction
                ],
                'activations': [
                    ('tanh', 'identity'),        # Smooth gradients
                    ('relu', 'identity')         # ReLU as a robust alternative
                ],
                # Balanced learning rates
                'learning_rates': [0.0001, 0.0003, 0.001],
                # Reduced to 3
                'weight_initialization': ['he', 'xavier', 'lecun'],
                'momentum': [0.6, 0.7],   # Keep 2 values
                'regularization': [
                    ('L2', 0.00005),    # Light L2
                    ('L2', 0.0001)      # Slightly stronger L2
                ],
                'dropout_rate': [0.0],   # Keep fixed at 0.0
                'epochs': [500, 1000]   # Two training durations
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
