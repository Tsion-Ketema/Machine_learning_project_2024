import json
import os
import random
from itertools import product


class HyperparameterManager:
    """Manages generation, saving, and loading of hyperparameter combinations."""

    def __init__(self):
        # Define hyperparameters for CUP and MONK datasets
        self.hyperparameters = {
            'cup': {
                'hidden_layer_sizes': [(12, 8, 3), (12, 16, 8, 3)],
                'activations': [('relu', 'identity'), ('leaky_relu', 'leaky_relu', 'identity')],
                'learning_rates': [0.0001, 0.001],
                'batch_sizes': [8, 16],
                'weight_initialization': ['he', 'xavier'],
                'momentum': [0.2, 0.8],
                'regularization': [0.0001, 0.001, 0.01],
                'epochs': [1000],
                'optimizer': ['sgd', 'adam'],
                'dropout_rate': [0.1, 0.2],
            },
            'monks-1': {
                'hidden_layer_sizes': [
                    (17, 4, 1),  # Simplest architecture
                    (17, 8, 4, 1),  # Moderate complexity
                    (17, 12, 6, 1),  # Increased capacity
                    (17, 16, 8, 4, 1)  # More depth
                ],  # (4 options)

                'activations': [
                    ('relu', 'sigmoid'),  # Standard choice
                    ('relu', 'relu', 'sigmoid'),
                    ('tanh', 'tanh', 'sigmoid'),
                    ('leaky_relu', 'relu', 'sigmoid')
                ],  # (4 options)
                'learning_rates': [0.001, 0.005, 0.01],  # (3 options)
                # 'batch_sizes': [8, 16, 32],  # (3 options)
                'weight_initialization': ['xavier', 'he'],  # (2 options)
                'momentum': [0.8, 0.9],  # (2 options)
                # (2 options - test with and without)
                # 'regularization': [0, 0.0001],
                'epochs': [1000],  # (1 option)
                'weight_decay': [0, 0.00005]  # (2 options)
            },
            'monks-2': {
                # 3 options
                'hidden_layer_sizes': [(17, 8, 1), (17, 16, 1), (17, 8, 8, 1)],
                # 2 options
                'activations': [('relu', 'sigmoid'), ('tanh', 'sigmoid')],
                'learning_rates': [0.001, 0.005, 0.01],  # 3 options
                'weight_initialization': ['xavier', 'he'],  # 2 options
                'momentum': [0.9, 0.95, 0.99],  # 3 options
                'epochs': [500],  # 2 options
            },
            'monks-3': {
                # 3 options
                'hidden_layer_sizes': [(17, 8, 1), (17, 16, 1), (17, 8, 8, 1)],
                # 2 options
                'activations': [('relu', 'sigmoid'), ('tanh', 'sigmoid')],
                'learning_rates': [0.001, 0.005, 0.01],  # 3 options
                'weight_initialization': ['xavier', 'he'],  # 2 options
                'momentum': [0.9, 0.95, 0.99],  # 3 options
                'epochs': [500],  # 2 options
                'regularization': [0.0001, 0.0005],
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

    def load_best_hyperparameters(self, dataset_name, folder="hyperparameters/best"):
        """Load the best hyperparameters from a JSON file."""
        file_path = os.path.join(
            folder, f"best_{dataset_name}_hyperparams.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                print(f"[INFO] Loaded best hyperparameters from {file_path}")
                return json.load(f)
        else:
            print(
                f"[WARNING] No best hyperparameters found for {dataset_name}.")
            return None
