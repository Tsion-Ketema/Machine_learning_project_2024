import os
import json
import pandas as pd
import numpy as np
from preprocessor import *
from hyperparameters import HyperparameterManager
from training import *
from task_functions import taskCup, taskMonk
from neuralNet import *

# ***************** setting up folders and subfolders *****************


def setup_directories(base_folder, subfolders):
    """Ensure necessary directories exist and return paths."""
    return {name: os.makedirs(os.path.join(base_folder, name), exist_ok=True) or os.path.join(base_folder, name)
            for name in subfolders}

# ***************** Initiating hyperparameter combination generating function *****************


def generate_hyperparameters(dataset_name, config_file):
    """Generate hyperparameters if not already present."""
    if not os.path.exists(config_file):
        manager = HyperparameterManager()
        manager.set_dataset_context(dataset_name)
        with open(config_file, 'w') as f:
            json.dump(manager.generate_combinations_grid(), f, indent=4)

    with open(config_file, "r") as f:
        return json.load(f)

# ***************** Load dataset / setting up hyperparameters / cross-validation initiation *****************


def process_dataset(dataset_name, train_file, test_file=None, n_folds=5, task_type='classification'):
    """Generic function to process MONK or CUP datasets."""
    # Load dataset
    if "monk" in dataset_name:
        X_train, Y_train = load_and_preprocess_monk_dataset(train_file)
        X_test, Y_test = load_and_preprocess_monk_dataset(test_file)

        paths = setup_directories("monk_hyperparameters", [
            "monks-1", "monks-2", "monks-3-no-reg", "monks-3-l2"
        ])

        if dataset_name in paths:
            config_file = os.path.join(
                paths[dataset_name], "generated_all.json")
        else:
            raise KeyError(
                f"[ERROR] Dataset key '{dataset_name}' not found in paths dictionary. Available keys: {list(paths.keys())}"
            )

    else:
        X_train, Y_train, X_test, Y_test = fetch_cup_dataset(split_test=True)

        # Setup directories for CUP dataset
        paths = setup_directories("cup_hyperparameters", [
                                  "chosen", "generated"])
        config_file = os.path.join(paths["generated"], "cup_all.json")

    # Preparing the data to be trained and validation
    training_data, validation_data = (X_train, Y_train), (X_test, Y_test)

    # Dynamically update hyperparameters for monks-3-l2
    dynamic_updates = {}
    if dataset_name == "monks-3-l2":
        dynamic_updates = {
            "dropout_rate": 0.2,  # Apply dropout
            "batch_size": 32,    # Use batch size
        }

    # Generating hyperparameter combination
    hyperparams = generate_hyperparameters(dataset_name, config_file)

    # Train and evaluate
    best_models, best_hyperparams_list = cross_validate_and_train(
        training_data, validation_data, hyperparams,
        task_function=lambda train_data, val_data, config, dataset_name: (
            taskMonk(train_data, val_data, config,
                     dataset_name, **dynamic_updates)
            if "monk" in dataset_name else taskCup(train_data, val_data, config, dataset_name)
        ),
        train_file=train_file, n_folds=n_folds, task_type=task_type
    )

    # Save only the required number of models (3 for MONK, 5 for CUP)
    num_models_to_save = 3 if "monk" in dataset_name else 5
    best_hyperparams_list = best_hyperparams_list[:num_models_to_save]
    best_models = best_models[:num_models_to_save]

    # Test the best model on the entire training and testing data
    best_model = best_models[0]
    train_loss = best_model.compute_loss(Y_train, best_model.forward(X_train))
    test_loss = best_model.compute_loss(Y_test, best_model.forward(X_test))

    print("**************************************")

    print(f"Final Training Loss: {train_loss}")
    print(f"Final Test Loss: {test_loss}")

    # Get model predictions
    train_predictions = best_model.forward(X_train)
    test_predictions = best_model.forward(X_test)

    # Convert predictions to binary labels (classification threshold of 0.5)
    train_predicted_labels = (train_predictions >= 0.5).astype(int)
    test_predicted_labels = (test_predictions >= 0.5).astype(int)

    # Compute accuracy
    train_accuracy = np.mean(
        train_predicted_labels.flatten() == Y_train.flatten()) * 100
    test_accuracy = np.mean(
        test_predicted_labels.flatten() == Y_test.flatten()) * 100

    # Display accuracy
    print(f"Final Training Accuracy: {train_accuracy:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    print("**************************************")

    # Debugging: Ensure val_metrics is retained
    if hasattr(best_model, 'val_metrics') and best_model.val_metrics:
        print(
            f"[DEBUG] Final val_metrics length: {len(best_model.val_metrics)}")
        print(
            f"[DEBUG] val_metrics (Development Loss - First 5): {best_model.val_metrics[:5]}")
    else:
        print("[WARNING] val_metrics is empty! Check training process.")

    # Plot only the best model (the first model in the sorted best_models list)
    plot_folder = os.path.join(
        "plot", "monk" if "monk" in dataset_name else "cup"
    )
    os.makedirs(plot_folder, exist_ok=True)

    # Save plots for all top models
   # Save plots for all top models, marking the best one
    for i, model in enumerate(best_models):
        if i == 0:
            model_name = f"{dataset_name}_model_{i+1}_best"  # Best model
        else:
            model_name = f"{dataset_name}_model_{i+1}"  # Other top models

        plot_path = os.path.join(plot_folder, f"{model_name}.png")

        plot_training_curve(
            model, dataset_name=model_name, plot_path=plot_path, task_type=task_type
        )

    print(f"[INFO] Saved training plot for {model_name} at {plot_path}")

    if "cup" in dataset_name:
        chosen_folder = os.path.join("cup_hyperparameters", "chosen")
        os.makedirs(chosen_folder, exist_ok=True)

        for i, hyperparams in enumerate(best_hyperparams_list):
            chosen_file = os.path.join(chosen_folder, f"cup_model_{i+1}.json")
            with open(chosen_file, 'w') as f:
                json.dump(hyperparams, f, indent=4)

        print(
            f"[INFO] Saved hyperparameters for top 5 models in {chosen_folder}")


if __name__ == "__main__":
    RUN_MONK = False  # Set to False to run CUP instead
    MONK_VERSION = "monks-3"  # Choose among "monks-1", "monks-2", "monks-3"

    if RUN_MONK:
        if MONK_VERSION == "monks-3":
            # Process without regularization
            # process_dataset("monks-3-no-reg", f"datasets/monk/monks-3.train",
            #                 f"datasets/monk/monks-3.test", n_folds=5, task_type='classification')

            # # Process with L2 regularization
            process_dataset("monks-3-l2", f"datasets/monk/monks-3.train",
                            f"datasets/monk/monks-3.test", n_folds=5, task_type='classification')
        else:
            process_dataset(MONK_VERSION, f"datasets/monk/{MONK_VERSION}.train",
                            f"datasets/monk/{MONK_VERSION}.test", n_folds=5, task_type='classification')
    else:
        process_dataset("cup", "CUP-DEV-SET.csv",
                        n_folds=5, task_type='regression')
