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

# ***************** Initiating hyperparamter combination generating funciton *****************


def generate_hyperparameters(dataset_name, config_file):
    """Generate hyperparameters if not already present."""
    if not os.path.exists(config_file):
        manager = HyperparameterManager()
        manager.set_dataset_context(dataset_name)
        with open(config_file, 'w') as f:
            json.dump(manager.generate_combinations_grid(), f, indent=4)
        print(f"[INFO] {dataset_name} hyperparameter combinations saved.")

    with open(config_file, "r") as f:
        return json.load(f)

# ***************** Load dataset /setting up hyperparamters/ cross validation initiation *****************


def process_dataset(dataset_name, train_file, test_file=None, n_folds=5, task_type='classification'):
    """Generic function to process MONK or CUP datasets."""
    # Load dataset
    if "monk" in dataset_name:
        X_train, Y_train = load_and_preprocess_monk_dataset(train_file)
        X_test, Y_test = load_and_preprocess_monk_dataset(test_file)

        paths = setup_directories("monk_hyperparameters", [
            "monks-1", "monks-2", "monks-3-no-reg", "monks-3-l1", "monks-3-l2"
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

    # Generating hyperparameter combination
    hyperparams = generate_hyperparameters(dataset_name, config_file)

    # Train and evaluate
    best_models, best_hyperparams_list = cross_validate_and_train(
        training_data, validation_data, hyperparams,
        taskMonk if "monk" in dataset_name else taskCup,
        train_file, n_folds, task_type
    )

    # Evaluate the prediction of the entire training and testing data using the best model#
    best_model = best_models[0]
    # test_loss, test_accuracy = evaluate_model(
    #     best_model, X_test, Y_test, task_type)

    # Test the best model on the entire training and testing data
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

    # Plot only the best model (the first model in the sorted best_models list)
    best_model = best_models[0]  # Select the model with the lowest avg loss
    best_hyperparams = best_hyperparams_list[0]

    plot_folder = os.path.join(
        "plot", "monk" if "monk" in dataset_name else "cup"
    )
    os.makedirs(plot_folder, exist_ok=True)

    # Save the training curve of the best model only
    plot_path = os.path.join(plot_folder, f"{dataset_name}_best_model.png")

    plot_training_curve(
        best_model, dataset_name=f"{dataset_name}_best", plot_path=plot_path, task_type=task_type
    )

    # Save only the best hyperparameters under chosen folder
    chosen_folder = paths.get(dataset_name, os.path.join(
        "monk_hyperparameters", dataset_name))
    os.makedirs(chosen_folder, exist_ok=True)

    chosen_file = os.path.join(chosen_folder, f"{dataset_name}_best.json")

    with open(chosen_file, 'w') as f:
        json.dump(best_hyperparams, f, indent=4)

    print(f"[INFO] The best model plot saved to {plot_path}")

    # Save top 5 models in the chosen folder inside cup_hyperparameters
    chosen_folder = os.path.join("cup_hyperparameters", "chosen")
    os.makedirs(chosen_folder, exist_ok=True)

    for i in range(5):
        chosen_file = os.path.join(chosen_folder, f"cup_model{i+1}.json")
        with open(chosen_file, 'w') as f:
            json.dump(best_hyperparams_list[i], f, indent=4)

    print(f"[INFO] Top 5 models saved to {chosen_folder}")

    # Save final results the model and the corresponding json file
    os.makedirs("results", exist_ok=True)
    best_hp_file = os.path.join(
        "results", f"best_{dataset_name}_hyperparameters.json")
    best_model_file = os.path.join("results", f"best_{dataset_name}_model.npz")

    with open(best_hp_file, 'w') as f:
        json.dump(best_hyperparams_list[0], f, indent=4)
    best_model.save_model(best_model_file)


if __name__ == "__main__":
    RUN_MONK = True  # Set to False to run CUP instead
    MONK_VERSION = "monks-3"  # Choose among "monks-1", "monks-2", "monks-3"

    if RUN_MONK:
        if MONK_VERSION == "monks-3":
            # Process without regularization
            # process_dataset("monks-3-no-reg", f"datasets/monk/monks-3.train",
            #                 f"datasets/monk/monks-3.test", n_folds=5, task_type='classification')

            # Process with L1 regularization
            process_dataset("monks-3-l1", f"datasets/monk/monks-3.train",
                            f"datasets/monk/monks-3.test", n_folds=5, task_type='classification')

            # # Process with L2 regularization
            # process_dataset("monks-3-l2", f"datasets/monk/monks-3.train",
            #                 f"datasets/monk/monks-3.test", n_folds=5, task_type='classification')
        else:
            process_dataset(MONK_VERSION, f"datasets/monk/{MONK_VERSION}.train",
                            f"datasets/monk/{MONK_VERSION}.test", n_folds=5, task_type='classification')
    else:
        process_dataset("cup", "CUP-DEV-SET.csv",
                        n_folds=5, task_type='regression')
