import os
import json
import pandas as pd
import numpy as np
from preprocessor import *
from hyperparameters import HyperparameterManager
from training import *
from task_functions import taskCup, taskMonk
from neuralNet import NeuralNetwork


def setup_directories(base_folder, subfolders):
    """Ensure necessary directories exist and return paths."""
    return {name: os.makedirs(os.path.join(base_folder, name), exist_ok=True) or os.path.join(base_folder, name)
            for name in subfolders}


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


def process_dataset(dataset_name, train_file, test_file=None, n_folds=5, task_type='classification'):
    """Generic function to process MONK or CUP datasets."""
    # Load dataset
    if "monk" in dataset_name:
        X_train, Y_train = load_and_preprocess_monk_dataset(train_file)
        X_test, Y_test = load_and_preprocess_monk_dataset(test_file)
        
        # The encoded data will be returned here.

        # print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print("Sample X_train:", X_train[:1])
        print("Sample X_test:", X_test[:1])

        # common_samples = np.intersect1d(X_train, X_test)
        # print("Data Type after Conversion:", X_train.dtype, X_test.dtype)

        # if len(common_samples) > 0:
        #     print(
        #         f"Data Leakage Detected! {len(common_samples)} samples are duplicated in train and test sets.")
        # else:
        #     print("No data leakage detected. Train and test sets are unique.")

        # Setup directories for monk datasets using original names
        paths = setup_directories("monk_hyperparameters", [
                                  "monks-1", "monks-2", "monks-3"])

        if dataset_name not in paths:
            raise KeyError(
                f"[ERROR] Dataset key '{dataset_name}' not found in paths dictionary. Available keys: {list(paths.keys())}")

        config_file = os.path.join(paths[dataset_name], "generated_all.json")
    else:
        X_train, Y_train, X_test, Y_test = fetch_cup_dataset(split_test=True)

        # Setup directories for CUP dataset
        paths = setup_directories("cup_hyperparameters", [
                                  "chosen", "generated"])
        config_file = os.path.join(paths["generated"], "cup_all.json")

    training_data, validation_data = (X_train, Y_train), (X_test, Y_test)

    hyperparams = generate_hyperparameters(dataset_name, config_file)

    # Train and evaluate
    best_models, best_hyperparams_list = cross_validate_and_train(
        training_data, validation_data, hyperparams,
        taskMonk if "monk" in dataset_name else taskCup,
        train_file, n_folds, task_type
    )

    # Save results
    for i, model in enumerate(best_models):
        plot_folder = os.path.join(
            "plot", "monk" if "monk" in dataset_name else "cup")
        os.makedirs(plot_folder, exist_ok=True)

        plot_path = os.path.join(plot_folder, f"{dataset_name}_model{i+1}.png")

        plot_training_curve(
            model, dataset_name=f"{dataset_name}_model{i+1}", plot_path=plot_path, task_type=task_type)

        chosen_file = os.path.join(
            paths["chosen"], f"cup_json_{i+1}.json") if dataset_name == "cup" else os.path.join(paths[dataset_name], "selected_one.json")
        with open(chosen_file, 'w') as f:
            json.dump(best_hyperparams_list[i], f, indent=4)

    best_model = best_models[0]
    test_loss, test_accuracy = evaluate_model(
        best_model, X_test, Y_test, task_type)

    # print(X_test.shape, X_train.shape)
    # print(Y_test[:5])  # Check a few sample labels

    if test_accuracy is not None:
        print(
            f"[FINAL] {dataset_name} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    else:
        print(f"[FINAL] {dataset_name} Test Loss: {test_loss:.4f}")

    # Save final results
    os.makedirs("results", exist_ok=True)
    best_hp_file = os.path.join(
        "results", f"best_{dataset_name}_hyperparameters.json")
    best_model_file = os.path.join("results", f"best_{dataset_name}_model.npz")

    with open(best_hp_file, 'w') as f:
        json.dump(best_hyperparams_list[0], f, indent=4)
    best_model.save_model(best_model_file)


if __name__ == "__main__":
    # Configuration: choose dataset and parameters in one place
    RUN_MONK = True  # Set to False to run CUP instead
    MONK_VERSION = "monks-1"  # Choose among "monks-1", "monks-2", "monks-3"

    if RUN_MONK:
        process_dataset(MONK_VERSION, f"datasets/monk/{MONK_VERSION}.train",
                        f"datasets/monk/{MONK_VERSION}.test", n_folds=3, task_type='classification')
    else:
        process_dataset("cup", "CUP-DEV-SET.csv",
                        n_folds=5, task_type='regression')
