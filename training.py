import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from neuralNet import *


def extract_dataset_name(train_file):
    """Extract dataset name from the train file path."""
    return os.path.basename(train_file).split('.')[0]


def manual_kfold_split(X, Y, n_folds=5, shuffle=True, random_seed=42):
    """Manually split data into K folds for cross-validation."""
    indices = np.arange(len(X))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    fold_sizes = np.full(n_folds, len(X) // n_folds, dtype=int)
    fold_sizes[: len(X) % n_folds] += 1

    folds = np.array_split(indices, n_folds)
    splits = [(np.hstack([folds[i] for i in range(n_folds) if i != f]), folds[f])
              for f in range(n_folds)]
    return splits


def evaluate_model(model, X, y, task_type):
    """Evaluate model and return loss and accuracy (if classification)."""
    y_pred = model.forward(X)
    loss = model.compute_loss(y, y_pred)

    accuracy = None  # Default to None for regression
    if task_type == 'classification':
        # print("Predicted Probabilities:", y_pred[:10].flatten())
        # print("Predicted Binary Labels:",
        #       (y_pred >= 0.5).astype(int)[:10].flatten())
        # print("Actual Labels:", y.flatten()[:10])

        # Predicted Probabilities:
        y_pred[:10].flatten()
        # Predicted Binary Labels:"
        (y_pred >= 0.5).astype(int)[:10].flatten()
        # Actual Labels:"
        y.flatten()[:10]

        # Ensure binary classification thresholding
        y_pred_labels = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(y_pred_labels.flatten() == y.flatten()) * 100

        # print("Predicted Labels:", y_pred_labels[:10])
        # print("Actual Labels:", y.flatten()[:10])

        y_pred_labels = y_pred_labels.flatten()
        y_true = y.flatten()

        accuracy_debug = np.mean(y_pred_labels == y_true) * 100

        # print("Corrected Accuracy Calculation:", accuracy_debug)

        # print("Accuracy Calculation:", np.mean(
        #     y_pred_labels == y.flatten()) * 100)

        # Evaluate on the entire test set
        y_pred_full = (model.forward(X) >= 0.5).astype(int)
        accuracy_full = np.mean(y_pred_full.flatten() == y.flatten()) * 100
        # print(f"Full Test Accuracy: {accuracy_full}%")

    return loss, accuracy


def plot_training_curve(model, dataset_name, plot_path=None, task_type=None):
    """
    Plot training loss and accuracy curves.

    Parameters:
    - model: Trained neural network model
    - dataset_name: Name of the dataset (used for labeling the plots)
    - plot_path: Path to save the plot
    - task_type: Task type ('classification' or 'regression')
    """

    task_type = task_type or (
        'classification' if 'monk' in dataset_name.lower() else 'regression'
    )

    if not model.train_losses or len(model.train_losses) == 0:
        print(f"[WARNING] No training curve data for {dataset_name}.")
        return

    epochs = range(1, len(model.train_losses) + 1)
    plt.figure(figsize=(14, 6))

    # Plot Loss/Error
    plt.subplot(1, 2, 1)
    plt.plot(epochs, model.train_losses, '--',
             color="blue", label="Train Loss")

    if model.val_losses and len(model.val_losses) == len(epochs):
        plt.plot(epochs, model.val_losses,
                 color="red", label="Validation Loss")
    else:
        print(
            f"[WARNING] Validation loss data is missing or inconsistent for {dataset_name}.")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss vs. Epochs for {dataset_name}")
    plt.legend()

    # Plot Accuracy (if classification task)
    plt.subplot(1, 2, 2)
    if task_type == 'classification':
        if model.train_accuracies and model.val_accuracies:
            if len(model.train_accuracies) == len(epochs) and len(model.val_accuracies) == len(epochs):
                plt.plot(epochs, model.train_accuracies, '--',
                         color="blue", label="Train Accuracy")
                plt.plot(epochs, model.val_accuracies,
                         color="red", label="Validation Accuracy")
            else:
                print(
                    f"[WARNING] Accuracy data is inconsistent for {dataset_name}.")
        else:
            print(f"[WARNING] Accuracy data missing for {dataset_name}.")

        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy vs. Epochs for {dataset_name}")
        plt.legend()

    plt.xlabel("Epochs")

    # Ensure directories exist before saving the plot
    plot_folder = "plot/cup" if "cup" in dataset_name.lower() else f"plot/monk"
    os.makedirs(plot_folder, exist_ok=True)

    plot_path = os.path.join(plot_folder, f"{dataset_name}_training_curve.png")

    try:
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] Training curve saved to {plot_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save plot: {e}")


def cross_validate_and_train(training_data, validation_data, hyperparams, task_function, train_file, n_folds=5, task_type='classification'):
    """Perform cross-validation to find the best hyperparameters."""
    X_dev, Y_dev = training_data
    dataset_name = extract_dataset_name(train_file)
    fold_splits = manual_kfold_split(X_dev, Y_dev, n_folds)

    print(
        f"\n[INFO] Performing {n_folds}-Fold CV on {len(hyperparams)} hyperparam combos for {dataset_name}...\n")

    results = []
    for config in tqdm(hyperparams, desc="Hyperparam combos"):
        fold_losses, model_for_config = [], None

        for train_idx, val_idx in fold_splits:
            X_train_cv, Y_train_cv = X_dev[train_idx], Y_dev[train_idx]
            X_val_cv, Y_val_cv = X_dev[val_idx], Y_dev[val_idx]
            model = task_function((X_train_cv, Y_train_cv),
                                  (X_val_cv, Y_val_cv), config)
            loss, accuracy = evaluate_model(
                model, X_val_cv, Y_val_cv, task_type)
            fold_losses.append(loss)
            model_for_config = model

        avg_loss = np.mean(fold_losses)
        results.append(
            {"hyperparams": config, "avg_loss": avg_loss, "model": model_for_config})

    top_results = sorted(results, key=lambda x: x['avg_loss'])[:3]
    best_models, best_hyperparams = zip(
        *[(res["model"], res["hyperparams"]) for res in top_results])

    print("\n[Top 3 Models Based on Validation Loss]")
    for rank, res in enumerate(top_results, 1):
        print(f"[Rank {rank}] Loss: {res['avg_loss']:.4f}")

    print(
        f"\n[Summary] Total Hyperparameter Combos: {len(hyperparams)}, Total Folds: {n_folds}, Total Iterations: {len(hyperparams) * n_folds}")

    return best_models, best_hyperparams
