import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from neuralNet import *


def extract_dataset_name(train_file):
    """Extract d36ataset name from the train file path."""
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
        # Ensure binary classification thresholding
        y_pred_labels = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(y_pred_labels.flatten() == y.flatten()) * 100

        accuracy = moving_average([accuracy], window_size=10)[-1]  # Smoothing
    return loss, accuracy


def moving_average(data, window_size=10):
    """Apply a moving average to smooth the data."""
    if len(data) < window_size:
        return data  # Return original data if it's smaller than window size
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_training_curve(model, dataset_name, plot_path=None, task_type=None):
    """
    Plots training and validation loss and accuracy curves.

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

    # Determine the appropriate loss label
    loss_label = "Loss (MSE)" if task_type == 'classification' else "Loss (MEE)"
    train_label = "Train Loss (MSE)" if task_type == 'classification' else "Train Loss (MEE)"
    val_label = "Validation Loss (MSE)" if task_type == 'classification' else "Validation Loss (MEE)"

    # Plot Loss (Train vs Validation)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, model.train_losses, '--', color="blue", label=train_label)

    # Adjust validation loss length to match epochs
    val_losses_to_plot = model.val_losses[:len(epochs)]
    plt.plot(epochs, val_losses_to_plot, color="red", label=val_label)

    plt.xlabel("Epochs")
    plt.ylabel(loss_label)
    plt.title(f"Loss vs. Epochs for {dataset_name}")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy or Metric (Train vs Validation or Development vs Internal Test)
    plt.subplot(1, 2, 2)
    if task_type == 'classification':
        if model.train_accuracies and len(model.train_accuracies) > 0 and model.val_accuracies and len(model.val_accuracies) > 0:
            smoothed_train_acc = moving_average(
                model.train_accuracies, window_size=10)
            smoothed_val_acc = moving_average(
                model.val_accuracies, window_size=10)

            plt.plot(epochs[:len(smoothed_train_acc)], smoothed_train_acc,
                     '--', color="blue", label="Train Accuracy")
            plt.plot(epochs[:len(smoothed_val_acc)], smoothed_val_acc,
                     color="red", label="Validation Accuracy")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy vs. Epochs for {dataset_name}")
        plt.legend()
        plt.grid(True)
    elif task_type == 'regression' and model.val_metrics:
        val_metrics_to_plot = model.val_metrics[:len(epochs)]
        # Ensure both are of same length
        val_losses_to_plot = model.val_losses[:len(epochs)]

        plt.plot(epochs[:len(val_metrics_to_plot)], val_metrics_to_plot, '--',
                 color="blue", label="Development Loss (MEE)")
        plt.plot(epochs[:len(val_losses_to_plot)], val_losses_to_plot, color="red",
                 label="Internal Test Loss (MEE)")

        plt.ylabel("Metric (MEE)")
        plt.title(f"MEE vs. Epochs for {dataset_name}")
        plt.legend()
        plt.grid(True)

    else:
        print(
            f"[WARNING] Skipping val_metrics plot for {dataset_name} as it does not exist or is empty.")

    plt.xlabel("Epochs")

    if plot_path is None:
        plot_folder = "plots"
        os.makedirs(plot_folder, exist_ok=True)
        plot_path = os.path.join(
            plot_folder, f"{dataset_name}_training_curve.png")

    try:
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"[ERROR] Failed to save plot: {e}")


def cross_validate_and_train(training_data, validation_data, hyperparams, task_function, train_file, n_folds=5, task_type='classification'):
    """Perform cross-validation to find the best hyperparameters, considering regularization."""

    X_dev, Y_dev = training_data
    dataset_name = extract_dataset_name(train_file)
    fold_splits = manual_kfold_split(X_dev, Y_dev, n_folds)

    print(
        f"\n[INFO] Performing {n_folds}-Fold CV on {len(hyperparams)} hyperparam combos for {dataset_name}...\n"
    )

    results = []
    for config in tqdm(hyperparams, desc="Hyperparam combos"):
        fold_losses, model_for_config = [], None

        # Handle regularization for specific datasets
        if dataset_name in ["monks-3", "cup"]:
            reg_type = config.get('regularization_type', 'none')
            reg_value = config.get('regularization', 0)
        else:
            reg_type = 'none'
            reg_value = 0

        for fold_num, (train_idx, val_idx) in enumerate(fold_splits):
            X_train_cv, Y_train_cv = X_dev[train_idx], Y_dev[train_idx]
            X_val_cv, Y_val_cv = X_dev[val_idx], Y_dev[val_idx]

            # Updating the configuration to include the chosen regularization settings
            config['regularization_type'] = reg_type
            config['regularization'] = reg_value

            # Train the model with current hyperparameters and regularization
            model = task_function((X_train_cv, Y_train_cv),
                                  (X_val_cv, Y_val_cv), config, dataset_name)
            loss, accuracy = evaluate_model(
                model, X_val_cv, Y_val_cv, task_type)
            fold_losses.append(loss)
            model_for_config = model

        avg_loss = np.mean(fold_losses)
        results.append(
            {"hyperparams": config, "avg_loss": avg_loss, "model": model_for_config})

    # Sort results by average loss
    results.sort(key=lambda x: x['avg_loss'])

    # Determine the number of models to return based on the dataset
    max_models = 5 if dataset_name.lower() == "cup" else 3

    # Filter for unique losses
    unique_results = []
    seen_losses = set()

    for result in results:
        if result['avg_loss'] not in seen_losses:
            unique_results.append(result)
            seen_losses.add(result['avg_loss'])

        # Stop once we have the required number of unique results
        if len(unique_results) >= max_models:
            break

    # Prepare output as lists
    best_models = [res["model"] for res in unique_results]
    best_hyperparams = [res["hyperparams"] for res in unique_results]

    print(f"\n[Top {max_models} Models Based on Validation Loss]")
    for rank, res in enumerate(unique_results, 1):
        print(f"[Rank {rank}] Loss: {res['avg_loss']:.6f}")

    print(
        f"\n[Summary] Total Hyperparameter Combos: {len(hyperparams)}, Total Folds: {n_folds}, Total Iterations: {len(hyperparams) * n_folds}"
    )

    return best_models, best_hyperparams
