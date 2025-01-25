import os
import pandas as pd
import numpy as np
from neuralNet import *
from preprocessor import load_and_preprocess_monk_dataset
from training import *

# Function to load and evaluate the best MONK model


def predict_monk(dataset_name):
    """Load the best saved MONK model and evaluate it on the test dataset."""

    # Define the file paths
    model_path = f"results/best_{dataset_name}_model.npz"
    test_file = f"datasets/monk/{dataset_name}.test"

    # Check if the model exists
    if not os.path.exists(model_path):
        print(
            f"[ERROR] No saved model found for {dataset_name} at {model_path}")
        return

    print(f"[INFO] Loading model for {dataset_name} from {model_path}")
    model = NeuralNetwork.load_model(model_path)

    # Load and preprocess the MONK test dataset
    X_test, Y_test = load_and_preprocess_monk_dataset(test_file)

    # Perform evaluation
    test_loss, test_accuracy = evaluate_model(
        model, X_test, Y_test, task_type='classification')

    # Display the results
    print(
        f"[FINAL] {dataset_name} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


# Choose which MONK model to evaluate by modifying this variable
if __name__ == "__main__":
    # Change the value to 'monks-1', 'monks-2', or 'monks-3'
    MONK_VERSION = "monks-3"

    predict_monk(MONK_VERSION)
