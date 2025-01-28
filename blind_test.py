import os
import pandas as pd
import numpy as np
from datetime import datetime
from neuralNet import NeuralNetwork
from main import *


def apply_blind_test(model_path, blind_test_path, output_folder="results"):
    """Apply the selected best model to the blind test set and save predictions."""
    print(f"[INFO] Applying blind test using model: {model_path}")

    # Load the trained model
    model = NeuralNetwork.load_model(model_path)
    print("[INFO] Model successfully loaded!")

    # Load and preprocess blind test dataset (remove first 7 rows, use 12 feature columns)
    blind_test_data = pd.read_csv(
        blind_test_path, sep=',', skiprows=7, header=None, usecols=range(1, 13)
    ).to_numpy()

    print(f"[DEBUG] Blind test shape: {blind_test_data.shape}")

    # Perform predictions
    predictions = model.forward(blind_test_data)

    # Define metadata rows (with automatic date)
    metadata_rows = [
        ["# TeamName: ML-crackers"],
        ["# Competition: ML-CUP24"],
        [f"# Date: {datetime.today().strftime('%m/%d/%Y')}"]
    ]

    # Convert metadata rows to a NumPy array with string type
    metadata_array = np.array(metadata_rows, dtype=str)

    # Add index column to predictions
    index_column = np.arange(
        1, predictions.shape[0] + 1).reshape(-1, 1)  # Index starts at 1
    predictions_with_index = np.hstack([index_column, predictions])

    # Convert predictions to string format for consistent saving
    predictions_str = np.array(predictions_with_index, dtype=str)

    # Ensure metadata_array and predictions_str align for concatenation
    if metadata_array.shape[1] < predictions_str.shape[1]:
        num_columns = predictions_str.shape[1]
        # Add empty columns to metadata_array to match predictions_str
        metadata_array = np.hstack([metadata_array, np.full(
            (metadata_array.shape[0], num_columns - metadata_array.shape[1]), '', dtype=str)])

    # Combine metadata and predictions
    final_output = np.vstack([metadata_array, predictions_str])

    # Save predictions with metadata to the results folder
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "blind_test_predictions.csv")
    np.savetxt(output_file, final_output, delimiter=",", fmt='%s')

    print(f"[INFO] Blind test predictions saved to {output_file}")


if __name__ == "__main__":

    # Define blind test file path
    best_model_file = "final_model/best_cup_model.npz"
    blind_test_path = "datasets/CUP/ML-CUP24-TS.csv"

    # Apply blind test on the saved model
    apply_blind_test(best_model_file, blind_test_path)
