import os
import pandas as pd
import numpy as np
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
        blind_test_path, sep=',', skiprows=7, header=None, usecols=range(1, 13,)
    ).to_numpy()

    print(f"[DEBUG] Blind test shape: {blind_test_data.shape}")

    # Perform predictions
    predictions = model.forward(blind_test_data)

    # Save predictions to the results folder
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "blind_test_predictions.csv")
    np.savetxt(output_file, predictions, delimiter=",", fmt='%.6f')

    print(f"[INFO] Blind test predictions saved to {output_file}")


if __name__ == "__main__":

    # Define blind test file path
    best_model_file = "results/best_cup_model.npz"
    blind_test_path = "datasets/CUP/ML-CUP24-TS.csv"

    # Apply blind test on the saved model
    apply_blind_test(best_model_file, blind_test_path)
