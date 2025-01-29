import json
from neuralNet import NeuralNetwork
from task_functions import train_model
from preprocessor import fetch_cup_dataset

# Load training and testing datasets for CUP


def load_cup_data():
    """Load the CUP training and testing data."""
    X_train, Y_train, X_test, Y_test = fetch_cup_dataset(
        split_test=True)  # Ensure this function is correct
    return X_train, Y_train, X_test, Y_test


# Load hyperparameter files
best_models_folder = "cup_hyperparameters/chosen"
model_files = ["cup_model_1.json", "cup_model_2.json", "cup_model_3.json"]

# Define default values for missing hyperparameters
default_hyperparams = {
    "learning_rate": 0.001,  # Default learning rate
    "momentum": 0.9,         # Default momentum
    "regularization": None,  # No regularization by default
    "epochs": 100,           # Default number of epochs
    "weight_initialization": "xavier",  # Default weight initialization
    "activations": ["relu", "identity"]  # Default activations
}

# Load CUP data
X_train, Y_train, X_test, Y_test = load_cup_data()

for i, model_file in enumerate(model_files):
    try:
        # Load hyperparameters
        with open(f"{best_models_folder}/{model_file}", "r") as f:
            params = json.load(f)

        # Fill missing hyperparameters with defaults
        for key, default_value in default_hyperparams.items():
            if key not in params:
                params[key] = default_value

        # Initialize the neural network model
        model = NeuralNetwork(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            momentum=params["momentum"],
            weight_initialization=params["weight_initialization"],
            regularization=(params.get("regularization_type"),
                            params.get("regularization", 0)),
            activations=params["activations"],
            task_type='regression',
            dataset_name="cup"
        )

        # Train the model
        model.train(X_train, Y_train, X_val=X_test, y_val=Y_test)

        # Evaluate train and test losses
        train_loss = model.compute_loss(Y_train, model.forward(X_train))
        test_loss = model.compute_loss(Y_test, model.forward(X_test))

        print(f"[RESULT] Model {i+1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")

    except Exception as e:
        print(f"[ERROR] Failed to train Model {i+1} ({model_file}): {e}")
