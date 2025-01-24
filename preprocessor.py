import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt

# ======================== MONK Preprocessing ========================


def load_and_preprocess_monk_dataset(file_path):
    """Load and preprocess the MONK dataset (train/test)."""
    # One hot encoding is being done on the feature set not on the label part.
    data = pd.read_csv(file_path, sep=" ", names=[
                       'y1', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'id']).set_index('id')
    return one_hot_encode_monk(data.drop(columns=['y1']).to_numpy()), data['y1'].to_numpy().reshape(-1, 1)


def one_hot_encode_monk(features):
    """One-hot encode MONK dataset categorical features ensuring consistent shape."""
    num_features = [3, 3, 2, 3, 4, 2]

    encoded = []
    for row in features:
        encoded_row = []
        for num, val in zip(num_features, row):
            # Validate the value to ensure it is within the expected range
            if 1 <= val <= num:
                one_hot = [1 if i == (val - 1) else 0 for i in range(num)]
            else:
                raise ValueError(
                    f"Invalid feature value {val} for feature with range {num}")
            encoded_row.extend(one_hot)
        encoded.append(encoded_row)
    return np.array(encoded)


# ======================== CUP Preprocessing ========================

def fetch_cup_dataset(split_test=True):
    """Load and preprocess CUP dataset, splitting if necessary."""
    dataset_dir = "./datasets/Cup/"
    dev_file, test_file = os.path.join(
        dataset_dir, "CUP-DEV-SET.csv"), os.path.join(dataset_dir, "CUP-INTERNAL-TEST.csv")

    if split_test and not (os.path.exists(dev_file) and os.path.exists(test_file)):
        _split_cup_dataset(dataset_dir)

    return _load_cup_data(dev_file, test_file, split_test)


def _split_cup_dataset(dataset_dir):
    """Split CUP dataset into 80% training and 20% internal test."""
    file_path = os.path.join(dataset_dir, "ML-CUP24-TR.csv")
    column_headers = [
        'id'] + [f'attr{i}' for i in range(1, 13)] + ['target_x', 'target_y', 'target_z']
    data = pd.read_csv(file_path, sep=',', names=column_headers, skiprows=7).sample(
        frac=1, random_state=42).reset_index(drop=True)

    # Print the shape of the entire data before splitting
    print("\n[DEBUG] Original dataset shape (before split):", data.shape)

    split_idx = math.floor(len(data) * 0.2)
    internal_test_data = data.iloc[:split_idx].to_csv(os.path.join(
        dataset_dir, "CUP-INTERNAL-TEST.csv"), index=False, header=False)
    dev_set_data = data.iloc[split_idx:].to_csv(os.path.join(
        dataset_dir, "CUP-DEV-SET.csv"), index=False, header=False)

    print("[DEBUG] Internal test set shape:", internal_test_data.shape)
    print("[DEBUG] Development set shape:  ", dev_set_data.shape)


def _load_cup_data(dev_file, test_file, split_test):
    """Load CUP dataset splits and shuffle the training set."""
    train_features, train_labels = _read_cup_file(dev_file)
    test_features, test_labels = _read_cup_file(
        test_file) if split_test else (None, None)

    shuffled_indices = np.arange(len(train_labels))
    np.random.shuffle(shuffled_indices)
    return train_features[shuffled_indices], train_labels[shuffled_indices], test_features, test_labels


def _read_cup_file(file_path):
    """Read CUP dataset file into feature and label arrays."""
    features = pd.read_csv(file_path, sep=',', usecols=range(
        1, 13)).to_numpy(dtype=np.float32)
    labels = pd.read_csv(file_path, sep=',', usecols=range(
        13, 16)).to_numpy(dtype=np.float32)
    return features, labels
