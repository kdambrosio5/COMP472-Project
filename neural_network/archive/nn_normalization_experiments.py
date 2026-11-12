
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize

from neural_network.model import NeuralNetwork, train_model


RANDOM_STATE = 42

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "image_train.csv"
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)


def load_data(normalization_method):

    # Load the image dataset
    df = pd.read_csv(data_path)

    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)

    # Use the same stratified split as previous experiments
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Apply the selected normalization method
    if normalization_method == "standard":

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    elif normalization_method == "l2":

        X_train = normalize(X_train, norm="l2")
        X_val = normalize(X_val, norm="l2")

    elif normalization_method == "none":

        pass

    else:
        raise ValueError("Unknown normalization method")

    # Convert NumPy arrays into PyTorch tensors
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long)
    )


def run_normalization_experiments():

    normalization_methods = ["none", "standard", "l2"]

    summaries = []

    for method in normalization_methods:

        # Reset seeds for reproducibility
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)

        print(f"\nTesting normalization: {method}\n")

        X_train, X_val, y_train, y_val = load_data(method)

        model = NeuralNetwork(
            input_size=500,
            num_classes=50,
            hidden_layers=(512, 256),
            dropout=0.0
        )

        start_time = time.perf_counter()

        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=20,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.001
        )

        training_time = time.perf_counter() - start_time

        # Save epoch-by-epoch results
        history_df = pd.DataFrame(history)

        history_df.insert(
            0, "epoch", range(1, len(history_df) + 1)
        )

        history_df.insert(0, "normalization", method)

        history_df.to_csv(
            results_dir / f"nn_image_normalization_{method}.csv",
            index=False
        )

        # Find the best validation epoch
        best_index = history_df["val_accuracy"].idxmax()

        summaries.append({
            "normalization": method,
            "best_epoch": int(
                history_df.loc[best_index, "epoch"]
            ),
            "best_val_accuracy": float(
                history_df.loc[best_index, "val_accuracy"]
            ),
            "train_accuracy_at_best_epoch": float(
                history_df.loc[best_index, "train_accuracy"]
            ),
            "val_loss_at_best_epoch": float(
                history_df.loc[best_index, "val_loss"]
            ),
            "training_time_seconds": round(
                training_time, 2
            )
        })

    # Save experiment summary
    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        results_dir / "nn_image_normalization_summary.csv",
        index=False
    )

    print("\nNormalization Experiment Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_normalization_experiments()
