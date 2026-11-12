
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_network.model import NeuralNetwork, train_model
from neural_network.nn_experiments import load_dataset


RANDOM_STATE = 42

# Define project directories
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "image_train.csv"
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)


def run_architecture_experiments():

    # Load the same training and validation subsets
    X_train, X_val, y_train, y_val = load_dataset(data_path)

    # Architectures to evaluate
    architectures = {
        "small": (64,),
        "medium": (128, 64),
        "baseline": (256, 64),
        "large": (512, 256),
        "deep": (256, 128, 64)
    }

    summaries = []

    for name, hidden_layers in architectures.items():

        # Reset random seeds for reproducibility
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)

        print(f"\nTesting architecture: {name}")
        print(f"Hidden layers: {hidden_layers}\n")

        model = NeuralNetwork(
            input_size=500,
            num_classes=50,
            hidden_layers=hidden_layers,
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
            learning_rate=0.001
        )

        training_time = time.perf_counter() - start_time

        # Save epoch-by-epoch results
        history_df = pd.DataFrame(history)

        history_df.insert(
            0, "epoch", range(1, len(history_df) + 1)
        )

        history_df.insert(0, "architecture", name)

        history_df.to_csv(
            results_dir / f"nn_image_architecture_{name}.csv",
            index=False
        )

        # Identify best validation epoch
        best_index = history_df["val_accuracy"].idxmax()

        summaries.append({
            "architecture": name,
            "hidden_layers": str(hidden_layers),
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
            "training_time_seconds": round(training_time, 2)
        })

    # Save architecture comparison
    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        results_dir / "nn_image_architecture_summary.csv",
        index=False
    )

    print("\nArchitecture Experiment Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_architecture_experiments()
