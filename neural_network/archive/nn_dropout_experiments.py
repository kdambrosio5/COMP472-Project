# to not overwrite the baseline model, we will create a new file for dropout experiments

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_network.model import NeuralNetwork, train_model
from neural_network.nn_experiments import load_dataset


RANDOM_STATE = 42

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "image_train.csv"
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)


def run_dropout_experiments():
    X_train, X_val, y_train, y_val = load_dataset(data_path)

    dropout_rates = [0.2, 0.3, 0.5]
    summaries = []

    for dropout in dropout_rates:
        # Reset seeds for reproducible initialization
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)

        print(f"\nTesting dropout = {dropout}\n")

        model = NeuralNetwork(
            input_size=500,
            num_classes=50,
            dropout=dropout
        )

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

        # Save the complete learning history
        history_df = pd.DataFrame(history)
        history_df.insert(
            0, "epoch", range(1, len(history_df) + 1)
        )
        history_df["dropout"] = dropout

        history_df.to_csv(
            results_dir / f"nn_image_dropout_{dropout}.csv",
            index=False
        )

        # Identify the epoch with the highest validation accuracy
        best_index = history_df["val_accuracy"].idxmax()

        summaries.append({
            "dropout": dropout,
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
            )
        })

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        results_dir / "nn_image_dropout_summary.csv",
        index=False
    )

    print("\nDropout Experiment Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_dropout_experiments()
