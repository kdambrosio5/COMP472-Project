
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_network.model import NeuralNetwork, train_model
from neural_network.nn_experiments import load_dataset


RANDOM_STATE = 42

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "sentiment_train.csv"
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)


def run_weight_decay_experiments():

    X_train, X_val, y_train, y_val = load_dataset(data_path)

    weight_decay_values = [0.0, 0.0001, 0.001, 0.01]

    summaries = []

    for weight_decay in weight_decay_values:

        # Reset seeds for reproducibility
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)

        print(f"\nTesting sentiment weight decay: {weight_decay}\n")

        model = NeuralNetwork(
            input_size=768,
            num_classes=3,
            hidden_layers=(512, 256),
            dropout=0.5
        )

        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=20,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=weight_decay
        )

        history_df = pd.DataFrame(history)

        history_df.insert(
            0, "epoch", range(1, len(history_df) + 1)
        )

        history_df.insert(0, "weight_decay", weight_decay)

        history_df.to_csv(
            results_dir / f"nn_sentiment_wd_{weight_decay}.csv",
            index=False
        )

        best_index = history_df["val_accuracy"].idxmax()

        summaries.append({
            "weight_decay": weight_decay,
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
        results_dir / "nn_sentiment_weight_decay_summary.csv",
        index=False
    )

    print("\nSentiment Weight Decay Experiment Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_weight_decay_experiments()
