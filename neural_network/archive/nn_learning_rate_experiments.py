
import random
import time
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


def run_learning_rate_experiments():

    X_train, X_val, y_train, y_val = load_dataset(data_path)

    learning_rates = [0.01, 0.001, 0.0001]

    summaries = []

    for learning_rate in learning_rates:

        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)

        print(f"\nTesting learning rate: {learning_rate}\n")

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
            learning_rate=learning_rate
        )

        training_time = time.perf_counter() - start_time

        history_df = pd.DataFrame(history)

        history_df.insert(
            0, "epoch", range(1, len(history_df) + 1)
        )

        history_df["learning_rate"] = learning_rate

        history_df.to_csv(
            results_dir / f"nn_image_lr_{learning_rate}.csv",
            index=False
        )

        best_index = history_df["val_accuracy"].idxmax()

        summaries.append({
            "learning_rate": learning_rate,
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

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        results_dir / "nn_image_learning_rate_summary.csv",
        index=False
    )

    print("\nLearning Rate Experiment Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_learning_rate_experiments()
