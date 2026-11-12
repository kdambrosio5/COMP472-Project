
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_network.model import NeuralNetwork, train_model
from neural_network.nn_experiments import load_dataset


RANDOM_STATE = 42

# Define project directories
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "sentiment_train.csv"
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)


def run_sentiment_baseline():

    # Set random seeds for reproducibility
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # Load sentiment dataset using the same 80/20 split
    X_train, X_val, y_train, y_val = load_dataset(data_path)

    print("Sentiment training features:", X_train.shape)
    print("Sentiment validation features:", X_val.shape)
    print("Sentiment training labels:", y_train.shape)
    print("Sentiment validation labels:", y_val.shape)

    # Initialize sentiment Neural Network
    model = NeuralNetwork(
        input_size=768,
        num_classes=3,
        hidden_layers=(256, 64),
        dropout=0.0
    )

    print("\nTraining sentiment Neural Network baseline\n")

    # Train model
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=20,
        batch_size=64,
        learning_rate=0.001,
        weight_decay=0.0
    )

    # Save epoch-by-epoch results
    results_df = pd.DataFrame(history)

    results_df.insert(
        0, "epoch", range(1, len(results_df) + 1)
    )

    results_df.insert(0, "experiment", "sentiment_nn_baseline")

    results_df.to_csv(
        results_dir / "nn_sentiment_baseline.csv",
        index=False
    )

    # Find epoch with highest validation accuracy
    best_index = results_df["val_accuracy"].idxmax()

    best_epoch = int(results_df.loc[best_index, "epoch"])
    best_accuracy = float(
        results_df.loc[best_index, "val_accuracy"]
    )

    print("\nSentiment Neural Network Baseline Summary:")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Best validation accuracy (%): {best_accuracy * 100:.2f}%")
    print(
        "Training accuracy at best epoch: "
        f"{results_df.loc[best_index, 'train_accuracy']:.4f}"
    )
    print(
        "Validation loss at best epoch: "
        f"{results_df.loc[best_index, 'val_loss']:.4f}"
    )

    print("\nResults saved to results/nn_sentiment_baseline.csv")


if __name__ == "__main__":
    run_sentiment_baseline()
