
import random
from pathlib import Path

import numpy as np
import torch

from neural_network.model import NeuralNetwork, train_model
from neural_network.experiments import load_data


RANDOM_STATE = 42

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
results_dir = project_root / "results"

results_dir.mkdir(exist_ok=True)


def save_best_model(
    dataset_name,
    input_size,
    num_classes,
    dropout,
    weight_decay
):

    # Reset random seeds for reproducibility
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    print(f"\nTraining best {dataset_name} Neural Network\n")

    # Load the same training-validation split
    X_train, X_val, y_train, y_val = load_data(
    dataset=dataset_name,
    normalization="standard"
    )

    # Initialize the selected architecture
    model = NeuralNetwork(
        input_size=input_size,
        num_classes=num_classes,
        hidden_layers=(512, 256),
        dropout=dropout
    )

    checkpoint_path = (
        results_dir / f"nn_{dataset_name}_best.pth"
    )

    # Train and save the best validation checkpoint
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=20,
        batch_size=64,
        learning_rate=0.001,
        weight_decay=weight_decay,
        checkpoint_path=checkpoint_path
    )

    print(f"\nCheckpoint saved to: {checkpoint_path}")

    return model, history


if __name__ == "__main__":

    # Best image configuration
    image_model, image_history = save_best_model(
        dataset_name="image",
        input_size=500,
        num_classes=50,
        dropout=0.0,
        weight_decay=0.001
    )

    # Best sentiment configuration
    sentiment_model, sentiment_history = save_best_model(
        dataset_name="sentiment",
        input_size=768,
        num_classes=3,
        dropout=0.5,
        weight_decay=0.0
    )
