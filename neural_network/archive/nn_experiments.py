
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neural_network.model import NeuralNetwork, train_model
from pathlib import Path


# Reproducibility
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def load_dataset(file_path):
    # Load the labeled dataset
    df = pd.read_csv(file_path)

    # Extract numerical features and class labels
    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)

    # Create an 80/20 stratified training-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Fit preprocessing only on training data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert data into PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    return X_train, X_val, y_train, y_val


if __name__ == "__main__":

    X_train, X_val, y_train, y_val = load_dataset(
        "data/image_train.csv"
    )

    print("Image dataset:")
    print("Training features:", X_train.shape)
    print("Validation features:", X_val.shape)
    print("Training labels:", y_train.shape)
    print("Validation labels:", y_val.shape)
    
    
    # Initialize image classification model
    image_model = NeuralNetwork(
        input_size=500,
        num_classes=50
    )

    print("\nTraining Image Neural Network...\n")

    image_history = train_model(
        model=image_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=20,
        batch_size=64,
        learning_rate=0.001
    )
    
    
    # Locate the project results directory
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Save baseline metrics for every epoch
    results_df = pd.DataFrame(image_history)

    results_df.insert(
        0, "epoch", range(1, len(results_df) + 1)
    )

    results_df.insert(0, "experiment", "image_nn_baseline")

    results_df.to_csv(
        results_dir / "nn_image_baseline.csv",
        index=False
    )

    # Summarize baseline performance
    best_epoch = results_df["val_accuracy"].idxmax()

    print("\nBaseline Summary:")
    print(
        "Best validation accuracy:",
        f"{results_df.loc[best_epoch, 'val_accuracy']:.4f}"
    )
    print(
        "Best validation epoch:",
        int(results_df.loc[best_epoch, "epoch"])
    )

    print("\nResults saved to results/nn_image_baseline.csv")


