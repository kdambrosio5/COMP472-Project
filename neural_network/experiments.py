
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize

from neural_network.model import NeuralNetwork, train_model


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 64

ARCHITECTURES = {
    "small": (64,),
    "medium": (128, 64),
    "baseline": (256, 64),
    "large": (512, 256),
    "deep": (256, 128, 64),
}


def set_seed():
    """Reset random seeds for reproducible experiments."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


def load_data(dataset, normalization="standard"):
    """Load and preprocess a dataset using the original split."""
    path = DATA_DIR / f"{dataset}_train.csv"
    df = pd.read_csv(path)

    feature_columns = [
        col for col in df.columns if col.startswith("feature_")
    ]

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if normalization == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    elif normalization == "l2":
        X_train = normalize(X_train, norm="l2")
        X_val = normalize(X_val, norm="l2")

    elif normalization != "none":
        raise ValueError(f"Unknown normalization: {normalization}")

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long),
    )


def build_experiments():
    """Reproduce all original experiment configurations."""
    experiments = []

    for dataset in ("image", "sentiment"):
        default_dropout = 0.0 if dataset == "image" else 0.5

        def add(name, hidden=(256, 64), dropout=0.0,
                lr=0.001, wd=0.0, norm="standard"):
            experiments.append({
                "dataset": dataset,
                "experiment": name,
                "hidden_layers": hidden,
                "dropout": dropout,
                "learning_rate": lr,
                "weight_decay": wd,
                "normalization": norm,
            })

        # Original baseline: no dropout for either dataset
        add("baseline")

        # Architecture comparison
        for name, hidden in ARCHITECTURES.items():
            add(
                f"architecture_{name}",
                hidden=hidden,
                dropout=default_dropout,
            )

        # Learning-rate comparison
        for lr in (0.01, 0.001, 0.0001):
            add(
                f"lr_{lr}",
                hidden=(512, 256),
                dropout=default_dropout,
                lr=lr,
            )

        # Dropout comparison
        dropout_values = (
            (0.2, 0.3, 0.5)
            if dataset == "image"
            else (0.0, 0.2, 0.3, 0.5)
        )

        for dropout in dropout_values:
            add(f"dropout_{dropout}", dropout=dropout)

        # Weight-decay comparison
        for wd in (0.0, 0.0001, 0.001, 0.01):
            add(
                f"wd_{wd}",
                hidden=(512, 256),
                dropout=default_dropout,
                wd=wd,
            )

        # Normalization comparison
        for norm in ("none", "standard", "l2"):
            add(
                f"normalization_{norm}",
                hidden=(512, 256),
                dropout=default_dropout,
                wd=0.001 if dataset == "image" else 0.0,
                norm=norm,
            )

    return experiments


def run_experiments(dataset_filter="all", force=False):
    """Run missing experiments and update the consolidated CSV."""
    output_path = RESULTS_DIR / "nn_results.csv"

    if output_path.exists():
        results = pd.read_csv(output_path)
    else:
        results = pd.DataFrame()

    experiments = build_experiments()

    for config in experiments:
        dataset = config["dataset"]
        name = config["experiment"]

        if dataset_filter != "all" and dataset != dataset_filter:
            continue

        # Skip configurations already recorded in nn_results.csv
        already_completed = (
            not results.empty
            and (
                (results["dataset"] == dataset)
                & (results["experiment"] == name)
            ).any()
        )

        if already_completed and not force:
            print(f"SKIP: {dataset} / {name}")
            continue

        print(f"\nRUN: {dataset} / {name}")
        set_seed()

        X_train, X_val, y_train, y_val = load_data(
            dataset, config["normalization"]
        )

        input_size = 500 if dataset == "image" else 768
        num_classes = 50 if dataset == "image" else 3

        model = NeuralNetwork(
            input_size=input_size,
            num_classes=num_classes,
            hidden_layers=config["hidden_layers"],
            dropout=config["dropout"],
        )

        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        best_index = int(np.argmax(history["val_accuracy"]))

        record = {
            "dataset": dataset,
            "experiment": name,
            "hidden_layers": str(config["hidden_layers"]),
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "normalization": config["normalization"],
            "epochs": EPOCHS,
            "best_epoch": best_index + 1,
            "best_val_accuracy": history["val_accuracy"][best_index],
            "train_accuracy_at_best_epoch":
                history["train_accuracy"][best_index],
            "val_loss_at_best_epoch":
                history["val_loss"][best_index],
            "source_file": "consolidated_runner",
        }

        # Replace an existing record when --force is used
        if already_completed:
            results = results[
                ~(
                    (results["dataset"] == dataset)
                    & (results["experiment"] == name)
                )
            ]

        results = pd.concat(
            [results, pd.DataFrame([record])],
            ignore_index=True,
        )

        results.to_csv(output_path, index=False)

        print(
            f"Best validation accuracy: "
            f"{record['best_val_accuracy']:.2%}"
        )

    print(f"\nResults saved to: {output_path}")
    print(f"Total experiment records: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["all", "image", "sentiment"],
        default="all",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun and replace existing experiment results",
    )

    args = parser.parse_args()
    run_experiments(args.dataset, args.force)
