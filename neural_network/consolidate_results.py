
from pathlib import Path
import pandas as pd

# Locate the project's results directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ARCHIVE_DIR = RESULTS_DIR / "nn_archive"

# Architecture configurations used in the experiments
ARCHITECTURES = {
    "small": (64,),
    "medium": (128, 64),
    "baseline": (256, 64),
    "large": (512, 256),
    "deep": (256, 128, 64),
}


def consolidate_results():
    """Consolidate archived Neural Network experiments into one CSV."""

    records = []

    # Read the existing epoch-by-epoch experiment files
    for file_path in sorted(ARCHIVE_DIR.glob("nn_*.csv")):

        # Skip summary files and previously consolidated results
        if "summary" in file_path.stem or file_path.name == "nn_results.csv":
            continue

        filename = file_path.stem

        dataset = (
            "image" if filename.startswith("nn_image_")
            else "sentiment" if filename.startswith("nn_sentiment_")
            else None
        )

        if dataset is None:
            continue

        experiment = filename.removeprefix(f"nn_{dataset}_")
        df = pd.read_csv(file_path)

        # Skip files without valid validation results
        if df.empty or "val_accuracy" not in df.columns:
            print(f"Skipping invalid file: {file_path.name}")
            continue

        # Recover hyperparameters from the original experiment
        architecture = (256, 64)
        dropout = 0.0
        learning_rate = 0.001
        weight_decay = 0.0
        normalization = "standard"

        # Sentiment experiments generally use dropout = 0.5
        if dataset == "sentiment" and experiment != "baseline":
            dropout = 0.5

        # Architecture experiments
        if experiment.startswith("architecture_"):
            name = experiment.removeprefix("architecture_")
            architecture = ARCHITECTURES[name]

        # Learning rate experiments
        elif experiment.startswith("lr_"):
            architecture = (512, 256)
            learning_rate = float(experiment.removeprefix("lr_"))

        # Dropout experiments
        elif experiment.startswith("dropout_"):
            dropout = float(experiment.removeprefix("dropout_"))

        # Weight decay experiments
        elif experiment.startswith("wd_"):
            architecture = (512, 256)
            weight_decay = float(experiment.removeprefix("wd_"))

        # Normalization experiments
        elif experiment.startswith("normalization_"):
            architecture = (512, 256)
            normalization = experiment.removeprefix("normalization_")

            if dataset == "image":
                weight_decay = 0.001

        # Identify the epoch with the highest validation accuracy
        best_index = df["val_accuracy"].idxmax()
        best_row = df.loc[best_index]

        # Store the best result for this experiment
        records.append({
            "dataset": dataset,
            "experiment": experiment,
            "hidden_layers": str(architecture),
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "normalization": normalization,
            "epochs": len(df),
            "best_epoch": int(best_row["epoch"]),
            "best_val_accuracy": float(best_row["val_accuracy"]),
            "train_accuracy_at_best_epoch": float(
                best_row["train_accuracy"]
            ),
            "val_loss_at_best_epoch": float(
                best_row["val_loss"]
            ),
            "source_file": file_path.name,
        })

    # Create the consolidated results DataFrame
    results_df = pd.DataFrame(records)

    if results_df.empty:
        print("No Neural Network experiment histories found.")
        return

    # Save all experiment summaries in one CSV
    output_path = RESULTS_DIR / "nn_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nConsolidated {len(results_df)} experiments.")
    print(f"Saved to: {output_path}")

    # Display the best experiment for each dataset
    print("\nBest experiment per dataset:")

    for dataset, group in results_df.groupby("dataset"):
        best = group.loc[group["best_val_accuracy"].idxmax()]

        print(
            f"{dataset}: {best['best_val_accuracy']:.2%} "
            f"({best['experiment']})"
        )


# Execute consolidation only when this script is run directly
if __name__ == "__main__":
    consolidate_results()
