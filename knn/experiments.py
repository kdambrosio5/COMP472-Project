
import os
import csv
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn.model import KNN

# Define the hyperparameter combinations to evaluate.
# Each configuration specifies:
# - dataset: image or sentiment classification
# - metric: distance function used to find nearest neighbours
# - normalization: preprocessing applied to feature vectors
# - weighted: whether closer neighbours receive more voting weight
# - k_values: numbers of nearest neighbours to evaluate
#
# Each k value represents a separate experiment.
# All configurations use the same train/validation split
# to ensure that their results are comparable.

EXPERIMENTS = [
    # Image dataset: Euclidean distance without normalization
    # Compare unweighted and distance-weighted voting.
    {
        "dataset": "image",
        "metric": "euclidean",
        "normalization": "none",
        "weighted": False,
        "k_values": [1, 3, 5, 10, 20]
    },
    {
        "dataset": "image",
        "metric": "euclidean",
        "normalization": "none",
        "weighted": True,
        "k_values": [1, 3, 5, 10, 20]
    },

    # Image dataset: Manhattan distance
    # Investigate how changing the distance function affects accuracy.
    {
        "dataset": "image",
        "metric": "manhattan",
        "normalization": "none",
        "weighted": False,
        "k_values": [1, 3, 5, 10, 20]
    },
    {
        "dataset": "image",
        "metric": "manhattan",
        "normalization": "none",
        "weighted": True,
        "k_values": [1, 3, 5, 10, 20]
    },

    # Image dataset: L2 normalization
    # Normalize each feature vector to unit length before classification.
    {
        "dataset": "image",
        "metric": "euclidean",
        "normalization": "l2",
        "weighted": False,
        "k_values": [1, 3, 5, 10, 20]
    },
    {
        "dataset": "image",
        "metric": "euclidean",
        "normalization": "l2",
        "weighted": True,
        "k_values": [1, 3, 5, 10, 20]
    },

    # Sentiment dataset: Original BERT embeddings
    # Test larger neighbourhood sizes because the dataset has
    # only three sentiment classes.
    {
        "dataset": "sentiment",
        "metric": "euclidean",
        "normalization": "none",
        "weighted": False,
        "k_values": [1, 3, 5, 10, 20, 30, 50]
    },
    {
        "dataset": "sentiment",
        "metric": "euclidean",
        "normalization": "none",
        "weighted": True,
        "k_values": [1, 3, 5, 10, 20, 30, 50]
    },

    # Sentiment dataset: L2-normalized embeddings
    # Compare classification accuracy before and after normalization.
    {
        "dataset": "sentiment",
        "metric": "euclidean",
        "normalization": "l2",
        "weighted": False,
        "k_values": [1, 3, 5, 10, 20, 30, 50]
    },
    {
        "dataset": "sentiment",
        "metric": "euclidean",
        "normalization": "l2",
        "weighted": True,
        "k_values": [1, 3, 5, 10, 20, 30, 50]
    },

    # Sentiment dataset: Additional experiments with larger k values
    # Investigate whether increasing k beyond 50 improves accuracy.
    {
        "dataset": "sentiment",
        "metric": "euclidean",
        "normalization": "none",
        "weighted": True,
        "k_values": [60, 75, 100, 150, 200]
    },

    # Sentiment dataset: Refine the search around promising k values
    # to identify the best neighbourhood size more precisely.
    {
        "dataset": "sentiment",
        "metric": "euclidean",
        "normalization": "none",
        "weighted": True,
        "k_values": [65, 70, 80, 85, 90, 95]
    }
]
# Dataset Loading

def load_dataset(dataset):
    """
    Load the specified dataset and create a reproducible
    stratified 80/20 training-validation split.
    """

    # Read the labeled training dataset from the data folder.
    df = pd.read_csv(f"data/{dataset}_train.csv")

    # Extract numerical feature vectors.
    # Exclude 'id' because it is only a sample identifier,
    # and 'label' because it is the target variable.
    X = df.drop(columns=["id", "label"]).to_numpy()

    # Extract the correct class labels.
    y = df["label"].to_numpy()

    # Reserve 80% of the labeled examples for training
    # and 20% for validation.
    #
    # random_state=42 ensures reproducible splits.
    # stratify=y preserves class proportions in both subsets.
    #
    # Image:     16,000 training / 4,000 validation
    # Sentiment:  9,600 training / 2,400 validation
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# Feature Preprocessing

def preprocess(X_train, X_val, normalization):
    """
    Apply the selected preprocessing method.

    Supported methods:
    - none: Original feature values
    - l2: Unit-length normalization per example
    - standard: Feature-wise standardization using
      training-set statistics only
    """

    # No preprocessing: retain the original feature values.
    if normalization == "none":
        return X_train, X_val

    # L2 normalization:
    # Divide each feature vector by its Euclidean norm
    # so that its length becomes 1 (unless it is a zero vector).
    if normalization == "l2":

        # Calculate the norm of each training example.
        train_norms = np.linalg.norm(
            X_train, axis=1, keepdims=True
        )

        # Calculate the norm of each validation example.
        val_norms = np.linalg.norm(
            X_val, axis=1, keepdims=True
        )

        # Prevent division by zero for zero-length vectors.
        train_norms[train_norms == 0] = 1
        val_norms[val_norms == 0] = 1

        # Return the normalized training and validation features.
        return X_train / train_norms, X_val / val_norms

    # Standardization:
    # Transform each feature to have approximately zero mean
    # and unit variance based on the training subset.
    if normalization == "standard":

        # Calculate statistics using training data only
        # to prevent information leakage from validation data.
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        # Avoid division by zero for constant-valued features.
        std[std == 0] = 1

        # Apply the same training statistics to both subsets.
        return (
            (X_train - mean) / std,
            (X_val - mean) / std
        )

    # Reject unsupported preprocessing options.
    raise ValueError(
        f"Unsupported normalization method: {normalization}"
    )

# Experiment Results Configuration


# Store all k-NN experiment results in a single CSV file.
RESULTS_FILE = "results/knn_results.csv"

# Define the columns recorded for every experiment.
COLUMNS = [
    "dataset",
    "k",
    "metric",
    "normalization",
    "weighted",
    "accuracy",
    "train_size",
    "validation_size",
    "random_state"
]

# Experiment Identification

def experiment_key(dataset, k, metric, normalization, weighted):
    """
    Return a unique identifier for an experiment.
    """

    # Use a tuple of hyperparameters to identify each experiment.
    # This allows the program to detect configurations
    # that have already been evaluated.
    return (
        str(dataset),
        int(k),
        str(metric),
        str(normalization),
        bool(weighted)
    )


def initialize_results():

    # Create the results file if it does not already exist.
    

    # Create the results directory if necessary.
    os.makedirs("results", exist_ok=True)

    # Only create a new CSV when one does not already exist.
    # This prevents overwriting previously completed experiments.
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=COLUMNS
            )

            # Write the CSV column headers.
            writer.writeheader()


# Load previously completed experiments to avoid repeating them.

def load_completed_experiments():
    """
    Read existing results to identify experiments
    that have already been completed.
    """

    # Store experiment identifiers in a set for fast lookup.
    completed = set()

    # If no results file exists, no experiments are completed.
    if not os.path.exists(RESULTS_FILE):
        return completed

    # Read each previously saved experiment.
    with open(RESULTS_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:

            # CSV values are strings, so convert the weighted
            # field back into a Boolean value.
            weighted = row["weighted"].strip().lower() == "true"

            # Reconstruct the experiment's unique identifier.
            key = experiment_key(
                row["dataset"],
                int(row["k"]),
                row["metric"],
                row["normalization"],
                weighted
            )

            # Mark the experiment as already completed.
            completed.add(key)

    return completed

# Save experiment results to CSV

def save_result(result):

    # Append the new result without replacing existing records.
    with open(RESULTS_FILE, "a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS
        )

        writer.writerow(result)

# Main experiment


def main():

    # Ensure the results file exists before starting.
    initialize_results()

    # Identify experiments that were completed in previous runs.
    completed = load_completed_experiments()

    print(
        f"Previously completed experiments: {len(completed)}",
        flush=True
    )

    # Iterate through every predefined experiment configuration.
    for config in EXPERIMENTS:

        # Extract the selected hyperparameters.
        dataset = config["dataset"]
        metric = config["metric"]
        normalization = config["normalization"]
        weighted = config["weighted"]

        # Identify k values that have not yet been evaluated.
        remaining_k = []

        for k in config["k_values"]:

            # Generate the unique identifier for this configuration.
            key = experiment_key(
                dataset,
                k,
                metric,
                normalization,
                weighted
            )

            # Skip configurations already recorded in the CSV.
            if key in completed:
                print(
                    f"Skipping completed experiment: "
                    f"{dataset}, k={k}, {metric}, "
                    f"{normalization}, weighted={weighted}",
                    flush=True
                )
            else:
                remaining_k.append(k)

        # Avoid loading and preprocessing the dataset if
        # every k value in this group has already been tested.
        if not remaining_k:
            continue

        # Display the configuration currently being evaluated.
        print(
            f"\nDataset: {dataset}, "
            f"Metric: {metric}, "
            f"Normalization: {normalization}, "
            f"Weighted: {weighted}",
            flush=True
        )

        # Load the full labeled dataset and create the
        # reproducible 80/20 training-validation split.
        X_train, X_val, y_train, y_val = load_dataset(
            dataset
        )

        # Apply the selected preprocessing method.
        X_train, X_val = preprocess(
            X_train,
            X_val,
            normalization
        )

        # Evaluate each remaining k value independently.
        for k in remaining_k:

            # Initialize the custom k-NN classifier with
            # the current hyperparameter configuration.
            model = KNN(
                k=k,
                metric=metric,
                weighted=weighted
            )

            # Store the training feature vectors and labels.
            # Unlike parametric models, k-NN does not learn
            # model weights during fitting.
            model.fit(X_train, y_train)

            # Predict labels for every validation example.
            # Batching limits the memory required for
            # pairwise distance calculations.
            predictions = model.predict(
                X_val,
                batch_size=100
            )

            # Calculate the proportion of correctly
            # classified validation examples.
            accuracy = accuracy_score(
                y_val,
                predictions
            )

            # Record the hyperparameters, validation accuracy,
            # dataset sizes, and random seed for reproducibility.
            result = {
                "dataset": dataset,
                "k": k,
                "metric": metric,
                "normalization": normalization,
                "weighted": weighted,
                "accuracy": accuracy,
                "train_size": len(X_train),
                "validation_size": len(X_val),
                "random_state": 42
            }

            # Save immediately after each experiment so that
            # completed results survive an interrupted run.
            save_result(result)

            # Update the completed set to prevent duplicate
            # experiments during the current execution.
            completed.add(
                experiment_key(
                    dataset,
                    k,
                    metric,
                    normalization,
                    weighted
                )
            )

            # Display the validation accuracy for this experiment.
            print(
                f"  k={k}: {accuracy:.2%} - Saved",
                flush=True
            )

   # Experiment Summary

    # Load all recorded experiments, including previous runs.
    results_df = pd.read_csv(RESULTS_FILE)

    print("\n--- EXPERIMENT SUMMARY ---")

    # Display the total number of completed configurations.
    print(
        f"Total recorded experiments: {len(results_df)}"
    )

    # Identify the best-performing configuration separately
    # for image classification and sentiment classification.
    for dataset in ["image", "sentiment"]:

        # Filter results for the current dataset.
        subset = results_df[
            results_df["dataset"] == dataset
        ]

        # Skip datasets with no recorded experiments.
        if subset.empty:
            continue

        # Select the configuration with the highest
        # validation accuracy.
        best = subset.loc[
            subset["accuracy"].idxmax()
        ]

        # Display the selected configuration and its accuracy.
        print(
            f"\nBest {dataset} configuration:"
        )

        print(
            f"Accuracy: {best['accuracy']:.2%}"
        )
        print(f"k: {best['k']}")
        print(f"Metric: {best['metric']}")
        print(
            f"Normalization: {best['normalization']}"
        )
        print(f"Weighted: {best['weighted']}")

    # Confirm where all experimental results are stored.
    print(
        f"\nAll results saved to {RESULTS_FILE}"
    )


# Execute the experiment runner only when this file is
# run directly, not when its functions are imported.
if __name__ == "__main__":
    main()
