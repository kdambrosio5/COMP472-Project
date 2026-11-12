
import numpy as np


class KNN:
    def __init__(self, k=5, metric="euclidean", weighted=False):
        if k < 1:
            raise ValueError("k must be at least 1")

        if metric not in ("euclidean", "manhattan"):
            raise ValueError("Unsupported distance metric")

        self.k = k
        self.metric = metric
        self.weighted = weighted

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=int)

        if self.k > len(self.X_train):
            raise ValueError("k cannot exceed training set size")

        if np.any(self.y_train < 0):
            raise ValueError("Class labels must be nonnegative integers")

        if len(self.X_train) != len(self.y_train):
            raise ValueError("Features and labels must have equal lengths")

    def _distances(self, X_batch):
        if self.metric == "euclidean":
            train_squared = np.sum(self.X_train ** 2, axis=1)
            batch_squared = np.sum(X_batch ** 2, axis=1)

            distances_squared = (
                batch_squared[:, None]
                + train_squared[None, :]
                - 2 * X_batch @ self.X_train.T
            )

            return np.sqrt(np.maximum(distances_squared, 0))

        # Manhattan distance, calculated one example at a time
        distances = np.empty(
            (len(X_batch), len(self.X_train))
        )

        for i, x in enumerate(X_batch):
            distances[i] = np.sum(
                np.abs(self.X_train - x),
                axis=1
            )

        return distances

    def predict(self, X_test, batch_size=100):
        X_test = np.asarray(X_test, dtype=float)
        predictions = []

        for start in range(0, len(X_test), batch_size):
            X_batch = X_test[start:start + batch_size]
            distances = self._distances(X_batch)

            # Find k nearest neighbours
            nearest_indices = np.argsort(
                distances, axis=1
            )[:, :self.k]

            nearest_labels = self.y_train[nearest_indices]
            nearest_distances = np.take_along_axis(
                distances, nearest_indices, axis=1
            )

            for labels, dists in zip(
                nearest_labels, nearest_distances
            ):
                if self.weighted:
                    # Exact matches receive all voting influence
                    zero_mask = dists == 0

                    if np.any(zero_mask):
                        predictions.append(
                            np.argmax(np.bincount(labels[zero_mask]))
                        )
                        continue

                    weights = 1.0 / (dists + 1e-8)

                    class_weights = np.bincount(
                        labels,
                        weights=weights
                    )

                    prediction = np.argmax(class_weights)

                else:
                    prediction = np.argmax(
                        np.bincount(labels)
                    )

                predictions.append(prediction)

        return np.array(predictions)
