
import numpy as np
from sklearn.metrics import accuracy_score

from knn.model import KNN
from knn.experiments import load_dataset, preprocess


# Load the image dataset using the established split
X_train, X_val, y_train, y_val = load_dataset("image")

# Use the original, unnormalized features
X_train, X_val = preprocess(
    X_train, X_val, normalization="none"
)

# Test only 100 validation examples
X_val_small = X_val[:100]
y_val_small = y_val[:100]

# Test Euclidean distance with k=1
model = KNN(
    k=1,
    metric="euclidean",
    weighted=False
)

model.fit(X_train, y_train)
predictions = model.predict(X_val_small, batch_size=20)

accuracy = accuracy_score(y_val_small, predictions)

print("Training shape:", X_train.shape)
print("Validation examples tested:", len(X_val_small))
print(f"Accuracy: {accuracy:.2%}")

# Verify prediction count and class labels
assert len(predictions) == len(y_val_small)
assert np.all(np.isin(predictions, y_train))

print("Verification passed.")
