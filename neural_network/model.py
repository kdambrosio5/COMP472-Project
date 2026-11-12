
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes,
        hidden_layers=(256, 64),
        dropout=0.0
    ):
        super().__init__()

        layers = []
        previous_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=20,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.0,
    checkpoint_path=None
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_accuracy = -1.0
    best_epoch = 0
    best_model_weights = None

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

            correct += (
                outputs.argmax(dim=1) == y_batch
            ).sum().item()

            total += X_batch.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total

        model.eval()

        with torch.no_grad():

            val_outputs = model(X_val)

            val_loss = criterion(
                val_outputs, y_val
            ).item()

            val_accuracy = (
                val_outputs.argmax(dim=1) == y_val
            ).float().mean().item()

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        # Preserve weights from the best validation epoch
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

            if checkpoint_path is not None:

                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": best_model_weights,
                        "val_accuracy": best_val_accuracy
                    },
                    checkpoint_path
                )

    # Restore best weights before returning
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    print(
        f"\nBest Epoch: {best_epoch} | "
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    return history
