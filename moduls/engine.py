import torch
from typing import Tuple
from copy import deepcopy


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Perform one training step (one full pass through the training dataloader).

    Args:
        model (torch.nn.Module): The neural network model to train.
        dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        loss_fn (torch.nn.Module): Loss function (e.g., BCEWithLogitsLoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam).
        device (torch.device): Device to use ('cpu' or 'cuda').

    Returns:
        Tuple[float, float]: Average training loss and accuracy for the epoch.
    """
    model.train()  # Set model to training mode
    train_loss, train_acc = 0.0, 0.0

    for X, y in dataloader:
        # Move data to device
        X, y = X.to(device), y.to(device).unsqueeze(1).float()

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        train_loss += loss.item()

        # Calculate accuracy
        y_pred_class = torch.round(torch.sigmoid(y_pred))
        train_acc += (y_pred_class == y).sum().item() / len(y)

    # Return mean loss and accuracy over the epoch
    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Perform one validation/testing step (no gradient updates).

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader for validation/test data.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device): Device to use ('cpu' or 'cuda').

    Returns:
        Tuple[float, float]: Average test loss and accuracy for the epoch.
    """
    model.eval()  # Set model to evaluation mode
    test_loss, correct, total = 0.0, 0, 0

    with torch.inference_mode():  # Disable gradient calculation
        for X, y in dataloader:
            # Move data to device
            X, y = X.to(device), y.to(device).unsqueeze(1).float()

            # Forward pass
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Accuracy calculation
            y_pred_class = torch.round(torch.sigmoid(y_pred))
            correct += (y_pred_class == y).sum().item()
            total += y.size(0)

    return test_loss / len(dataloader), correct / total


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, float]:
    """
    Train the model for a given number of epochs and track the best model.

    A custom score is used to select the best model:
    score = 0.5 * (1 - normalized test loss) + 0.5 * test accuracy

    Args:
        model (torch.nn.Module): Model to train.
        train_dataloader (torch.utils.data.DataLoader): Training dataloader.
        test_dataloader (torch.utils.data.DataLoader): Validation/test dataloader.
        loss_fn (torch.nn.Module): Loss function (e.g., BCEWithLogitsLoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam).
        epochs (int): Number of training epochs.
        device (torch.device): Device to use ('cpu' or 'cuda').

    Returns:
        Tuple[torch.nn.Module, float]: Best model (with best weights loaded) and best score achieved.
    """
    best_score = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Training and validation steps
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device
        )
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        # Compute custom score (balance between low loss and high accuracy)
        normalized_loss = max(
            0.0, 1 - min(test_loss, 1)
        )  # Cap to avoid negative values
        score = 0.5 * normalized_loss + 0.5 * test_acc

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.4f} | Score: {score:.4f}"
        )

        # Save best model based on score
        if score > best_score:
            best_score = score
            best_model_state = deepcopy(model.state_dict())
            print(f"🔥 New best model in memory (Score: {best_score:.4f})")

    # Load the best model weights (if found)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("✅ Loaded best model from memory")

    return model, best_score
