import os
import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms

# ==========================
# Configuration
# ==========================
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DATA_DIR = "data/deepfakes/db"

# Select device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Data Transformations
# ==========================
# For RGB model (no extra preprocessing)
rgb_transform = transforms.Compose([])

# For Grayscale model (convert to single channel)
gray_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])


def train_and_save(transform, input_channels: int, model_name: str) -> None:
    """
    Train a CNN model with the given transformation and save it.

    Args:
        transform (torchvision.transforms.Compose): Transformations applied to the dataset.
        input_channels (int): Number of input channels (3 for RGB, 1 for Grayscale).
        model_name (str): Name of the saved model file.

    Returns:
        None
    """
    # Create data loaders
    train_dataloader, val_dataloader, _, class_names = data_setup.create_dataloaders(
        dir=DATA_DIR, batch_size=BATCH_SIZE, torchvision_transform=transform
    )

    # Initialize CNN model
    model = model_builder.CNN(input_shape=input_channels).to(device)

    # Define loss function and optimizer
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device,
    )

    # Save the trained model
    utils.save_model(model=model, target_dir="models", model_name=model_name)


# ==========================
# Training
# ==========================

# Train and save RGB model
train_and_save(rgb_transform, input_channels=3, model_name="CNN_RGB.pth")

# Train and save Grayscale model
train_and_save(gray_transform, input_channels=1, model_name="CNN_GRAY.pth")
