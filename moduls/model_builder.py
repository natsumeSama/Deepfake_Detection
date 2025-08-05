import torch
from torch import nn


class CNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for binary classification.

    This model is designed for images of size 64x64, either RGB (3 channels) or grayscale (1 channel).

    Architecture:
        - Conv Block 1: Conv2D → ReLU → MaxPool
        - Conv Block 2: Conv2D → ReLU → MaxPool → Dropout
        - Classifier: Flatten → Linear → ReLU → Linear (binary output)

    Attributes:
        flatten_size (int): The number of features after the convolutional blocks (calculated dynamically).
        conv_block_1 (nn.Sequential): First convolutional block.
        conv_block_2 (nn.Sequential): Second convolutional block.
        classifier (nn.Sequential): Fully connected layers for classification.
    """

    def __init__(self, input_shape: int):
        """
        Initializes the CNN.

        Args:
            input_shape (int): Number of input channels (3 for RGB or 1 for grayscale).
        """
        super().__init__()

        # First convolutional block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, 32, kernel_size=3, stride=1, padding=0),  # 64 → 62
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 62 → 31
        )

        # Second convolutional block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),  # 31 → 29
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 29 → 14
            nn.Dropout(0.5),  # Dropout for regularization
        )

        # Dynamically compute the flatten size after convolution
        with torch.no_grad():
            dummy = torch.zeros(1, input_shape, 64, 64)  # Dummy input
            out = self.conv_block_2(self.conv_block_1(dummy))
            self.flatten_size = out.numel()  # Number of features after conv layers

        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Linear(
                64, 1
            ),  # Single output for binary classification (with BCEWithLogitsLoss)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1).
        """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)


if __name__ == "__main__":
    # Test the CNN with RGB images
    print("Testing RGB CNN:")
    model_rgb = CNN(input_shape=3)
    print(model_rgb)
    print("Output shape:", model_rgb(torch.randn(1, 3, 64, 64)).shape)

    print("\nTesting Grayscale CNN:")
    # Test the CNN with grayscale images
    model_gray = CNN(input_shape=1)
    print(model_gray)
    print("Output shape:", model_gray(torch.randn(1, 1, 64, 64)).shape)
