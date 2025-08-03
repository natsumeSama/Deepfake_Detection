import os
import cv2
import numpy as np
import pathlib
import torch

from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, List, Callable, Optional


class ImageFolderCustom(Dataset):
    """
    Custom dataset loader for image classification using OpenCV and PyTorch.

    This dataset:
      - Loads images from a directory organized in subfolders by class.
      - Supports custom OpenCV (NumPy-based) transformations.
      - Can be combined later with torchvision transforms (after converting to Tensor).
      - Automatically assigns labels based on folder naming conventions.

    Folder structure (example):
    ├── data/
        ├── 0_real/
        │     ├── img1.jpg
        │     └── img2.png
        ├── 1_fake/
              ├── img3.jpeg
              └── img4.png

    Args:
        targ_dir (str): Path to the target directory containing class folders.
        transform (Callable | List[Callable], optional): OpenCV/NumPy transforms to apply before converting to tensor.
        torchvision_transform (Callable, optional): torchvision transforms to apply after converting to tensor.
    """

    def __init__(
        self,
        targ_dir: str,
        transform: Optional[Callable] = None,
        torchvision_transform: Optional[Callable] = None,
    ) -> None:

        # Collect all image paths with valid extensions
        self.paths = [
            p
            for p in pathlib.Path(targ_dir).rglob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        if not self.paths:
            raise RuntimeError(f"No images found in {targ_dir}.")

        # Store transforms
        self.transform = transform  # OpenCV/NumPy-based transforms
        self.torchvision_transform = (
            torchvision_transform  # PyTorch transforms (after tensor conversion)
        )

        # Define class mapping
        self.classes, self.class_to_idx = ["real", "fake"], {"real": 0, "fake": 1}

    def load_image(self, index: int) -> np.ndarray:
        """
        Loads an image from disk using OpenCV.

        Args:
            index (int): Index of the image in the dataset.

        Returns:
            np.ndarray: Loaded image in RGB format.
        """
        image_path = str(self.paths[index])  # cv2 needs string path
        img = cv2.imread(image_path)  # Loads in BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return img

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves an image and its label by index.

        Args:
            index (int): Index of the image.

        Returns:
            Tuple[torch.Tensor, int]: Image tensor (C, H, W) and label index.
        """
        img = self.load_image(index)

        # Resize image to a fixed size (can be parameterized later)
        img = cv2.resize(img, (256, 256))  # (width, height)

        # Apply OpenCV/NumPy-based transforms
        if self.transform:
            if isinstance(self.transform, list):  # Multiple transforms
                for t in self.transform:
                    img = t(img)
            else:  # Single transform
                img = self.transform(img)

        # Convert to tensor and normalize to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Apply torchvision transforms (after tensor conversion)
        if self.torchvision_transform:
            img = self.torchvision_transform(img)

        # Determine class from folder name
        name = self.paths[index].parent.name
        class_name = "real" if name.startswith("0") else "fake"
        class_idx = self.class_to_idx[class_name]

        return img, class_idx


if __name__ == "__main__":
    import cv2
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # Example OpenCV transform
    def blur(img):
        """Applies Gaussian blur using OpenCV."""
        return cv2.GaussianBlur(img, (5, 5), 0)

    # Example Torch transforms (applied after tensor conversion)
    torch_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Create dataset instance
    dataset = ImageFolderCustom(
        targ_dir="data/deepfakes/db",
        transform=[blur],  # OpenCV transforms
        torchvision_transform=torch_transforms,  # PyTorch transforms
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Fetch one batch
    images, labels = next(iter(dataloader))

    print(f"Images shape: {images.shape}")  # [B, 3, H, W]
    print(f"Labels: {labels}")
    print(f"Images min/max: {images.min().item():.4f}/{images.max().item():.4f}")
