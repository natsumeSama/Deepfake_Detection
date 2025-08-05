import os
import cv2
import numpy as np
import torch

from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List, Callable, Optional


# Automatically determine the number of workers for DataLoader
def get_num_workers() -> int:
    """
    Determines the number of workers for data loading.

    On Windows, multiprocessing with DataLoader can be problematic,
    so this function returns 0 for Windows and the CPU count otherwise.

    Returns:
        int: Number of workers to be used by DataLoader.
    """
    return 0 if os.name == "nt" else os.cpu_count()


NUM_WORKERS = get_num_workers()


class ImageFolderCustom(Dataset):
    """
    Custom dataset class for loading images using OpenCV and applying both
    OpenCV and torchvision transformations.

    This dataset is designed for binary classification tasks such as deepfake detection.

    Attributes:
        paths (List[Path]): List of image file paths.
        transform (Optional[Callable]): OpenCV-based transforms applied before converting to tensor.
        torchvision_transform (Optional[Callable]): Torch-based transforms applied after tensor conversion.
        classes (List[str]): List of class names ("real", "fake").
        class_to_idx (dict): Mapping of class names to integer labels.
    """

    def __init__(
        self,
        targ_dir: str,
        transform: Optional[Callable] = None,
        torchvision_transform: Optional[Callable] = None,
    ) -> None:
        """
        Initializes the custom dataset.

        Args:
            targ_dir (str): Path to the target image directory.
            transform (Optional[Callable]): OpenCV transforms applied before tensor conversion.
            torchvision_transform (Optional[Callable]): Torch transforms applied after tensor conversion.
        """
        self.paths = [
            p
            for p in Path(targ_dir).rglob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        if not self.paths:
            raise RuntimeError(f"No images found in {targ_dir}.")

        self.transform = transform
        self.torchvision_transform = torchvision_transform

        # Define classes and mapping
        self.classes, self.class_to_idx = ["real", "fake"], {"real": 0, "fake": 1}

    def load_image(self, index: int) -> np.ndarray:
        """
        Loads an image from disk using OpenCV.

        Args:
            index (int): Index of the image to load.

        Returns:
            np.ndarray: Loaded image in RGB format.
        """
        image_path = str(self.paths[index])
        img = cv2.imread(image_path)  # BGR format by default
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return img

    def __len__(self) -> int:
        """Returns the total number of images."""
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves a single image and its label.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the transformed image tensor and its class index.
        """
        # Load and resize image
        img = self.load_image(index)
        img = cv2.resize(img, (64, 64))

        # Apply OpenCV transforms if provided
        if self.transform:
            if isinstance(self.transform, list):
                for t in self.transform:
                    img = t(img)
            else:
                img = self.transform(img)

        # Convert to tensor and normalize to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Apply torchvision transforms if provided
        if self.torchvision_transform:
            img = self.torchvision_transform(img)

        # Determine class based on folder name
        name = self.paths[index].parent.name
        class_name = "real" if name.startswith("0") else "fake"
        class_idx = self.class_to_idx[class_name]

        return img, class_idx


# Example of an OpenCV-based transform
def blur(img: np.ndarray) -> np.ndarray:
    """
    Applies Gaussian blur to an image.

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Blurred image.
    """
    return cv2.GaussianBlur(img, (5, 5), 0)


def create_dataloaders(
    dir: str,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    transform: Optional[Callable] = None,
    torchvision_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates train, validation, and test DataLoaders.

    Args:
        dir (str): Directory containing the dataset.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for data loading.
        transform (Optional[Callable]): OpenCV transforms applied before tensor conversion.
        torchvision_transform (Optional[Callable]): Torch transforms applied after tensor conversion.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
            - train_dataloader
            - val_dataloader
            - test_dataloader
            - class_names
    """
    dataset = ImageFolderCustom(dir, transform, torchvision_transform)

    # Split dataset: 80% train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)

    train_data, test_data = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    # Split train into train/val (80/20)
    val_size = int(0.2 * train_size)
    train_size = train_size - val_size
    train_data, val_data = random_split(
        train_data, [train_size, val_size], generator=generator
    )

    class_names = dataset.classes
    pin_mem = torch.cuda.is_available()

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names


if __name__ == "__main__":
    # Torch transforms applied after converting image to tensor
    torch_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Create data loaders
    train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloaders(
        dir="data/deepfakes/db",
        batch_size=32,
        transform=[blur],  # Example of using OpenCV-based transform
        torchvision_transform=torch_transforms,
    )

    # Check a batch of data
    images, labels = next(iter(train_dataloader))
    print(f"Images shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Images min/max: {images.min().item():.4f}/{images.max().item():.4f}")
