import os
import zipfile
from pathlib import Path
import gdown


def data_download(file_id: str) -> Path:
    """
    Downloads and extracts the deepfake dataset if it does not already exist.

    This function checks if the dataset is already available locally.
    If not, it downloads the dataset from Google Drive using `gdown`, extracts it,
    and removes the downloaded ZIP file.

    Args:
        file_id (str): Google Drive file ID of the dataset.

    Returns:
        Path: Path to the extracted dataset directory.
    """
    # Ensure the data folder is created outside the 'modules' folder
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data"
    dataset_path = data_path / "deepfakes" / "db"

    # Create base data directory if it doesn't exist
    data_path.mkdir(parents=True, exist_ok=True)

    # Check if dataset already exists and contains files
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"✅ Dataset already exists at {dataset_path}")
        return dataset_path

    print(f"📂 Creating dataset directory at {dataset_path}")
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Download the dataset as a ZIP file
    zip_path = data_path / "deepfake.zip"
    print("⬇️ Downloading dataset ...")
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}", str(zip_path), quiet=False
    )

    # Extract the dataset ZIP file
    print("📦 Extracting dataset ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    # Remove the ZIP file after extraction
    zip_path.unlink()
    print(f"✅ Dataset ready at {dataset_path}")

    return dataset_path


def data_info(path: str) -> None:
    """
    Displays basic information about the dataset.

    This function scans the dataset directory and counts the number of
    images in "real" and "fake" categories based on folder names.

    Args:
        path (str): Path to the dataset directory.

    Returns:
        None
    """
    path = Path(path)
    if not path.exists():
        print(f"❌ Path {path} does not exist.")
        return

    real_count, fake_count = 0, 0

    # Iterate through folders and count images
    for folder in sorted(path.iterdir()):
        if folder.is_dir():
            images = list(folder.iterdir())
            count = len(images)

            # Categorize images based on folder name
            if folder.name.startswith("0"):
                real_count += count
            elif folder.name.startswith("1"):
                fake_count += count

            print(f"{folder.name}: {count} images")

    print(f"📊 Total real images: {real_count}")
    print(f"📊 Total fake images: {fake_count}")


if __name__ == "__main__":
    # Download dataset if not already present
    dataset_path = data_download("1RvERAZT7CjBdA_y4fTENYbG6qib921x9")

    # Show dataset info
    data_folder = Path(__file__).resolve().parent.parent / "data" / "deepfakes" / "db"
    data_info(data_folder)
