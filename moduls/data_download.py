import os
import zipfile
from pathlib import Path
import gdown


def data_download(file_id):
    data_path = Path("data/")
    dataset_path = data_path / "deepfakes" / "db"

    # Skip download if dataset already exists
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"✅ Dataset already exists at {dataset_path}")
        return dataset_path

    print(f"📂 Creating dataset directory at {dataset_path}")
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Download the zip file using gdown
    zip_path = data_path / "deepfake.zip"
    print("⬇️ Downloading dataset ...")
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}", str(zip_path), quiet=False
    )

    # Unzip the file
    print("📦 Extracting dataset ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    zip_path.unlink()  # Remove the zip file
    print(f"✅ Dataset ready at {dataset_path}")

    return dataset_path


def data_info(path):
    path = Path(path)
    if not path.exists():
        print(f"❌ Path {path} does not exist.")
        return

    real_count = 0
    fake_count = 0

    for folder in sorted(path.iterdir()):
        if folder.is_dir():
            images = list(folder.iterdir())
            count = len(images)

            if folder.name.startswith("0"):
                real_count += count
            elif folder.name.startswith("1"):
                fake_count += count

            print(f"{folder.name}: {count} images")

    print(f"Total real images: {real_count}")
    print(f"Total fake images: {fake_count}")


if __name__ == "__main__":
    data_download("1RvERAZT7CjBdA_y4fTENYbG6qib921x9")
    data_folder = os.path.join(os.path.dirname(__file__), "../data/deepfakes/db")
    data_info(data_folder)
