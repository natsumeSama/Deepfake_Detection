import gdown
import zipfile
import os

# Correct relative path to data folder
data_folder = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(data_folder, exist_ok=True)

# Google Drive file ID
file_id = "1RvERAZT7CjBdA_y4fTENYbG6qib921x9"
output = os.path.join(data_folder, "deepfake_dataset.zip")

# Download from Google Drive into the "data" folder
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Extract ZIP file into the data folder
with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall(data_folder)


os.remove(output)
print("✅ Done! Dataset downloaded and extracted into 'data/'")
