import torch
import data_setup
import model_builder
import engine
from pathlib import Path
from torchvision import transforms
from typing import Tuple

# ==========================
# Configuration
# ==========================
BATCH_SIZE = 32
MODEL_DIR = Path(__file__).parent / "../models"
DATA_DIR = "data/deepfakes/db"
device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = torch.nn.BCEWithLogitsLoss()


def evaluate_model(model_name: str, input_channels: int) -> Tuple[float, float]:
    """
    Evaluate a model (RGB or Grayscale) on the test dataset.

    Args:
        model_name (str): Name of the model file (e.g. 'CNN_RGB.pth' or 'CNN_GRAY.pth').
        input_channels (int): Number of input channels (3 for RGB, 1 for Grayscale).

    Returns:
        Tuple[float, float]: (test_loss, test_accuracy)
    """
    print(f"\n🔍 Evaluating model: {model_name}")

    # 1. Select appropriate transform
    transform = (
        transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        if input_channels == 1
        else transforms.Compose([])
    )

    # 2. Create dataloaders with the transform
    _, _, test_dataloader, _ = data_setup.create_dataloaders(
        dir=DATA_DIR, batch_size=BATCH_SIZE, torchvision_transform=transform
    )

    # 3. Load the model
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    model = model_builder.CNN(input_shape=input_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Evaluate
    test_loss, test_acc = engine.test_step(
        model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
    )

    print(f"✅ {model_name} → Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

    return test_loss, test_acc


def evaluate_all_models():
    """
    Evaluate both the RGB and Grayscale models and display their performance.
    """
    print("📊 Starting evaluation for all models...")
    results = {}

    results["RGB"] = evaluate_model("CNN_RGB.pth", input_channels=3)
    results["Grayscale"] = evaluate_model("CNN_GRAY.pth", input_channels=1)

    print("\n📌 Summary:")
    for model_type, (loss, acc) in results.items():
        print(f" - {model_type}: Loss={loss:.4f} | Accuracy={acc:.4f}")

    return results


if __name__ == "__main__":
    evaluate_all_models()
