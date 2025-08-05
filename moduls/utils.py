import torch
from pathlib import Path


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:
    """
    Save a PyTorch model's state dictionary to a specified directory.

    This function saves only the `state_dict()` of the model (recommended best practice)
    so it can be easily reloaded later into the same model architecture.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        target_dir (str): Directory where the model should be saved.
        model_name (str): Name of the saved file (must end with ".pth" or ".pt").

    Raises:
        AssertionError: If `model_name` does not end with ".pth" or ".pt".

    Example:
        >>> save_model(model=model_0,
        ...            target_dir="models",
        ...            model_name="cnn_deepfake_model.pth")
    """
    # Ensure target directory exists
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Validate file extension
    assert model_name.endswith(
        (".pth", ".pt")
    ), "❌ model_name should end with '.pt' or '.pth'"

    # Define the full model save path
    model_save_path = target_dir_path / model_name

    # Save only the model's state dictionary (best practice for PyTorch)
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
