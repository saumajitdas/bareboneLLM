import os
import torch

def pick_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
