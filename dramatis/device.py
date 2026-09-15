"""Device selection: CUDA if available, else Apple MPS, else CPU."""

import torch


def pick_device(name: str = "auto") -> torch.device:
    """Resolve a device name. "auto" picks cuda > mps > cpu; anything else is passed through."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
