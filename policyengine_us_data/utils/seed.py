import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible behavior."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
