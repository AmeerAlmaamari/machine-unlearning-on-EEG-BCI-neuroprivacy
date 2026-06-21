"""Shared utilities: deterministic seeding and device selection."""
from __future__ import annotations

import os
import random

import numpy as np

from src import config


def set_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility."""
    if seed is None:
        seed = config.SEED
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def get_device():
    """Return 'cuda' if a GPU is available, else 'cpu'. Falls back gracefully."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
