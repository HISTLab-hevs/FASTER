"""Global RNG seeding shared by the worker (main.py) and the GP engine (utils.gp).

Kept in utils so both the experiment entry point and the GP evolution can apply the
same seed without a circular import.
"""

import random

import numpy as np
import torch


def set_global_seed(seed, deterministic=False):
    """Set seed for all global RNG states to ensure reproducibility.

    Args:
        seed: Integer seed value. If None, no seeding is performed.
        deterministic: If True, enable torch deterministic algorithms (may slow down training).

    Sets:
        - Python random module seed
        - NumPy global seed
        - Torch CPU manual seed
        - Torch CUDA manual seed (if CUDA is available)
    """
    if seed is None:
        return

    # Cast to int to ensure valid seed type
    seed = int(seed)

    # Seed Python's random module
    random.seed(seed)

    # Seed NumPy
    np.random.seed(seed)

    # Seed Torch CPU
    torch.manual_seed(seed)

    # Seed Torch CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optionally enable deterministic behavior in Torch
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
