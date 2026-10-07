"""Execution-device selection for training."""

import torch
import os


def gpu_check():
    """Select the best available execution device for training."""
    print("Checking for GPU...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            f"Using CUDA device (visible ID): {torch.cuda.current_device()} / Name: {torch.cuda.get_device_name(0)}"
        )
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    elif torch.backends.mps.is_available():
        # Keep macOS execution on CPU until MPS support is considered stable here.
        device = torch.device(
            "cpu"
        )
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device
