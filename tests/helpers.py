"""Shared helpers for the test suite."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_MANAGER_ROOT = REPO_ROOT / "job_manager"


def ensure_import_paths() -> None:
    """Ensure repository roots are importable during tests."""
    preferred_order = [str(REPO_ROOT), str(JOB_MANAGER_ROOT)]
    for path in preferred_order:
        if path in sys.path:
            sys.path.remove(path)
    # Keep repository root ahead of bundled subsystem paths so ``import main``
    # resolves to Faster's training entrypoint during tests.
    sys.path[:0] = preferred_order


ensure_import_paths()


def reload_module(module_name: str):
    """Import or reload a module by name."""
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def valid_run_config() -> dict:
    """Return a compact valid run configuration for API tests."""
    return {
        "method": "fed_avg",
        "dataset_name": "pathmnist",
        "num_clients": 2,
        "batch_size": 16,
        "server_warmup_epochs": 2,
        "local_model_epochs": 6,
        "weights_sending_frequency": 3,
        "server_learning_rate": 0.001,
        "learning_rate": 0.001,
        "evaluation_split_mode": "train_val_test",
        "custom_model_mode": "default",
        "custom_model_code": "",
        "run_name": "smoke_run",
    }
