"""Shared backend contracts that must stay aligned across runtime surfaces.

These constants are intentionally small and stable. They capture validation
rules that affect both the SPA and backend behavior so tests and docs can
verify contract coherence without duplicating string literals everywhere.
"""

from __future__ import annotations

SUPPORTED_DATASET_NAMES: tuple[str, ...] = (
    "pathmnist",
    "dermamnist",
    "octmnist",
    "bloodmnist",
    "tissuemnist",
    "pneumoniamnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "fashionmnist",
    "cifar100",
    "cifar10",
    "custom_csv",
    "custom_image_npz",
    "custom_image_folder",
)

CUSTOM_DATASET_NAMES: frozenset[str] = frozenset(
    {"custom_csv", "custom_image_npz", "custom_image_folder"}
)

# The SPA exposes one "custom" option and then resolves the concrete backend
# dataset type from the selected uploaded asset.
FRONTEND_CUSTOM_DATASET_PLACEHOLDER = "custom"


def is_supported_dataset_name(dataset_name: str) -> bool:
    """Return whether one backend ``dataset_name`` value is accepted.

    Args:
        dataset_name: Dataset identifier supplied in a run configuration.

    Returns:
        ``True`` when the dataset name is supported by the backend contract.
    """
    return str(dataset_name or "").strip().lower() in SUPPORTED_DATASET_NAMES


def is_custom_dataset_name(dataset_name: str) -> bool:
    """Return whether one backend ``dataset_name`` uses custom dataset flows.

    Args:
        dataset_name: Dataset identifier supplied in a run configuration.

    Returns:
        ``True`` when the dataset name uses one of the custom-dataset flows.
    """
    return str(dataset_name or "").strip().lower() in CUSTOM_DATASET_NAMES
