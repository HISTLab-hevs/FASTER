"""Lightweight row models returned by repository methods."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class AppSettingRow:
    """Repository row model representing a persisted application setting."""

    id: int
    setting_key: str
    setting_json: dict[str, Any]
    updated_by: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class UserRow:
    """Repository row model representing a user record."""

    id: int
    identifier: str
    email: str | None
    password_hash: str
    role: str
    active: bool
    created_at: datetime
    created_by: str


@dataclass(slots=True)
class RunRow:
    """Repository row model representing a persisted run record."""

    id: int
    run_id: str
    owner_identifier: str
    display_name: str
    status: str
    method: str | None
    dataset_name: str | None
    evaluation_split_mode: str | None
    model_name: str | None
    stop_reason: str | None
    best_accuracy: float | None
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class ScenarioRow:
    """Repository row model representing a persisted user scenario."""

    id: int
    owner_identifier: str
    scenario_name: str
    scenario_yaml: str
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class AlertRow:
    """Repository row model representing a user alert."""

    id: int
    owner_identifier: str
    level: str
    message: str
    consumed: bool
    created_at: datetime


@dataclass(slots=True)
class CustomDatasetRow:
    """Repository row model representing uploaded custom dataset metadata."""

    id: int
    owner_identifier: str
    dataset_ref: str
    dataset_name: str
    dataset_type: str
    storage_uri: str
    rows_count: int | None
    features_count: int | None
    classes_count: int | None
    created_at: datetime
