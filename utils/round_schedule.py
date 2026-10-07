"""Helpers for round-level training and per-round client allocation schedules."""

from __future__ import annotations

import math
from typing import Any


ROUND_TRAIN_SCHEDULE_AUTO = "auto"
ROUND_TRAIN_SCHEDULE_MANUAL = "manual"
ROUND_TRAIN_SCHEDULE_NO_SPLIT = "no_split"
ROUND_TRAIN_SCHEDULE_ALLOWED = {
    ROUND_TRAIN_SCHEDULE_AUTO,
    ROUND_TRAIN_SCHEDULE_MANUAL,
    ROUND_TRAIN_SCHEDULE_NO_SPLIT,
}
ROUND_TRAIN_SCHEDULE_SUM_TARGET = 100.0
ROUND_TRAIN_SCHEDULE_SUM_EPSILON = 1e-6

ROUND_CLIENT_ALLOCATION_AUTO = "auto"
ROUND_CLIENT_ALLOCATION_MANUAL = "manual"
ROUND_CLIENT_ALLOCATION_ALLOWED = {
    ROUND_CLIENT_ALLOCATION_AUTO,
    ROUND_CLIENT_ALLOCATION_MANUAL,
}


def aggregation_rounds_from_config(config: dict[str, Any]) -> int:
    """Return the number of aggregation rounds derived from one config."""
    local_epochs = int(config.get("local_model_epochs", 0))
    frequency = int(config.get("weights_sending_frequency", 0))
    if local_epochs <= 0 or frequency <= 0:
        raise ValueError("local_model_epochs and weights_sending_frequency must be positive")
    if local_epochs % frequency != 0:
        raise ValueError("local_model_epochs must be divisible by weights_sending_frequency")
    return local_epochs // frequency


def even_round_percentages(aggregation_rounds: int) -> list[float]:
    """Return one even 100%-coverage schedule for the requested round count."""
    rounds = int(aggregation_rounds)
    if rounds <= 0:
        raise ValueError("aggregation_rounds must be >= 1")
    if rounds == 1:
        return [ROUND_TRAIN_SCHEDULE_SUM_TARGET]

    base = round(ROUND_TRAIN_SCHEDULE_SUM_TARGET / rounds, 6)
    percentages = [base for _ in range(rounds)]
    percentages[-1] = round(
        ROUND_TRAIN_SCHEDULE_SUM_TARGET - sum(percentages[:-1]),
        6,
    )
    return percentages


def even_client_percentages(num_clients: int) -> list[float]:
    """Return one even client-allocation schedule for a round."""
    clients = int(num_clients)
    if clients <= 0:
        raise ValueError("num_clients must be >= 1")
    if clients == 1:
        return [ROUND_TRAIN_SCHEDULE_SUM_TARGET]
    return even_round_percentages(clients)


def normalize_percentage_row(
    raw_values: Any,
    *,
    field_name: str,
) -> list[float]:
    """Parse one percentage row into finite floats."""
    if raw_values in (None, "", []):
        return []
    if isinstance(raw_values, str):
        parts = [item.strip() for item in raw_values.split(",") if item.strip()]
    elif isinstance(raw_values, (list, tuple)):
        parts = list(raw_values)
    else:
        raise ValueError(f"{field_name} must be a list or comma-separated string")

    out: list[float] = []
    for item in parts:
        try:
            value = float(item)
        except Exception as exc:
            raise ValueError(f"{field_name} must contain valid numbers") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain finite numbers")
        out.append(float(value))
    return out


def parse_round_schedule_percentages(raw_value: Any) -> list[float]:
    """Parse one round-coverage payload into a list of percentage floats."""
    return normalize_percentage_row(
        raw_value,
        field_name="round_train_schedule_percentages",
    )


def parse_round_client_allocation_percentages(raw_value: Any) -> list[list[float]]:
    """Parse one per-round client allocation payload into numeric rows."""
    if raw_value in (None, "", []):
        return []
    if not isinstance(raw_value, (list, tuple)):
        raise ValueError(
            "round_client_allocation_percentages must be a list of per-round client percentage lists"
        )

    rows: list[list[float]] = []
    for idx, row in enumerate(raw_value):
        rows.append(
            normalize_percentage_row(
                row,
                field_name=f"round_client_allocation_percentages[{idx + 1}]",
            )
        )
    return rows


def _validate_percentage_sum(percentages: list[float], *, field_name: str) -> None:
    """Validate that one percentage list sums to 100."""
    total = sum(percentages)
    if abs(total - ROUND_TRAIN_SCHEDULE_SUM_TARGET) > ROUND_TRAIN_SCHEDULE_SUM_EPSILON:
        raise ValueError(f"{field_name} must sum to 100")


def normalize_round_train_schedule_config(
    config: dict[str, Any],
    *,
    aggregation_rounds: int | None = None,
) -> list[float]:
    """Normalize one run config's round-level train schedule in place."""
    rounds = (
        int(aggregation_rounds)
        if aggregation_rounds is not None
        else aggregation_rounds_from_config(config)
    )
    if rounds <= 0:
        raise ValueError("aggregation_rounds must be >= 1")

    mode = str(
        config.get("round_train_schedule_mode")
        or ROUND_TRAIN_SCHEDULE_AUTO
    ).strip().lower()
    if mode not in ROUND_TRAIN_SCHEDULE_ALLOWED:
        raise ValueError("round_train_schedule_mode must be 'auto', 'manual', or 'no_split'")

    raw_percentages = config.get("round_train_schedule_percentages", [])
    parsed_percentages = parse_round_schedule_percentages(raw_percentages)

    if mode == ROUND_TRAIN_SCHEDULE_AUTO:
        normalized_percentages: list[float] = []
        resolved_percentages = even_round_percentages(rounds)
    elif mode == ROUND_TRAIN_SCHEDULE_NO_SPLIT:
        # No split: every round reuses the full federated train pool, so each
        # round is resolved to 100% coverage instead of a disjoint slice.
        normalized_percentages = []
        resolved_percentages = [ROUND_TRAIN_SCHEDULE_SUM_TARGET for _ in range(rounds)]
    else:
        if len(parsed_percentages) != rounds:
            raise ValueError(
                "round_train_schedule_percentages count must match aggregation rounds"
            )
        if any(value <= 0 for value in parsed_percentages):
            raise ValueError(
                "round_train_schedule_percentages must contain only positive values"
            )
        _validate_percentage_sum(
            parsed_percentages,
            field_name="round_train_schedule_percentages",
        )
        normalized_percentages = [round(float(value), 6) for value in parsed_percentages]
        resolved_percentages = list(normalized_percentages)

    config["round_train_schedule_mode"] = mode
    config["round_train_schedule_percentages"] = normalized_percentages
    config["resolved_round_train_schedule_percentages"] = resolved_percentages
    config["round_train_schedule_rounds"] = rounds
    config["round_train_schedule_total_percentage"] = round(sum(resolved_percentages), 6)
    return resolved_percentages


def normalize_round_client_allocation_config(
    config: dict[str, Any],
    *,
    aggregation_rounds: int | None = None,
    num_clients: int | None = None,
    auto_client_percentages: list[float] | None = None,
) -> list[list[float]]:
    """Normalize one run config's per-round client allocation schedule in place."""
    rounds = (
        int(aggregation_rounds)
        if aggregation_rounds is not None
        else aggregation_rounds_from_config(config)
    )
    clients = int(num_clients if num_clients is not None else config.get("num_clients", 0))
    if rounds <= 0:
        raise ValueError("aggregation_rounds must be >= 1")
    if clients <= 0:
        raise ValueError("num_clients must be >= 1")

    mode = str(
        config.get("round_client_allocation_mode")
        or ROUND_CLIENT_ALLOCATION_AUTO
    ).strip().lower()
    if mode not in ROUND_CLIENT_ALLOCATION_ALLOWED:
        raise ValueError("round_client_allocation_mode must be 'auto' or 'manual'")

    raw_allocations = config.get("round_client_allocation_percentages", [])
    parsed_allocations = parse_round_client_allocation_percentages(raw_allocations)

    if mode == ROUND_CLIENT_ALLOCATION_AUTO:
        normalized_allocations: list[list[float]] = []
        base_row = (
            [round(float(value), 6) for value in auto_client_percentages]
            if auto_client_percentages
            else even_client_percentages(clients)
        )
        if len(base_row) != clients:
            raise ValueError(
                "resolved automatic round_client_allocation_percentages must match num_clients"
            )
        if any(value < 0 for value in base_row):
            raise ValueError(
                "resolved automatic round_client_allocation_percentages must be non-negative"
            )
        _validate_percentage_sum(
            base_row,
            field_name="resolved automatic round_client_allocation_percentages",
        )
        resolved_allocations = [list(base_row) for _ in range(rounds)]
    else:
        if len(parsed_allocations) != rounds:
            raise ValueError(
                "round_client_allocation_percentages count must match aggregation rounds"
            )
        resolved_allocations = []
        for round_idx, row in enumerate(parsed_allocations):
            if len(row) != clients:
                raise ValueError(
                    f"round_client_allocation_percentages[{round_idx + 1}] count must match num_clients"
                )
            if any(value < 0 for value in row):
                raise ValueError(
                    f"round_client_allocation_percentages[{round_idx + 1}] must contain only non-negative values"
                )
            _validate_percentage_sum(
                row,
                field_name=f"round_client_allocation_percentages[{round_idx + 1}]",
            )
            resolved_allocations.append([round(float(value), 6) for value in row])
        normalized_allocations = [list(row) for row in resolved_allocations]

    config["round_client_allocation_mode"] = mode
    config["round_client_allocation_percentages"] = normalized_allocations
    config["resolved_round_client_allocation_percentages"] = resolved_allocations
    config["round_client_allocation_rounds"] = rounds
    config["round_client_allocation_num_clients"] = clients
    config["round_client_allocation_total_percentage"] = ROUND_TRAIN_SCHEDULE_SUM_TARGET
    return resolved_allocations

