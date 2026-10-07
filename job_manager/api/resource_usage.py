"""Pure helpers for run-local hardware resource usage files."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any


def safe_float(value: Any) -> float | None:
    """Convert a metric-like value into a finite float.

    Args:
        value: Raw metric value from a hardware sample.

    Returns:
        A finite float when conversion succeeds, otherwise ``None``.
    """
    try:
        numeric_value = float(value)
    except Exception:
        return None
    return numeric_value if math.isfinite(numeric_value) else None


def summarize_resource_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate resource usage values from hardware samples.

    Args:
        samples: Ordered resource usage samples containing CPU, RAM, and
            optional GPU readings.

    Returns:
        A summary dictionary JSON format.
    """
    if not samples:
        return {}

    def _avg(values: list[float]) -> float | None:
        clean = [value for value in values if value is not None]
        if not clean:
            return None
        return float(sum(clean) / len(clean))

    cpu_vals: list[float] = []
    ram_pct_vals: list[float] = []
    ram_used_vals: list[float] = []
    ram_total_vals: list[float] = []
    gpu_load_vals: list[float] = []
    gpu_temp_vals: list[float] = []
    vram_used_vals: list[float] = []
    vram_total_vals: list[float] = []
    vram_usage_pct_vals: list[float] = []

    first_ts = str(samples[0].get("ts") or "")
    last_ts = str(samples[-1].get("ts") or "")

    for sample in samples:
        cpu_value = safe_float(sample.get("cpu_percent"))
        ram_pct_value = safe_float(sample.get("ram_percent"))
        ram_used_value = safe_float(sample.get("ram_used_gb"))
        ram_total_value = safe_float(sample.get("ram_total_gb"))

        if cpu_value is not None:
            cpu_vals.append(cpu_value)
        if ram_pct_value is not None:
            ram_pct_vals.append(ram_pct_value)
        if ram_used_value is not None:
            ram_used_vals.append(ram_used_value)
        if ram_total_value is not None:
            ram_total_vals.append(ram_total_value)

        gpus = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
        if not gpus:
            continue

        per_sample_load: list[float] = []
        per_sample_temp: list[float] = []
        sample_used_gb = 0.0
        sample_total_gb = 0.0
        have_vram = False

        for gpu in gpus:
            if not isinstance(gpu, dict):
                continue
            gpu_load = safe_float(gpu.get("load"))
            gpu_temp = safe_float(gpu.get("temperature"))
            mem_used_mb = safe_float(gpu.get("memory_used_mb"))
            mem_total_mb = safe_float(gpu.get("memory_total_mb"))

            if gpu_load is not None:
                per_sample_load.append(gpu_load)
            if gpu_temp is not None:
                per_sample_temp.append(gpu_temp)
            if mem_used_mb is not None and mem_total_mb is not None and mem_total_mb > 0:
                have_vram = True
                sample_used_gb += mem_used_mb / 1024.0
                sample_total_gb += mem_total_mb / 1024.0

        if per_sample_load:
            gpu_load_vals.append(sum(per_sample_load) / len(per_sample_load))
        if per_sample_temp:
            gpu_temp_vals.append(sum(per_sample_temp) / len(per_sample_temp))
        if have_vram and sample_total_gb > 0:
            vram_used_vals.append(sample_used_gb)
            vram_total_vals.append(sample_total_gb)
            vram_usage_pct_vals.append((sample_used_gb / sample_total_gb) * 100.0)

    duration_s = None
    try:
        t0 = datetime.fromisoformat(first_ts) if first_ts else None
        t1 = datetime.fromisoformat(last_ts) if last_ts else None
        if t0 and t1:
            duration_s = max(0.0, (t1 - t0).total_seconds())
    except Exception:
        duration_s = None

    return {
        "samples_count": len(samples),
        "first_ts": first_ts or None,
        "last_ts": last_ts or None,
        "duration_s": duration_s,
        "cpu_percent_avg": _avg(cpu_vals),
        "ram_percent_avg": _avg(ram_pct_vals),
        "ram_used_gb_avg": _avg(ram_used_vals),
        "ram_total_gb_avg": _avg(ram_total_vals),
        "gpu_load_avg": _avg(gpu_load_vals),
        "gpu_temp_avg": _avg(gpu_temp_vals),
        "vram_used_gb_avg": _avg(vram_used_vals),
        "vram_total_gb_avg": _avg(vram_total_vals),
        "vram_usage_percent_avg": _avg(vram_usage_pct_vals),
    }


def append_resource_sample_file(
    path: str,
    sample: dict[str, Any],
    max_samples: int,
) -> None:
    """Append one hardware sample to a run-local resource usage JSON file.

    Args:
        path: Destination ``resource_usage.json`` path.
        sample: Hardware usage sample to append.
        max_samples: Maximum retained sample count.

    Side Effects:
        Rewrites ``path`` with the appended sample and refreshed summary.
    """
    payload: dict[str, Any] = {"samples": [], "summary": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            payload = raw
    except Exception:
        payload = {"samples": [], "summary": {}}

    samples = payload.get("samples") if isinstance(payload.get("samples"), list) else []
    samples.append(sample)
    if len(samples) > max_samples:
        samples = samples[-max_samples:]

    payload["samples"] = samples
    payload["summary"] = summarize_resource_samples(samples)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass
