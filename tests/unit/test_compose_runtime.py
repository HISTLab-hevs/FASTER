"""Tests asserting the supported Docker Compose runtime contract."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class ComposeRuntimeTests(unittest.TestCase):
    """Verify the supported Docker Compose runtime contract."""

    def test_compose_services_match_supported_stack(self) -> None:
        """Ensure the Compose file exposes only the supported four-service stack."""
        compose_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

        services = set((compose.get("services") or {}).keys())

        self.assertEqual(
            {"reverse-proxy", "faster-app", "job-manager-api", "mariadb"},
            services,
        )
        self.assertNotIn("job-watcher", services)

    def test_job_manager_api_exposes_repo_root_on_python_path(self) -> None:
        """Ensure the Job Manager container can import shared top-level modules."""
        compose_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

        job_manager_api = (compose.get("services") or {}).get("job-manager-api") or {}
        env = job_manager_api.get("environment") or {}

        self.assertEqual("/app", env.get("PYTHONPATH"))

    def test_mariadb_uses_single_schema_baseline(self) -> None:
        """Ensure MariaDB initializes from the single checked-in schema baseline."""
        repo_root = Path(__file__).resolve().parents[2]
        compose_path = repo_root / "docker-compose.yml"
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

        mariadb = (compose.get("services") or {}).get("mariadb") or {}
        volumes = mariadb.get("volumes") or []

        self.assertIn(
            "./database/schema/baseline.sql:/docker-entrypoint-initdb.d/010-faster-schema.sql:ro",
            volumes,
        )
        self.assertNotIn("./database/migrations:/docker-entrypoint-initdb.d/faster:ro", volumes)
        self.assertFalse((repo_root / "database" / "migrations").exists())
