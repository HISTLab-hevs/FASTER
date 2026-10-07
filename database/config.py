"""Environment-driven settings for MariaDB connectivity."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .exceptions import DatabaseConfigError


@dataclass(frozen=True)
class DatabaseSettings:
    """Connection settings for a MariaDB instance.

    Instances are typically created via :meth:`from_env`, which reads values
    from the process environment and applies the documented defaults when
    variables are not present.
    """

    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "faster"
    password: str = "faster"
    database: str = "faster"
    connect_timeout: int = 8
    pool_size: int = 10

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        """Build database settings from environment variables.

        Supported environment variables:
            - ``FASTER_DB_HOST``
            - ``FASTER_DB_PORT``
            - ``FASTER_DB_USER``
            - ``FASTER_DB_PASSWORD``
            - ``FASTER_DB_NAME``
            - ``FASTER_DB_CONNECT_TIMEOUT``
            - ``FASTER_DB_POOL_SIZE``

        Returns:
            A populated :class:`DatabaseSettings` instance.

        Raises:
            DatabaseConfigError: If an integer-valued setting cannot be parsed
                or if required string settings are empty after normalization.
        """
        try:
            port = int(os.getenv("FASTER_DB_PORT", "3306"))
            connect_timeout = int(os.getenv("FASTER_DB_CONNECT_TIMEOUT", "8"))
            pool_size = int(os.getenv("FASTER_DB_POOL_SIZE", "10"))
        except ValueError as exc:
            raise DatabaseConfigError("Invalid integer in DB env settings") from exc

        host = os.getenv("FASTER_DB_HOST", "127.0.0.1").strip()
        user = os.getenv("FASTER_DB_USER", "faster").strip()
        password = os.getenv("FASTER_DB_PASSWORD", "faster")
        database = os.getenv("FASTER_DB_NAME", "faster").strip()

        if not host or not user or not database:
            raise DatabaseConfigError(
                "FASTER_DB_HOST, FASTER_DB_USER and FASTER_DB_NAME are required"
            )
        if pool_size <= 0:
            raise DatabaseConfigError("FASTER_DB_POOL_SIZE must be greater than zero")

        return cls(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            connect_timeout=connect_timeout,
            pool_size=pool_size,
        )
