"""Tests for environment-driven database settings and the connection factory."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from tests.helpers import reload_module


class MariaDBConnectionFactoryTests(unittest.TestCase):
    """Cover pooled and direct MariaDB connection behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.connection_mod = reload_module("database.connection")
        cls.config_mod = reload_module("database.config")

    def test_connect_uses_pool_when_available(self) -> None:
        """Ensure the connection factory borrows from the pool when it can."""
        settings = self.config_mod.DatabaseSettings(pool_size=7)
        pooled_conn = mock.Mock()
        pooled_conn.is_connected.return_value = True
        pool = mock.Mock()
        pool.get_connection.return_value = pooled_conn
        pooling = SimpleNamespace(MySQLConnectionPool=mock.Mock(return_value=pool))

        with mock.patch.object(
            self.connection_mod,
            "mysql",
            SimpleNamespace(connector=SimpleNamespace(pooling=pooling)),
        ):
            factory = self.connection_mod.MariaDBConnectionFactory(settings)
            conn = factory.connect()

        self.assertIs(conn, pooled_conn)
        pooling.MySQLConnectionPool.assert_called_once_with(
            pool_name=mock.ANY,
            pool_size=7,
            pool_reset_session=True,
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            database=settings.database,
            connection_timeout=settings.connect_timeout,
            autocommit=False,
        )
        pool.get_connection.assert_called_once_with()

    def test_connect_falls_back_to_direct_connection_when_pool_borrow_fails(self) -> None:
        """Ensure the factory falls back to a direct connection when pool borrow fails."""
        settings = self.config_mod.DatabaseSettings()
        direct_conn = mock.Mock()
        direct_conn.is_connected.return_value = True
        pool = mock.Mock()
        pool.get_connection.side_effect = RuntimeError("pool exhausted")
        pooling = SimpleNamespace(MySQLConnectionPool=mock.Mock(return_value=pool))
        connector = SimpleNamespace(
            pooling=pooling,
            connect=mock.Mock(return_value=direct_conn),
        )

        with mock.patch.object(self.connection_mod, "mysql", SimpleNamespace(connector=connector)):
            factory = self.connection_mod.MariaDBConnectionFactory(settings)
            conn = factory.connect()

        self.assertIs(conn, direct_conn)
        connector.connect.assert_called_once_with(
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            database=settings.database,
            connection_timeout=settings.connect_timeout,
            autocommit=False,
        )

    def test_connect_reconnects_stale_borrowed_connection(self) -> None:
        """Ensure stale pooled connections are reconnected before being returned."""
        settings = self.config_mod.DatabaseSettings()
        stale_conn = mock.Mock()
        stale_conn.is_connected.return_value = False
        pool = mock.Mock()
        pool.get_connection.return_value = stale_conn
        pooling = SimpleNamespace(MySQLConnectionPool=mock.Mock(return_value=pool))

        with mock.patch.object(
            self.connection_mod,
            "mysql",
            SimpleNamespace(connector=SimpleNamespace(pooling=pooling)),
        ):
            factory = self.connection_mod.MariaDBConnectionFactory(settings)
            conn = factory.connect()

        self.assertIs(conn, stale_conn)
        stale_conn.reconnect.assert_called_once_with(attempts=1, delay=0)


class DatabaseSettingsTests(unittest.TestCase):
    """Cover environment-driven database settings parsing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config_mod = reload_module("database.config")

    def test_from_env_reads_pool_size(self) -> None:
        """Ensure the pool size is read from the environment configuration."""
        with mock.patch.dict(
            "os.environ",
            {
                "FASTER_DB_HOST": "db",
                "FASTER_DB_USER": "app",
                "FASTER_DB_NAME": "faster",
                "FASTER_DB_POOL_SIZE": "12",
            },
            clear=False,
        ):
            settings = self.config_mod.DatabaseSettings.from_env()

        self.assertEqual(12, settings.pool_size)

    def test_from_env_rejects_non_positive_pool_size(self) -> None:
        """Ensure invalid non-positive pool sizes are rejected during parsing."""
        with mock.patch.dict(
            "os.environ",
            {
                "FASTER_DB_HOST": "db",
                "FASTER_DB_USER": "app",
                "FASTER_DB_NAME": "faster",
                "FASTER_DB_POOL_SIZE": "0",
            },
            clear=False,
        ):
            with self.assertRaises(self.config_mod.DatabaseConfigError):
                self.config_mod.DatabaseSettings.from_env()
