"""MariaDB connection/session helpers with explicit transaction boundaries."""

from __future__ import annotations

from contextlib import AbstractContextManager
import threading
from typing import Any

from .config import DatabaseSettings
from .exceptions import DatabaseConnectionError

try:
    import mysql.connector  # type: ignore
except Exception:  # pragma: no cover - optional dependency in non-DB environments
    mysql = None


class MariaDBConnectionFactory:
    """Build low-level MariaDB DB-API connections from application settings."""

    def __init__(self, settings: DatabaseSettings) -> None:
        """Initialize the connection factory.

        Args:
            settings: Database settings used to open MariaDB connections.
        """
        self._settings = settings
        self._pool = None
        self._pool_lock = threading.Lock()

    def _connection_kwargs(self) -> dict[str, Any]:
        """Build the shared connector keyword arguments.

        Returns:
            Connector keyword arguments derived from the configured database
            settings.
        """
        return {
            "host": self._settings.host,
            "port": self._settings.port,
            "user": self._settings.user,
            "password": self._settings.password,
            "database": self._settings.database,
            "connection_timeout": self._settings.connect_timeout,
            "autocommit": False,
        }

    def _get_pool(self):
        """Return the lazily initialized connection pool, if available.

        Returns:
            A MySQL connection pool when connector pooling is available,
            otherwise ``None``.
        """
        if self._pool is not None:
            return self._pool

        pooling = getattr(mysql.connector, "pooling", None)
        if pooling is None:
            return None

        with self._pool_lock:
            if self._pool is None:
                self._pool = pooling.MySQLConnectionPool(
                    pool_name=f"faster_{id(self)}",
                    pool_size=self._settings.pool_size,
                    pool_reset_session=True,
                    **self._connection_kwargs(),
                )
        return self._pool

    @staticmethod
    def _ensure_connection_ready(conn: Any) -> Any:
        """Reconnect a borrowed connection when it has gone stale.

        Args:
            conn: Low-level MariaDB connection object.

        Returns:
            A connection object ready for immediate use.

        Raises:
            DatabaseConnectionError: If a stale connection cannot be
                reconnected.
        """
        try:
            is_connected = getattr(conn, "is_connected", None)
            if callable(is_connected) and not is_connected():
                reconnect = getattr(conn, "reconnect", None)
                if callable(reconnect):
                    reconnect(attempts=1, delay=0)
        except Exception as exc:
            raise DatabaseConnectionError(f"Cannot reconnect MariaDB connection: {exc}") from exc
        return conn

    def connect(self):
        """Create and return a new database connection.

        Returns:
            A low-level MariaDB connection object produced by
            ``mysql.connector.connect``.

        Raises:
            DatabaseConnectionError: If the MariaDB connector is unavailable or
                if the connection attempt fails.
        """
        if mysql is None:
            raise DatabaseConnectionError(
                "mysql-connector-python is not installed. Install it before enabling DB mode."
            )

        try:
            pool = self._get_pool()
            if pool is not None:
                try:
                    return self._ensure_connection_ready(pool.get_connection())
                except Exception:
                    # Fall back to a direct connection so transient pool issues
                    # do not change repository behavior under load.
                    pass

            return self._ensure_connection_ready(
                mysql.connector.connect(**self._connection_kwargs())
            )
        except Exception as exc:
            raise DatabaseConnectionError(f"Cannot connect to MariaDB: {exc}") from exc


class MariaDBSession(AbstractContextManager):
    """Context-managed MariaDB session with explicit transaction boundaries.

    Repositories use this wrapper as a lightweight unit-of-work abstraction.
    Successful context exits commit the current transaction; exceptional exits
    roll it back.
    """

    def __init__(self, factory: MariaDBConnectionFactory) -> None:
        """Initialize the session wrapper.

        Args:
            factory: Connection factory used to create the underlying database
                connection.
        """
        self._factory = factory
        self._conn = None

    def __enter__(self) -> "MariaDBSession":
        """Open a connection and enter the session context.

        Returns:
            The active :class:`MariaDBSession` instance.
        """
        self._conn = self._factory.connect()
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb) -> None:
        """Finalize the session and close the underlying connection.

        On normal exit the current transaction is committed. If an exception is
        propagating, the transaction is rolled back instead.

        Args:
            exc_type: Exception type raised inside the context, if any.
            _exc_val: Exception instance raised inside the context, if any.
            _exc_tb: Traceback associated with the exception, if any.

        Returns:
            ``None``.
        """
        if not self._conn:
            return

        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._conn.close()

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
        """Execute a write statement and return the affected row count.

        Args:
            query: SQL statement to execute.
            params: Positional query parameters.

        Returns:
            The number of rows affected by the statement.
        """
        cursor = self._conn.cursor()
        cursor.execute(query, params)
        rowcount = int(cursor.rowcount or 0)
        cursor.close()
        return rowcount

    def execute_returning_id(self, query: str, params: tuple[Any, ...] = ()) -> int:
        """Execute an insert statement and return the generated identifier.

        Args:
            query: SQL statement to execute.
            params: Positional query parameters.

        Returns:
            The last generated primary-key identifier reported by the driver.
        """
        cursor = self._conn.cursor()
        cursor.execute(query, params)
        last_id = int(cursor.lastrowid or 0)
        cursor.close()
        return last_id

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Execute a query and return a single row as a dictionary.

        Args:
            query: SQL statement to execute.
            params: Positional query parameters.

        Returns:
            A dictionary representing the first matching row, or ``None`` when
            no row is returned.
        """
        cursor = self._conn.cursor(dictionary=True)
        cursor.execute(query, params)
        row = cursor.fetchone()
        cursor.close()
        return dict(row) if row else None

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a query and return all rows as dictionaries.

        Args:
            query: SQL statement to execute.
            params: Positional query parameters.

        Returns:
            A list of dictionaries representing all returned rows.
        """
        cursor = self._conn.cursor(dictionary=True)
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        return rows
