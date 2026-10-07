"""Shared database connection helper with connection pooling.

Provides a :class:`~dbutils.pooled_db.PooledDB` connection pool and
a :func:`connection` context manager that handles commit, rollback,
and connection return automatically.

All modules that need database access should use::

    from api.db import connection

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ...")

Environment variables ``DB_HOST``, ``DB_PORT``, ``DB_USER``,
``DB_PASSWORD``, and ``DB_NAME`` override the corresponding values
from ``config.yaml → database``.
"""

import os
from contextlib import contextmanager

import pymysql
import pymysql.cursors
from dbutils.pooled_db import PooledDB

from api.config import Config

config = Config()
_db_cfg = config.get("database", {})

_pool = PooledDB(
    creator=pymysql,
    maxconnections=10,
    mincached=2,
    maxcached=5,
    blocking=True,
    host=os.environ.get("DB_HOST", _db_cfg.get("host", "127.0.0.1")),
    port=int(os.environ.get("DB_PORT", _db_cfg.get("port", 3306))),
    user=os.environ.get("DB_USER", _db_cfg.get("user", "root")),
    password=os.environ.get("DB_PASSWORD", _db_cfg.get("password", "root")),
    database=os.environ.get("DB_NAME", _db_cfg.get("database", "job-manager")),
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=False,
)
"""Module-level connection pool shared by all database modules."""


def get_connection():
    """Obtain a raw pooled connection.

    Prefer the :func:`connection` context manager which handles
    commit / rollback automatically.

    Returns:
        A :class:`pymysql.connections.Connection` from the pool.
    """
    return _pool.connection()


@contextmanager
def connection():
    """Context manager that yields a pooled database connection.

    The connection is **committed** when the ``with`` block exits
    normally and **rolled back** on exception.  In both cases the
    connection is returned to the pool.

    Yields:
        A :class:`pymysql.connections.Connection`.

    Example::

        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO ...")
    """
    conn = _pool.connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
