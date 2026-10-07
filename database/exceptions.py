"""Database-layer exceptions."""


class DatabaseError(Exception):
    """Base exception for database layer failures."""


class DatabaseConfigError(DatabaseError):
    """Raised when required DB configuration is missing or invalid."""


class DatabaseConnectionError(DatabaseError):
    """Raised when opening/using DB connection fails."""


class EntityNotFoundError(DatabaseError):
    """Raised when an entity lookup by key returns no row."""
