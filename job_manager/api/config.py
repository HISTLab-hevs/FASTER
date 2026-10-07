"""Singleton configuration loader from YAML.

Loads the application configuration from a YAML file on first
instantiation and returns the same instance on subsequent calls.

Example::

    config = Config()
    db_host = config.get("database", {}).get("host")
    app_cfg = config.get_app_config("my-app")
"""

import os
import yaml


class Config:
    """Thread-safe singleton that loads settings from a YAML file.

    The first call to ``Config()`` reads and parses the file; all
    subsequent calls return the cached instance.

    Attributes:
        _instance: The single shared instance (class-level).
        _config: Parsed configuration dictionary.
    """

    _instance = None

    def __new__(cls, config_file: str = "config.yaml") -> "Config":
        """Return the singleton instance, creating it on first call.

        Args:
            config_file: Path to the YAML configuration file.
                Only used during the very first instantiation.

        Returns:
            The shared :class:`Config` instance.

        Raises:
            FileNotFoundError: If the first instantiation cannot find
                *config_file*. The singleton stays unset so a later call with a
                valid path can still succeed.
        """
        if cls._instance is None:
            # Only publish the instance once the file has been read, so a failed
            # load leaves the singleton unset instead of caching a half-built
            # object whose later ``get()`` calls would raise ``AttributeError``.
            instance = super().__new__(cls)
            instance._config = instance._load_config(config_file)
            cls._instance = instance
        return cls._instance

    def _load_config(self, path: str) -> dict:
        """Read and parse a YAML file.

        Args:
            path: Filesystem path to the YAML file.

        Returns:
            Parsed configuration as a dictionary.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default=None):
        """Retrieve a top-level configuration value.

        Args:
            key: Configuration key to look up.
            default: Value returned when *key* is absent.

        Returns:
            The configuration value, or *default*.
        """
        return self._config.get(key, default)

    def get_app_config(self, app_name: str) -> dict | None:
        """Retrieve the configuration block for a specific app.

        Args:
            app_name: Logical application identifier as defined
                under the ``apps`` section in config.yaml.

        Returns:
            The app's config dict, or ``None`` if not found.
        """
        return self._config.get("apps", {}).get(app_name)

    def is_ip_allowed(self, app_name: str, ip: str) -> bool:
        """Check whether an IP address is whitelisted for an app.

        An empty ``allowed_ips`` list means *all* IPs are allowed.

        Args:
            app_name: Logical application identifier.
            ip: Client IP address to check.

        Returns:
            ``True`` if the IP is permitted, ``False`` otherwise.
        """
        app_config = self.get_app_config(app_name)
        if not app_config:
            return False
        allowed_ips = app_config.get("allowed_ips", [])
        return ip in allowed_ips or not allowed_ips

    def get_callback_url(self, app_name: str) -> str | None:
        """Return the callback URL configured for an app.

        Args:
            app_name: Logical application identifier.

        Returns:
            The URL string, or ``None`` if not configured.
        """
        app_config = self.get_app_config(app_name)
        if app_config:
            return app_config.get("callback_url")
        return None

    def get_notify_mode(self, app_name: str, default: str = "callback") -> str:
        """Return the notification mode for an app.

        Supported modes:
            - ``"callback"`` — POST results to a webhook.
            - ``"polling"`` — client polls for results.
            - ``"none"`` — no notifications.

        Args:
            app_name: Logical application identifier.
            default: Fallback mode if not configured.

        Returns:
            The notification mode string.
        """
        app_config = self.get_app_config(app_name)
        if app_config:
            return app_config.get("notify_mode", default)
        return default
