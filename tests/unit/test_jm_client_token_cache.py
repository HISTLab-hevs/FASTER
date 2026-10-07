"""Unit tests for JobManagerClient's cross-process service-token cache.

The CLI runs as a fresh process per command, so without a shared token cache
every invocation would re-login and quickly trip the Job Manager's per-IP login
rate limit. These tests pin that behavior: one login seeds the cache, and a
second (independent) client reuses it instead of logging in again.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from web_backend.jm_client import JobManagerClient


class JmClientTokenCacheTests(unittest.TestCase):
    """Validate the on-disk service-token cache behavior."""

    def setUp(self) -> None:
        self._env = mock.patch.dict(
            os.environ,
            {
                "JM_API_BASE_URL": "http://job-manager-api:5000",
                "JM_SERVICE_USERNAME": "svc-faster",
                "JM_SERVICE_PASSWORD": "Str0ng!Service123",
            },
            clear=False,
        )
        self._env.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.tmp.name, "token.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()
        self._env.stop()

    def _client_with_cache(self) -> JobManagerClient:
        with mock.patch.dict(os.environ, {"FL_JM_TOKEN_CACHE": self.cache_path}):
            return JobManagerClient()

    def test_successful_login_writes_token_cache(self) -> None:
        """A successful login persists the token (0600) to the cache file."""
        client = self._client_with_cache()
        with mock.patch.object(
            client, "_request_json", return_value=(True, 200, {"token": "tok-1"})
        ) as request_json:
            self.assertTrue(client._service_login())

        request_json.assert_called_once()
        self.assertTrue(os.path.exists(self.cache_path))
        self.assertEqual(0o600, os.stat(self.cache_path).st_mode & 0o777)
        with open(self.cache_path, encoding="utf-8") as fh:
            self.assertEqual("tok-1", json.load(fh)["token"])

    def test_second_client_reuses_cache_without_relogin(self) -> None:
        """A fresh client reuses a valid cached token and does not log in again."""
        first = self._client_with_cache()
        with mock.patch.object(
            first, "_request_json", return_value=(True, 200, {"token": "tok-1"})
        ):
            self.assertTrue(first._service_login())

        second = self._client_with_cache()
        with mock.patch.object(second, "_request_json") as request_json:
            self.assertTrue(second._service_login())
            request_json.assert_not_called()
        self.assertEqual("tok-1", second._token)

    def test_expired_cache_triggers_relogin(self) -> None:
        """A cached token past its expiry is ignored and a fresh login occurs."""
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump({"token": "stale", "expiry": 1.0}, fh)  # far in the past

        client = self._client_with_cache()
        with mock.patch.object(
            client, "_request_json", return_value=(True, 200, {"token": "tok-fresh"})
        ) as request_json:
            self.assertTrue(client._service_login())
            request_json.assert_called_once()
        self.assertEqual("tok-fresh", client._token)

    def test_no_cache_path_means_no_file_and_normal_login(self) -> None:
        """Without FL_JM_TOKEN_CACHE the client logs in and writes no cache file."""
        with mock.patch.dict(os.environ, {"FL_JM_TOKEN_CACHE": ""}):
            client = JobManagerClient()
        self.assertEqual("", client._token_cache_path)
        with mock.patch.object(
            client, "_request_json", return_value=(True, 200, {"token": "tok-x"})
        ):
            self.assertTrue(client._service_login())
        self.assertFalse(os.path.exists(self.cache_path))

    def test_unreadable_cache_falls_back_to_login(self) -> None:
        """A malformed cache file is ignored and login proceeds."""
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            fh.write("{ not json")

        client = self._client_with_cache()
        with mock.patch.object(
            client, "_request_json", return_value=(True, 200, {"token": "tok-ok"})
        ) as request_json:
            self.assertTrue(client._service_login())
            request_json.assert_called_once()


if __name__ == "__main__":
    unittest.main()
