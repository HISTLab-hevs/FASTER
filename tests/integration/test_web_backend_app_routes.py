"""Integration tests exercising the FastAPI routes over HTTP."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from tests.helpers import reload_module


class WebBackendAppRouteTests(unittest.TestCase):
    """Exercise the FastAPI runtime through HTTP behavior rather than source checks."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app_module = reload_module("web_backend.app")
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.wrapper_index = cls.repo_root / "wrapper" / "index.html"
        cls.wrapper_css = cls.repo_root / "wrapper" / "css" / "styles.css"

    def setUp(self) -> None:
        self.app_module._login_attempts.clear()

    def _runtime_deps(
        self,
        *,
        api_handler=None,
        verify_access_token=None,
        can_user_access_run=None,
        svc=None,
    ):
        api_handler = api_handler or mock.Mock(return_value={"success": True})
        verify_access_token = verify_access_token or mock.Mock(return_value=(True, "alice"))
        can_user_access_run = can_user_access_run or mock.Mock(return_value=True)
        svc = svc or mock.Mock()
        return can_user_access_run, api_handler, verify_access_token, svc

    def test_root_redirects_to_signin(self) -> None:
        """Ensure `/` still redirects to the SPA sign-in route."""
        with TestClient(self.app_module.app) as client:
            response = client.get("/", follow_redirects=False)

        self.assertEqual(307, response.status_code)
        self.assertEqual("/Faster/signin", response.headers["location"])
        self.assertEqual("nosniff", response.headers["x-content-type-options"])
        self.assertEqual("DENY", response.headers["x-frame-options"])

    def test_spa_fallback_serves_index_html(self) -> None:
        """Ensure history-style SPA paths still resolve to the wrapper index."""
        expected = self.wrapper_index.read_text(encoding="utf-8")

        with TestClient(self.app_module.app) as client:
            response = client.get("/Faster/signin")

        self.assertEqual(200, response.status_code)
        self.assertTrue(response.headers["content-type"].startswith("text/html"))
        self.assertEqual(expected, response.text)

    def test_static_wrapper_asset_serves_successfully(self) -> None:
        """Ensure the wrapper static mount continues to serve frontend assets."""
        expected = self.wrapper_css.read_text(encoding="utf-8")

        with TestClient(self.app_module.app) as client:
            response = client.get("/wrapper/css/styles.css")

        self.assertEqual(200, response.status_code)
        self.assertIn("text/css", response.headers["content-type"])
        self.assertEqual(expected, response.text)

    def test_api_route_dispatches_handler(self) -> None:
        """Ensure `POST /api` remains available and forwards JSON to the handler."""
        handler = mock.Mock(return_value={"success": True, "message": "pong"})

        with mock.patch.object(
            self.app_module,
            "_load_runtime_deps",
            return_value=self._runtime_deps(api_handler=handler),
        ):
            with TestClient(self.app_module.app) as client:
                response = client.post("/api", json={"command": "ping", "data": {"x": 1}})

        self.assertEqual(200, response.status_code)
        self.assertEqual({"success": True, "message": "pong"}, response.json())
        handler.assert_called_once()
        self.assertEqual({"command": "ping", "data": {"x": 1}}, json.loads(handler.call_args.args[0]))

    def test_api_route_masks_unhandled_exceptions(self) -> None:
        """Ensure `/api` still masks unexpected failures behind a stable error payload."""
        handler = mock.Mock(side_effect=RuntimeError("boom"))

        with mock.patch.object(
            self.app_module,
            "_load_runtime_deps",
            return_value=self._runtime_deps(api_handler=handler),
        ), mock.patch.object(self.app_module._LOGGER, "exception") as log_exception:
            with TestClient(self.app_module.app) as client:
                response = client.post("/api", json={"command": "ping", "data": {}})

        self.assertEqual(500, response.status_code)
        self.assertEqual({"error": "Internal server error"}, response.json())
        log_exception.assert_called_once()

    def test_api_login_attempts_are_rate_limited(self) -> None:
        """Ensure repeated login attempts still trip the in-memory rate limiter."""
        handler = mock.Mock(return_value={"success": True})

        with mock.patch.object(
            self.app_module,
            "_load_runtime_deps",
            return_value=self._runtime_deps(api_handler=handler),
        ):
            with TestClient(self.app_module.app) as client:
                for attempt in range(10):
                    with self.subTest(attempt=attempt):
                        response = client.post(
                            "/api",
                            json={"command": "login", "data": {"identifier": "alice", "password": "pw"}},
                        )
                        self.assertEqual(200, response.status_code)

                blocked = client.post(
                    "/api",
                    json={"command": "login", "data": {"identifier": "alice", "password": "pw"}},
                )

        self.assertEqual(429, blocked.status_code)
        self.assertEqual(
            {"error": "Too many login attempts, retry later"},
            blocked.json(),
        )
        self.assertEqual(10, handler.call_count)

    def test_upload_requires_authentication_and_streams_multipart_data(self) -> None:
        """Ensure custom dataset uploads reject anonymous requests and stream valid multipart bodies."""
        with mock.patch.object(
            self.app_module,
            "_load_runtime_deps",
            return_value=self._runtime_deps(
                verify_access_token=mock.Mock(return_value=(False, "")),
                svc=mock.Mock(),
            ),
        ):
            with TestClient(self.app_module.app) as client:
                denied = client.post(
                    "/api/upload-custom-dataset",
                    data={"description": "ignored"},
                )

        self.assertEqual(401, denied.status_code)
        self.assertEqual({"error": "Unauthorized"}, denied.json())

        boundary = "boundary123"
        multipart_body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="description"\r\n\r\n'
            "uploaded from test\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="demo.csv"\r\n'
            "Content-Type: text/csv\r\n\r\n"
            "col1,col2\n1,2\n"
            f"\r\n--{boundary}--\r\n"
        ).encode("utf-8")
        captured: dict[str, object] = {}
        uploaded_file_path = ""
        uploaded_temp_dir = ""

        def save_custom_dataset_file(username, original_name, tmp_path, description=""):
            captured["username"] = username
            captured["original_name"] = original_name
            captured["tmp_path"] = tmp_path
            captured["description"] = description
            with open(tmp_path, "rb") as stream:
                captured["contents"] = stream.read()
            return {
                "dataset_ref": "demo.csv",
                "dataset_name": original_name,
                "dataset_type": "csv",
                "dataset_format": "csv",
                "rows": 2,
                "features": 2,
                "classes": 0,
                "description": description,
            }

        verify_access_token = mock.Mock(return_value=(True, "alice"))
        svc = mock.Mock(save_custom_dataset_file=mock.Mock(side_effect=save_custom_dataset_file))

        with mock.patch.object(
            self.app_module,
            "_load_runtime_deps",
            return_value=self._runtime_deps(
                verify_access_token=verify_access_token,
                svc=svc,
            ),
        ):
            with TestClient(self.app_module.app) as client:
                response = client.post(
                    "/api/upload-custom-dataset",
                    headers={"Authorization": "Bearer token-123"},
                    files={"file": ("demo.csv", b"col1,col2\n1,2\n", "text/csv")},
                    data={"description": "uploaded from test"},
                )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "success": True,
                "dataset_ref": "demo.csv",
                "dataset_name": "demo.csv",
                "dataset_type": "csv",
                "dataset_format": "csv",
                "rows": 2,
                "features": 2,
                "classes": 0,
                "description": "uploaded from test",
            },
            response.json(),
        )
        self.assertEqual("token-123", verify_access_token.call_args.args[0])
        self.assertEqual("alice", captured["username"])
        self.assertEqual("demo.csv", captured["original_name"])
        self.assertEqual("uploaded from test", captured["description"])
        self.assertEqual(b"col1,col2\n1,2\n", captured["contents"])
        uploaded_file_path = str(captured["tmp_path"])
        uploaded_temp_dir = str(Path(uploaded_file_path).parent)
        self.assertFalse(Path(uploaded_file_path).exists())
        self.assertFalse(Path(uploaded_temp_dir).exists())

    def test_download_requires_authentication_and_enforces_run_access(self) -> None:
        """Ensure download authorization still fails closed without valid access."""
        with mock.patch.object(
            self.app_module,
            "_load_runtime_deps",
            return_value=self._runtime_deps(
                verify_access_token=mock.Mock(return_value=(False, "")),
                can_user_access_run=mock.Mock(return_value=True),
                svc=mock.Mock(),
            ),
        ):
            with TestClient(self.app_module.app) as client:
                denied = client.get("/fl-download/demo-run.zip")

        self.assertEqual(401, denied.status_code)
        self.assertEqual({"error": "Unauthorized"}, denied.json())

        with mock.patch.object(
            self.app_module,
            "_load_runtime_deps",
            return_value=self._runtime_deps(
                verify_access_token=mock.Mock(return_value=(True, "alice")),
                can_user_access_run=mock.Mock(return_value=False),
                svc=mock.Mock(),
            ),
        ):
            with TestClient(self.app_module.app) as client:
                forbidden = client.get(
                    "/fl-download/demo-run.zip",
                    headers={"Authorization": "Bearer token-123"},
                )

        self.assertEqual(403, forbidden.status_code)
        self.assertEqual({"error": "Forbidden"}, forbidden.json())

    def test_download_streams_zip_file_when_authorized(self) -> None:
        """Ensure authorized download requests still stream the run archive."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(b"zip-bytes")
            tmp_path = tmp.name

        verify_access_token = mock.Mock(return_value=(True, "alice"))
        can_user_access_run = mock.Mock(return_value=True)
        svc = mock.Mock(export_run_to_zip=mock.Mock(return_value=tmp_path))

        try:
            with mock.patch.object(
                self.app_module,
                "_load_runtime_deps",
                return_value=self._runtime_deps(
                    verify_access_token=verify_access_token,
                    can_user_access_run=can_user_access_run,
                    svc=svc,
                ),
            ):
                with TestClient(self.app_module.app) as client:
                    response = client.get(
                        "/fl-download/demo-run.zip",
                        headers={"Authorization": "Bearer token-123"},
                    )

            self.assertEqual(200, response.status_code)
            self.assertEqual(b"zip-bytes", response.content)
            self.assertIn("demo-run.zip", response.headers["content-disposition"])
            self.assertEqual("token-123", verify_access_token.call_args.args[0])
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)
