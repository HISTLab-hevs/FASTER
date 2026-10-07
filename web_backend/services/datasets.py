"""Custom dataset service for upload, validation, metadata, and file handling."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import time
import zipfile
import unicodedata
from datetime import datetime
from typing import Any, Callable

import numpy as np
from PIL import Image


logger = logging.getLogger("web_backend.service")


class CustomDatasetsService:
    """Manage custom dataset storage, validation, and metadata persistence."""

    _VISIBLE_FORMATS = {"csv", "tsv", "npz", "zip"}
    _IO_BUFFER_SIZE = 8 * 1024 * 1024
    _MAX_ZIP_PATH_SEGMENT_LEN = 255
    _WINDOWS_RESERVED_SEGMENTS = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *{f"COM{i}" for i in range(1, 10)},
        *{f"LPT{i}" for i in range(1, 10)},
    }
    _IMAGE_EXTENSIONS = {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }

    def __init__(
        self,
        *,
        project_root_getter: Callable[[], str],
        custom_datasets_root: str,
        base_results_path: str,
        dump_config_fn: Callable[[str, dict[str, Any]], None],
        load_run_config_fn: Callable[[str], dict[str, Any]],
        list_runs_fn: Callable[..., list[str]],
        db_enabled_getter: Callable[[], bool],
        custom_datasets_repo_getter: Callable[[], Any],
        runs_repo_getter: Callable[[], Any],
    ) -> None:
        """Initialize the custom-datasets service.

        Args:
            project_root_getter: Callable returning the repository/project root.
            custom_datasets_root: Root directory for persisted custom datasets.
            base_results_path: Base directory containing run artifact folders.
            dump_config_fn: Callable used to persist updated run configuration files.
            load_run_config_fn: Callable used to load persisted run configuration.
            list_runs_fn: Callable listing runs visible to the caller scope.
            db_enabled_getter: Callable returning whether DB-backed storage is enabled.
            custom_datasets_repo_getter: Callable returning the custom-datasets repository.
            runs_repo_getter: Callable returning the runs repository.
        """
        self._project_root = project_root_getter
        self._custom_datasets_root = custom_datasets_root
        self._base_results_path = base_results_path
        self._dump_config = dump_config_fn
        self._load_run_config = load_run_config_fn
        self._list_runs = list_runs_fn
        self._db_enabled = db_enabled_getter
        self._custom_datasets_repo = custom_datasets_repo_getter
        self._runs_repo = runs_repo_getter

    def user_custom_datasets_dir(self, username: str) -> str:
        """Return the per-user custom datasets directory."""
        return os.path.join(self._project_root(), self._custom_datasets_root, username)

    @staticmethod
    def dataset_display_name_from_config(cfg: dict[str, Any]) -> str:
        """Return the user-facing dataset label for run summaries."""
        dataset_name = str(cfg.get("dataset_name") or "")
        if dataset_name in {"custom_csv", "custom_image_npz", "custom_image_folder"}:
            return str(cfg.get("custom_dataset_name") or dataset_name)
        return dataset_name

    @classmethod
    def normalize_dataset_format(
        cls,
        dataset_format: str | None = None,
        *,
        dataset_type: str | None = None,
        dataset_ref: str | None = None,
        storage_uri: str | None = None,
    ) -> str:
        """Return the user-visible dataset format label.

        The backend keeps ``dataset_type`` for execution semantics and exposes
        ``dataset_format`` for the precise file format shown to users.
        """
        normalized = str(dataset_format or "").strip().lower()
        if normalized in cls._VISIBLE_FORMATS:
            return normalized

        for candidate in (dataset_ref, storage_uri):
            ext = os.path.splitext(str(candidate or ""))[1].lower()
            if ext == ".tsv":
                return "tsv"
            if ext == ".npz":
                return "npz"
            if ext == ".csv":
                return "csv"
            if ext == ".zip":
                return "zip"

        dataset_type_text = str(dataset_type or "").strip().lower()
        if dataset_type_text in {"image_npz", "custom_image_npz"}:
            return "npz"
        if dataset_type_text in {"image_folder", "custom_image_folder"}:
            return "zip"
        return "csv"

    def resolve_custom_dataset_path(
        self,
        username: str,
        dataset_ref: str,
        is_admin: bool = False,
    ) -> str | None:
        """Resolve a user-owned custom dataset reference to an absolute path."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{6,180}", dataset_ref or ""):
            return None

        user_dirs: list[str]
        if is_admin:
            data_root = os.path.join(self._project_root(), self._custom_datasets_root)
            if os.path.isdir(data_root):
                user_dirs = [
                    os.path.join(data_root, d)
                    for d in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, d))
                ]
            else:
                user_dirs = []
        else:
            user_dirs = [self.user_custom_datasets_dir(username)]

        for root in user_dirs:
            candidate = os.path.abspath(os.path.join(root, dataset_ref))
            if os.path.isfile(candidate) or os.path.isdir(candidate):
                data_root_abs = os.path.abspath(
                    os.path.join(self._project_root(), self._custom_datasets_root)
                )
                if candidate.startswith(data_root_abs + os.sep):
                    return candidate

        repo = self._custom_datasets_repo()
        if self._db_enabled() and repo is not None:
            try:
                owner_filter = None if is_admin else username
                row = repo.get_dataset(owner_filter, dataset_ref)
                if row:
                    db_path = str(row.storage_uri or "").strip()
                    if db_path:
                        db_abs = os.path.abspath(db_path)
                        if os.path.isfile(db_abs) or os.path.isdir(db_abs):
                            return db_abs
            except Exception as exc:
                logger.warning(
                    "Failed to resolve custom dataset '%s' from DB metadata for user '%s': %s",
                    dataset_ref,
                    username,
                    exc,
                )

        return None

    def list_custom_datasets(
        self,
        username: str,
        is_admin: bool = False,
    ) -> list[dict[str, Any]]:
        """List custom datasets uploaded by the user or all users for admins."""
        out: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()

        def _apply_sidecar(meta_row: dict[str, Any], storage_uri: str) -> None:
            """Overlay sidecar metadata onto one dataset listing row."""
            side = self.load_custom_dataset_sidecar(storage_uri)
            if not side:
                meta_row["dataset_format"] = self.normalize_dataset_format(
                    meta_row.get("dataset_format"),
                    dataset_type=meta_row.get("dataset_type"),
                    dataset_ref=meta_row.get("dataset_ref"),
                    storage_uri=storage_uri,
                )
                return
            side_type = str(side.get("dataset_type") or "").strip().lower()
            storage_format = str(side.get("storage_format") or "").strip().lower()
            if side_type in {"custom_csv", "custom_image_npz", "custom_image_folder"}:
                meta_row["dataset_type"] = side_type
            elif storage_format == "image_folder":
                meta_row["dataset_type"] = "custom_image_folder"
            elif storage_format == "npz":
                meta_row["dataset_type"] = "custom_image_npz"
            elif storage_format == "csv":
                meta_row["dataset_type"] = "custom_csv"
            custom_name = str(side.get("dataset_name") or "").strip()
            if custom_name:
                meta_row["dataset_name"] = custom_name
            meta_row["description"] = str(side.get("description") or "").strip()
            rows = side.get("rows")
            if rows is None:
                split_counts = side.get("split_counts")
                if isinstance(split_counts, dict):
                    rows = sum(
                        int(v) for v in split_counts.values()
                        if isinstance(v, (int, float))
                    )
            if rows is not None:
                meta_row["rows"] = rows
            features = side.get("features")
            if features is None:
                image_shape = side.get("image_shape")
                if isinstance(image_shape, list) and image_shape:
                    try:
                        features = int(np.prod([int(dim) for dim in image_shape]))
                    except Exception:
                        features = None
            if features is not None:
                meta_row["features"] = features
            classes = side.get("classes")
            if classes is None:
                class_names = side.get("class_names")
                if isinstance(class_names, list):
                    classes = len(class_names)
            if classes is not None:
                meta_row["classes"] = classes
            created_at = str(side.get("created_at") or "").strip()
            if created_at:
                meta_row["created_at"] = created_at
            meta_row["dataset_format"] = self.normalize_dataset_format(
                side.get("dataset_format") or side.get("source_format"),
                dataset_type=meta_row.get("dataset_type"),
                dataset_ref=meta_row.get("dataset_ref"),
                storage_uri=storage_uri,
            )

        def _dataset_key(owner: str, dataset_ref: str) -> tuple[str, str]:
            """Return the deduplication key for one owner/dataset pair."""
            return (str(owner or "").strip().lower(), str(dataset_ref or "").strip())

        repo = self._custom_datasets_repo()
        if self._db_enabled() and repo is not None:
            def _row_value(row: Any, key: str, default: Any = None) -> Any:
                """Return one repository field from dict-like or attribute-like rows."""
                if isinstance(row, dict):
                    return row.get(key, default)
                return getattr(row, key, default)

            try:
                owner_filter = None if is_admin else username
                rows = repo.list_datasets(owner_filter)
                if rows:
                    for row in rows:
                        try:
                            owner_raw = _row_value(row, "owner_identifier")
                            dataset_ref_raw = _row_value(row, "dataset_ref")
                            dataset_name_raw = _row_value(row, "dataset_name")
                            dataset_type_raw = _row_value(row, "dataset_type")
                            storage_uri_raw = _row_value(row, "storage_uri")
                            created_at_raw = _row_value(row, "created_at")
                            owner = str(owner_raw or "unknown")
                            dataset_ref = str(dataset_ref_raw or "").strip()
                            storage_uri = str(storage_uri_raw or "")
                            if not dataset_ref or not storage_uri:
                                continue
                            created_at = ""
                            if hasattr(created_at_raw, "isoformat"):
                                try:
                                    created_at = created_at_raw.isoformat(timespec="seconds")
                                except TypeError:
                                    created_at = created_at_raw.isoformat()
                            elif created_at_raw is not None:
                                created_at = str(created_at_raw).strip()
                            item = {
                                "dataset_ref": dataset_ref,
                                "dataset_name": dataset_name_raw,
                                "dataset_type": dataset_type_raw,
                                "dataset_format": self.normalize_dataset_format(
                                    None,
                                    dataset_type=dataset_type_raw,
                                    dataset_ref=dataset_ref,
                                    storage_uri=storage_uri,
                                ),
                                "storage_uri": storage_uri,
                                "owner": owner,
                                "rows": _row_value(row, "rows_count"),
                                "features": _row_value(row, "features_count"),
                                "classes": _row_value(row, "classes_count"),
                                "created_at": created_at,
                                "description": "",
                            }
                            _apply_sidecar(item, storage_uri)
                            out.append(item)
                            seen_keys.add(_dataset_key(owner, dataset_ref))
                        except Exception as exc:
                            logger.warning(
                                "Skipping malformed custom dataset row for user '%s' (is_admin=%s): %s",
                                username,
                                is_admin,
                                exc,
                            )
                            continue
            except Exception as exc:
                logger.warning(
                    "Failed to list custom datasets from DB metadata for user '%s' (is_admin=%s): %s",
                    username,
                    is_admin,
                    exc,
                )

        if is_admin:
            data_root = os.path.join(self._project_root(), self._custom_datasets_root)
            if os.path.isdir(data_root):
                user_dirs = [
                    (d, os.path.join(data_root, d))
                    for d in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, d))
                ]
            else:
                user_dirs = []
        else:
            user_dirs = [(username, self.user_custom_datasets_dir(username))]

        for owner, root in user_dirs:
            if not os.path.isdir(root):
                continue
            for file_name in sorted(os.listdir(root), reverse=True):
                if file_name.endswith(".meta.json"):
                    continue
                path = os.path.join(root, file_name)
                if not (os.path.isfile(path) or os.path.isdir(path)):
                    continue
                key = _dataset_key(owner, file_name)
                if key in seen_keys:
                    continue
                side = self.load_custom_dataset_sidecar(os.path.abspath(path))
                ext = os.path.splitext(file_name)[1].lower()
                dataset_type = str(side.get("dataset_type") or "").strip().lower()
                if dataset_type not in {"custom_csv", "custom_image_npz", "custom_image_folder"}:
                    if os.path.isdir(path):
                        dataset_type = "custom_image_folder"
                    elif ext == ".npz":
                        dataset_type = "custom_image_npz"
                    else:
                        dataset_type = "custom_csv"
                dataset_format = self.normalize_dataset_format(
                    side.get("dataset_format"),
                    dataset_type=dataset_type,
                    dataset_ref=file_name,
                    storage_uri=path,
                )
                out.append(
                    {
                        "dataset_ref": file_name,
                        "dataset_name": os.path.splitext(file_name)[0],
                        "dataset_type": dataset_type,
                        "dataset_format": dataset_format,
                        "storage_uri": os.path.abspath(path),
                        "owner": owner,
                        "rows": None,
                        "features": None,
                        "classes": None,
                        "created_at": datetime.fromtimestamp(
                            os.path.getmtime(path)
                        ).isoformat(timespec="seconds"),
                        "description": "",
                    }
                )
                _apply_sidecar(out[-1], os.path.abspath(path))
                seen_keys.add(key)

        return out

    @staticmethod
    def sanitize_custom_dataset_label(value: str, max_len: int = 20) -> str:
        """Normalize custom dataset display names."""
        txt = str(value or "").strip()
        if not txt:
            raise ValueError("dataset name is required")
        if len(txt) > max_len:
            raise ValueError(f"dataset name must be at most {max_len} characters")
        if not re.fullmatch(r"[A-Za-z0-9_.\- ]+", txt):
            raise ValueError("dataset name contains invalid characters")
        return txt

    @staticmethod
    def sanitize_custom_dataset_description(value: str, max_len: int = 200) -> str:
        """Normalize free-text dataset description with safe printable subset."""
        txt = str(value or "")
        txt = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) > max_len:
            raise ValueError(f"description must be at most {max_len} characters")
        return txt

    @staticmethod
    def custom_dataset_meta_path(abs_dataset_path: str) -> str:
        """Return sidecar metadata path for a dataset file."""
        return f"{abs_dataset_path}.meta.json"

    def load_custom_dataset_sidecar(self, abs_dataset_path: str) -> dict[str, Any]:
        """Load optional sidecar metadata for a custom dataset file."""
        if not abs_dataset_path:
            return {}
        meta_path = self.custom_dataset_meta_path(abs_dataset_path)
        try:
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning(
                "Failed to read dataset sidecar '%s': %s",
                meta_path,
                exc,
            )
            return {}
        return {}

    def save_custom_dataset_sidecar(self, abs_dataset_path: str, payload: dict[str, Any]) -> None:
        """Persist sidecar metadata for a custom dataset file."""
        if not abs_dataset_path:
            return
        meta_path = self.custom_dataset_meta_path(abs_dataset_path)
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(payload or {}, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(
                "Failed to write dataset sidecar '%s': %s",
                meta_path,
                exc,
            )
            return

    def rename_custom_dataset(
        self,
        username: str,
        dataset_ref: str,
        new_name: str,
        is_admin: bool = False,
    ) -> tuple[bool, str]:
        """Rename a custom dataset display label without changing dataset_ref."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{6,180}", dataset_ref or ""):
            return False, "Invalid dataset reference"
        try:
            label = self.sanitize_custom_dataset_label(new_name)
        except ValueError as exc:
            return False, str(exc)

        records = self.list_custom_datasets(username, is_admin=is_admin)
        target = next((d for d in records if str(d.get("dataset_ref") or "") == dataset_ref), None)
        if not target:
            return False, "Dataset not found"

        dataset_owner = str(target.get("owner") or username)
        for row in records:
            if str(row.get("owner") or "") != dataset_owner:
                continue
            ref = str(row.get("dataset_ref") or "")
            if ref == dataset_ref:
                continue
            if str(row.get("dataset_name") or "").strip().lower() == label.lower():
                return False, f"A dataset named '{label}' already exists"

        storage_uri = str(target.get("storage_uri") or "")
        side = self.load_custom_dataset_sidecar(storage_uri)
        side["dataset_name"] = label
        side.setdefault("description", str(target.get("description") or ""))
        side["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.save_custom_dataset_sidecar(storage_uri, side)

        repo = self._custom_datasets_repo()
        if self._db_enabled() and repo is not None:
            try:
                repo.update_dataset(
                    dataset_owner,
                    dataset_ref,
                    {"dataset_name": label},
                )
            except Exception:
                pass

        updated_runs = self._propagate_custom_dataset_rename_to_runs(
            username=dataset_owner,
            dataset_ref=dataset_ref,
            new_name=label,
        )

        owner_suffix = f" (Owner: {dataset_owner})" if is_admin and dataset_owner != username else ""
        return True, f"Dataset renamed{owner_suffix} (updated {updated_runs} run metadata entries)"

    def _propagate_custom_dataset_rename_to_runs(
        self,
        username: str,
        dataset_ref: str,
        new_name: str,
    ) -> int:
        """Propagate custom dataset display-name changes to run metadata."""
        updated = 0
        try:
            run_ids = self._list_runs(username=username, include_all=False)
        except Exception:
            run_ids = []

        runs_repo = self._runs_repo()
        for run_id in run_ids:
            cfg: dict[str, Any] = {}
            if self._db_enabled() and runs_repo is not None:
                try:
                    cfg = runs_repo.get_run_config(run_id) or {}
                except Exception:
                    cfg = {}
            if not cfg:
                try:
                    cfg = self._load_run_config(
                        os.path.join(self._base_results_path, run_id)
                    ) or {}
                except Exception:
                    cfg = {}
            if str(cfg.get("custom_dataset_ref") or "") != dataset_ref:
                continue

            cfg["custom_dataset_name"] = new_name
            run_cfg_path = os.path.join(self._base_results_path, run_id, "config.yaml")
            try:
                if os.path.isfile(run_cfg_path):
                    self._dump_config(run_cfg_path, cfg)
            except Exception:
                pass

            if self._db_enabled() and runs_repo is not None:
                try:
                    runs_repo.update_run(run_id, {"dataset_name": new_name})
                except Exception:
                    pass
                try:
                    runs_repo.upsert_run_config(run_id, cfg)
                except Exception:
                    pass

            updated += 1

        return updated

    def update_custom_dataset_description(
        self,
        username: str,
        dataset_ref: str,
        description: str,
        is_admin: bool = False,
    ) -> tuple[bool, str]:
        """Update free-text description for a custom dataset."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{6,180}", dataset_ref or ""):
            return False, "Invalid dataset reference"
        try:
            desc = self.sanitize_custom_dataset_description(description)
        except ValueError as exc:
            return False, str(exc)

        records = self.list_custom_datasets(username, is_admin=is_admin)
        target = next((d for d in records if str(d.get("dataset_ref") or "") == dataset_ref), None)
        if not target:
            return False, "Dataset not found"

        storage_uri = str(target.get("storage_uri") or "")
        side = self.load_custom_dataset_sidecar(storage_uri)
        side.setdefault("dataset_name", str(target.get("dataset_name") or ""))
        side["description"] = desc
        side["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.save_custom_dataset_sidecar(storage_uri, side)
        return True, "Dataset description updated"

    def delete_custom_dataset(
        self,
        username: str,
        dataset_ref: str,
        is_admin: bool = False,
    ) -> tuple[bool, str]:
        """Delete a custom dataset file and metadata for the requesting user."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{6,180}", dataset_ref or ""):
            return False, "Invalid dataset reference"

        records = self.list_custom_datasets(username, is_admin=is_admin)
        target = next((d for d in records if str(d.get("dataset_ref") or "") == dataset_ref), None)
        if not target:
            return False, "Dataset not found"

        dataset_owner = str(target.get("owner") or username)
        storage_uri = str(target.get("storage_uri") or "")
        removed_any = False

        try:
            if storage_uri and os.path.isfile(storage_uri):
                os.remove(storage_uri)
                removed_any = True
            elif storage_uri and os.path.isdir(storage_uri):
                shutil.rmtree(storage_uri)
                removed_any = True
        except Exception:
            return False, "Failed to delete dataset storage"

        try:
            meta_path = self.custom_dataset_meta_path(storage_uri)
            if os.path.isfile(meta_path):
                os.remove(meta_path)
                removed_any = True
        except Exception:
            pass

        repo = self._custom_datasets_repo()
        if self._db_enabled() and repo is not None:
            try:
                repo.delete_dataset(dataset_owner, dataset_ref)
                removed_any = True
            except Exception:
                pass

        return (True, "Dataset deleted") if removed_any else (False, "Dataset not found")

    def validate_custom_dataset_structure(
        self,
        dataset_path: str,
        dataset_name: str,
    ) -> tuple[bool, str]:
        """Validate a custom dataset before launching training."""
        try:
            if dataset_name == "custom_csv":
                with open(dataset_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if not first_line:
                    return False, "Custom dataset file is empty"
                header = [column.strip().lower() for column in first_line.split(",")]
                if not any(column in {"split", "set", "partition"} for column in header):
                    return False, "Custom tabular dataset must include split column with train/val/test"
                return True, ""

            if dataset_name == "custom_image_npz":
                npz = np.load(dataset_path, allow_pickle=False)
                keys = set(npz.keys())
                split_keys_a = {
                    "train_images",
                    "train_labels",
                    "val_images",
                    "val_labels",
                    "test_images",
                    "test_labels",
                }
                split_keys_b = {
                    "x_train",
                    "y_train",
                    "x_val",
                    "y_val",
                    "x_test",
                    "y_test",
                }
                if not (split_keys_a.issubset(keys) or split_keys_b.issubset(keys)):
                    return False, "Custom image NPZ must provide explicit train/val/test split arrays"
                return True, ""

            if dataset_name == "custom_image_folder":
                self._scan_directory_image_dataset(dataset_path)
                return True, ""

            return True, ""
        except Exception as exc:
            return False, f"Invalid custom dataset structure: {exc}"

    def save_custom_dataset_file(
        self,
        username: str,
        original_name: str,
        source_path: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Validate and persist one uploaded dataset from an existing file path."""
        ext, description_txt, display_name, out_dir = self._prepare_dataset_save(
            username=username,
            original_name=original_name,
            description=description,
        )
        return self._save_custom_dataset_from_path(
            username=username,
            display_name=display_name,
            source_path=source_path,
            description_txt=description_txt,
            ext=ext,
            out_dir=out_dir,
        )

    def _prepare_dataset_save(
        self,
        *,
        username: str,
        original_name: str,
        description: str,
    ) -> tuple[str, str, str, str]:
        """Normalize upload metadata and reject obvious duplicates before writing output."""
        ext = os.path.splitext(original_name or "")[1].lower()
        if ext not in {".csv", ".tsv", ".npz", ".zip"}:
            raise ValueError("Unsupported dataset file type. Allowed: .csv, .tsv, .npz, .zip")

        description_txt = self.sanitize_custom_dataset_description(description)
        safe_name = os.path.basename(original_name or "dataset")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe_name)
        display_name = os.path.splitext(safe_name)[0][:80] or "custom_dataset"

        try:
            existing = self.list_custom_datasets(username)
            for dataset in existing:
                if str(dataset.get("dataset_name") or "") == display_name:
                    raise ValueError(
                        f"A custom dataset with name '{display_name}' already exists "
                        f"in your account. Please select it from the dropdown or "
                        f"rename the file."
                    )
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise

        out_dir = self.user_custom_datasets_dir(username)
        os.makedirs(out_dir, exist_ok=True)
        return ext, description_txt, display_name, out_dir

    def _save_custom_dataset_from_path(
        self,
        *,
        username: str,
        display_name: str,
        source_path: str,
        description_txt: str,
        ext: str,
        out_dir: str,
    ) -> dict[str, Any]:
        """Route one prepared upload file through the format-specific save path."""
        if ext == ".npz":
            return self._save_npz_dataset(
                username=username,
                display_name=display_name,
                source_path=source_path,
                description_txt=description_txt,
                out_dir=out_dir,
            )
        if ext == ".zip":
            return self._save_image_folder_dataset_archive(
                username=username,
                display_name=display_name,
                source_path=source_path,
                description_txt=description_txt,
                out_dir=out_dir,
            )
        return self._save_tabular_dataset(
            username=username,
            display_name=display_name,
            source_path=source_path,
            description_txt=description_txt,
            ext=ext,
            out_dir=out_dir,
        )

    @classmethod
    def _is_supported_image_file(cls, file_name: str) -> bool:
        """Return whether one file path uses a supported image extension."""
        return os.path.splitext(str(file_name or ""))[1].lower() in cls._IMAGE_EXTENSIONS

    @classmethod
    def _scan_directory_image_dataset(cls, dataset_root: str) -> dict[str, Any]:
        """Validate and index one extracted image-folder dataset."""
        root = os.path.abspath(dataset_root)
        if not os.path.isdir(root):
            raise ValueError("Image-folder dataset path is missing or not a directory")

        split_names = ("train", "val", "test")
        split_dirs = {name: os.path.join(root, name) for name in split_names}
        missing_splits = [name for name, path in split_dirs.items() if not os.path.isdir(path)]
        if missing_splits:
            raise ValueError(
                "Image-folder dataset must contain split directories: train/, val/, test/ "
                f"(missing: {', '.join(missing_splits)})"
            )

        split_class_names: dict[str, list[str]] = {}
        for split_name, split_dir in split_dirs.items():
            class_names = sorted(
                name
                for name in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, name))
            )
            if not class_names:
                raise ValueError(
                    f"Split '{split_name}' must contain at least one class subdirectory"
                )
            split_class_names[split_name] = class_names

        expected_class_names = split_class_names["train"]
        if len(expected_class_names) < 2:
            raise ValueError("Image-folder dataset must contain at least 2 classes in train/")

        for split_name, class_names in split_class_names.items():
            if class_names != expected_class_names:
                raise ValueError(
                    "Class subdirectories must match across train/val/test splits. "
                    f"Expected {expected_class_names} but found {class_names} in '{split_name}'."
                )

        class_to_idx = {
            class_name: idx for idx, class_name in enumerate(expected_class_names)
        }
        split_index: dict[str, list[dict[str, Any]]] = {}
        split_counts = {name: 0 for name in split_names}
        image_shape: tuple[int, ...] | None = None
        n_channels: int | None = None
        total_rows = 0

        for split_name, split_dir in split_dirs.items():
            items: list[dict[str, Any]] = []
            for class_name in expected_class_names:
                class_dir = os.path.join(split_dir, class_name)
                files = sorted(
                    file_name
                    for file_name in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, file_name))
                    and cls._is_supported_image_file(file_name)
                )
                if not files:
                    raise ValueError(
                        f"Class '{class_name}' in split '{split_name}' must contain at least one supported image file"
                    )
                for file_name in files:
                    abs_path = os.path.join(class_dir, file_name)
                    rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
                    try:
                        with Image.open(abs_path) as image:
                            width, height = image.size
                            bands = image.getbands() or ()
                            candidate_channels = len(bands) or 1
                            image.verify()
                    except Exception as exc:
                        raise ValueError(
                            f"Failed to read image '{rel_path}': {exc}"
                        ) from exc

                    candidate_shape = (
                        (int(height), int(width))
                        if candidate_channels == 1
                        else (int(height), int(width), int(candidate_channels))
                    )
                    if candidate_channels not in {1, 3, 4}:
                        raise ValueError(
                            f"Image '{rel_path}' must have 1, 3, or 4 channels; found {candidate_channels}"
                        )

                    if image_shape is None:
                        image_shape = candidate_shape
                        n_channels = candidate_channels
                    elif candidate_shape != image_shape:
                        raise ValueError(
                            "All images in an image-folder dataset must share the same shape. "
                            f"Expected {image_shape}, found {candidate_shape} at '{rel_path}'."
                        )

                    items.append({"path": rel_path, "label": class_to_idx[class_name]})
                    split_counts[split_name] += 1
                    total_rows += 1
            split_index[split_name] = items

        if any(split_counts[name] == 0 for name in split_names):
            raise ValueError("Image-folder dataset must contain non-empty train, val, and test splits")

        if image_shape is None or n_channels is None:
            raise ValueError("Image-folder dataset does not contain any readable images")

        features_count = int(np.prod(image_shape))
        return {
            "class_names": expected_class_names,
            "class_to_idx": class_to_idx,
            "split_index": split_index,
            "split_counts": split_counts,
            "rows": total_rows,
            "features": features_count,
            "classes": len(expected_class_names),
            "image_shape": list(image_shape),
            "n_channels": n_channels,
        }

    @classmethod
    def _inspect_image_dataset_archive(
        cls,
        archive: zipfile.ZipFile,
    ) -> dict[str, Any]:
        """Inspect one ZIP image dataset layout before extraction."""
        split_names = ("train", "val", "test")
        required_splits = set(split_names)
        raw_entries: list[tuple[str, list[str]]] = []

        for member in archive.infolist():
            member_name = str(member.filename or "")
            if not member_name:
                continue
            normalized = member_name.replace("\\", "/")
            if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
                raise ValueError("ZIP dataset contains unsafe paths")
            parts = [part for part in normalized.split("/") if part and part != "."]
            if any(cls._is_unsafe_zip_path_segment(part) for part in parts):
                raise ValueError("ZIP dataset contains unsafe paths")
            if parts:
                raw_entries.append((member_name, parts))

        if not raw_entries:
            raise ValueError("ZIP dataset archive is empty")

        root_splits = {parts[0] for _, parts in raw_entries if parts}
        wrapper_dir = None
        if not required_splits.issubset(root_splits):
            wrapper_candidates: dict[str, set[str]] = {}
            for _, parts in raw_entries:
                if len(parts) >= 2:
                    wrapper_candidates.setdefault(parts[0], set()).add(parts[1])
            matches = [name for name, child_splits in wrapper_candidates.items() if required_splits.issubset(child_splits)]
            if len(matches) != 1:
                raise ValueError(
                    "ZIP dataset must contain train/, val/, and test/ split directories at the root "
                    "or under one top-level wrapper directory"
                )
            wrapper_dir = matches[0]

        split_class_names: dict[str, set[str]] = {name: set() for name in split_names}
        supported_members: dict[str, dict[str, list[tuple[str, str]]]] = {
            name: {} for name in split_names
        }
        seen_rel_paths: set[str] = set()
        seen_rel_keys: set[str] = set()

        for member_name, parts in raw_entries:
            rel_parts = parts[1:] if wrapper_dir and parts[0] == wrapper_dir else parts
            if len(rel_parts) < 2:
                continue
            split_name = rel_parts[0]
            class_name = rel_parts[1]
            if split_name not in required_splits:
                continue
            split_class_names[split_name].add(class_name)
            if len(rel_parts) != 3:
                continue
            file_name = rel_parts[2]
            if not cls._is_supported_image_file(file_name):
                continue
            rel_path = "/".join(rel_parts)
            if rel_path in seen_rel_paths:
                raise ValueError(
                    "ZIP dataset contains duplicate normalized paths"
                )
            rel_key = unicodedata.normalize("NFC", rel_path).casefold()
            if rel_key in seen_rel_keys:
                raise ValueError(
                    "ZIP dataset contains duplicate normalized paths"
                )
            seen_rel_paths.add(rel_path)
            seen_rel_keys.add(rel_key)
            supported_members[split_name].setdefault(class_name, []).append((rel_path, member_name))

        missing_splits = [name for name in split_names if not split_class_names[name]]
        if missing_splits:
            raise ValueError(
                "Image-folder dataset must contain split directories: train/, val/, test/ "
                f"(missing: {', '.join(missing_splits)})"
            )

        expected_class_names = sorted(split_class_names["train"])
        if len(expected_class_names) < 2:
            raise ValueError("Image-folder dataset must contain at least 2 classes in train/")

        for split_name in split_names:
            class_names = sorted(split_class_names[split_name])
            if class_names != expected_class_names:
                raise ValueError(
                    "Class subdirectories must match across train/val/test splits. "
                    f"Expected {expected_class_names} but found {class_names} in '{split_name}'."
                )

        ordered_members: list[dict[str, Any]] = []
        for split_name in split_names:
            for class_name in expected_class_names:
                members = sorted(supported_members[split_name].get(class_name, []), key=lambda item: item[0])
                if not members:
                    raise ValueError(
                        f"Class '{class_name}' in split '{split_name}' must contain at least one supported image file"
                    )
                for rel_path, original_member_name in members:
                    ordered_members.append(
                        {
                            "split": split_name,
                            "class_name": class_name,
                            "path": rel_path,
                            "member_name": original_member_name,
                        }
                    )

        return {
            "class_names": expected_class_names,
            "class_to_idx": {
                class_name: idx for idx, class_name in enumerate(expected_class_names)
            },
            "ordered_members": ordered_members,
        }

    @classmethod
    def _is_unsafe_zip_path_segment(cls, segment: str) -> bool:
        """Return whether one ZIP path segment is unsafe across common filesystems."""
        text = str(segment or "")
        if not text or not text.strip():
            return True
        if len(text) > cls._MAX_ZIP_PATH_SEGMENT_LEN:
            return True
        if text.endswith((" ", ".")):
            return True
        normalized = text.rstrip(" .")
        if not normalized:
            return True
        base_name = normalized.split(".", 1)[0].upper()
        return base_name in cls._WINDOWS_RESERVED_SEGMENTS

    @classmethod
    def _read_npy_header_from_zip(
        cls,
        archive: zipfile.ZipFile,
        member_name: str,
    ) -> tuple[tuple[int, ...], np.dtype[Any]]:
        """Read one NPY header from an NPZ member without materializing the full array."""
        with archive.open(member_name, "r") as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(member)
            else:
                shape, _, dtype = np.lib.format.read_array_header_2_0(member)
        return tuple(int(dim) for dim in shape), np.dtype(dtype)

    @classmethod
    def _inspect_npz_dataset(cls, source_path: str) -> dict[str, int]:
        """Validate an NPZ dataset while only loading label arrays fully."""
        split_keys_a = {
            "train_images",
            "train_labels",
            "val_images",
            "val_labels",
            "test_images",
            "test_labels",
        }
        split_keys_b = {"x_train", "y_train", "x_val", "y_val", "x_test", "y_test"}

        with zipfile.ZipFile(source_path, "r") as archive:
            member_map = {
                os.path.splitext(name)[0]: name
                for name in archive.namelist()
                if name.endswith(".npy")
            }
            keys = set(member_map.keys())
            if split_keys_a.issubset(keys):
                image_keys = ("train_images", "val_images", "test_images")
                label_keys = ("train_labels", "val_labels", "test_labels")
            elif split_keys_b.issubset(keys):
                image_keys = ("x_train", "x_val", "x_test")
                label_keys = ("y_train", "y_val", "y_test")
            else:
                raise ValueError("NPZ must provide explicit train/val/test splits")

            image_shapes = {
                key: cls._read_npy_header_from_zip(archive, member_map[key])[0]
                for key in image_keys
            }

        def _shape_ok(shape: tuple[int, ...]) -> bool:
            """Return whether an NPZ image array shape has a supported rank."""
            return len(shape) in {3, 4}

        train_shape = image_shapes[image_keys[0]]
        val_shape = image_shapes[image_keys[1]]
        test_shape = image_shapes[image_keys[2]]
        if not _shape_ok(train_shape):
            raise ValueError("NPZ images must be rank-3 or rank-4 array")
        if not _shape_ok(val_shape) or not _shape_ok(test_shape):
            raise ValueError("Validation/Test image arrays must be rank-3 or rank-4")

        with np.load(source_path, allow_pickle=False) as npz:
            y = np.asarray(npz[label_keys[0]]).reshape(-1)
            y_val = np.asarray(npz[label_keys[1]]).reshape(-1)
            y_test = np.asarray(npz[label_keys[2]]).reshape(-1)

        if y.size == 0 or y_val.size == 0 or y_test.size == 0:
            raise ValueError("NPZ split must include non-empty train, val and test labels")
        if y.size < 20:
            raise ValueError("Image dataset too small: at least 20 samples required")
        if np.unique(np.concatenate([y, y_val, y_test])).size < 2:
            raise ValueError("Image dataset must include at least 2 classes")

        return {
            "rows": int(y.size),
            "features": int(train_shape[-1] if len(train_shape) == 3 else np.prod(train_shape[1:])),
            "classes": int(np.unique(y).size),
        }

    @classmethod
    def _finalize_uploaded_file(cls, source_path: str, out_path: str) -> None:
        """Move one staged upload into final storage, falling back to buffered copy when needed."""
        if os.path.abspath(source_path) == os.path.abspath(out_path):
            return
        try:
            shutil.move(source_path, out_path)
            return
        except Exception:
            pass

        with open(source_path, "rb", buffering=cls._IO_BUFFER_SIZE) as src, open(
            out_path,
            "wb",
            buffering=cls._IO_BUFFER_SIZE,
        ) as dst:
            shutil.copyfileobj(src, dst, length=cls._IO_BUFFER_SIZE)
        os.remove(source_path)

    def _save_image_folder_dataset_archive(
        self,
        *,
        username: str,
        display_name: str,
        source_path: str,
        description_txt: str,
        out_dir: str,
    ) -> dict[str, Any]:
        """Validate a ZIP archive containing an image-folder dataset and persist it."""
        start = time.perf_counter()
        safe_ref = (
            f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{secrets.token_hex(4)}_{display_name}"
        )
        out_path = os.path.join(out_dir, safe_ref)
        stage_dir = tempfile.mkdtemp(prefix=".image_folder_upload_", dir=out_dir)
        dataset_stage_root = os.path.join(stage_dir, "dataset")
        os.makedirs(dataset_stage_root, exist_ok=True)
        try:
            try:
                with zipfile.ZipFile(source_path, "r") as archive:
                    inspect_start = time.perf_counter()
                    archive_manifest = self._inspect_image_dataset_archive(archive)
                    extract_start = time.perf_counter()
                    split_index: dict[str, list[dict[str, Any]]] = {
                        "train": [],
                        "val": [],
                        "test": [],
                    }
                    split_counts = {name: 0 for name in ("train", "val", "test")}
                    total_rows = 0
                    image_shape: tuple[int, ...] | None = None
                    n_channels: int | None = None

                    for item in archive_manifest["ordered_members"]:
                        rel_path = str(item["path"])
                        split_name = str(item["split"])
                        class_name = str(item["class_name"])
                        target_path = os.path.join(dataset_stage_root, *rel_path.split("/"))
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with archive.open(str(item["member_name"]), "r") as src, open(
                            target_path,
                            "wb",
                            buffering=self._IO_BUFFER_SIZE,
                        ) as dst:
                            shutil.copyfileobj(src, dst, length=self._IO_BUFFER_SIZE)

                        try:
                            with Image.open(target_path) as image:
                                width, height = image.size
                                bands = image.getbands() or ()
                                candidate_channels = len(bands) or 1
                                image.verify()
                        except Exception as exc:
                            raise ValueError(
                                f"Failed to read image '{rel_path}': {exc}"
                            ) from exc

                        candidate_shape = (
                            (int(height), int(width))
                            if candidate_channels == 1
                            else (int(height), int(width), int(candidate_channels))
                        )
                        if candidate_channels not in {1, 3, 4}:
                            raise ValueError(
                                f"Image '{rel_path}' must have 1, 3, or 4 channels; found {candidate_channels}"
                            )
                        if image_shape is None:
                            image_shape = candidate_shape
                            n_channels = candidate_channels
                        elif candidate_shape != image_shape:
                            raise ValueError(
                                "All images in an image-folder dataset must share the same shape. "
                                f"Expected {image_shape}, found {candidate_shape} at '{rel_path}'."
                            )

                        split_index[split_name].append(
                            {
                                "path": rel_path,
                                "label": archive_manifest["class_to_idx"][class_name],
                            }
                        )
                        split_counts[split_name] += 1
                        total_rows += 1
            except zipfile.BadZipFile as exc:
                raise ValueError("Invalid ZIP dataset archive") from exc

            if image_shape is None or n_channels is None:
                raise ValueError("Image-folder dataset does not contain any readable images")
            manifest = {
                "class_names": archive_manifest["class_names"],
                "class_to_idx": archive_manifest["class_to_idx"],
                "split_index": split_index,
                "split_counts": split_counts,
                "rows": total_rows,
                "features": int(np.prod(image_shape)),
                "classes": len(archive_manifest["class_names"]),
                "image_shape": list(image_shape),
                "n_channels": n_channels,
            }
            finalize_start = time.perf_counter()
            os.replace(dataset_stage_root, out_path)
            logger.info(
                "ZIP dataset '%s' inspect=%.3fs import_validate=%.3fs finalize=%.3fs",
                safe_ref,
                extract_start - inspect_start,
                finalize_start - extract_start,
                time.perf_counter() - finalize_start,
            )
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

        out = {
            "dataset_ref": safe_ref,
            "dataset_name": display_name,
            "dataset_type": "image_folder",
            "dataset_format": "zip",
            "rows": manifest["rows"],
            "features": manifest["features"],
            "classes": manifest["classes"],
            "description": description_txt,
        }
        self.persist_custom_dataset_metadata(username, out, out_path)
        self.save_custom_dataset_sidecar(
            out_path,
            {
                "dataset_name": display_name,
                "description": description_txt,
                "dataset_type": "custom_image_folder",
                "dataset_format": "zip",
                "storage_format": "image_folder",
                "class_names": manifest["class_names"],
                "class_to_idx": manifest["class_to_idx"],
                "split_counts": manifest["split_counts"],
                "split_index": manifest["split_index"],
                "image_shape": manifest["image_shape"],
                "n_channels": manifest["n_channels"],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        logger.info(
            "Saved custom ZIP image-folder dataset '%s' for user '%s' in %.3fs",
            safe_ref,
            username,
            time.perf_counter() - start,
        )
        return out

    def _save_npz_dataset(
        self,
        *,
        username: str,
        display_name: str,
        source_path: str,
        description_txt: str,
        out_dir: str,
    ) -> dict[str, Any]:
        """Persist one uploaded NPZ dataset and its derived metadata.

        Args:
            username: Identifier of the uploading user.
            display_name: User-visible dataset label.
            source_path: Temporary uploaded-file path to validate and persist.
            description_txt: Optional user-provided dataset description.
            out_dir: Destination directory for the persisted dataset file.

        Returns:
            Metadata payload describing the persisted dataset.
        """
        start = time.perf_counter()
        try:
            manifest = self._inspect_npz_dataset(source_path)
        except ValueError:
            raise
        except Exception:
            raise ValueError("Invalid .npz file")

        safe_ref = (
            f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{secrets.token_hex(4)}_{display_name}.npz"
        )
        out_path = os.path.join(out_dir, safe_ref)
        self._finalize_uploaded_file(source_path, out_path)

        out = {
            "dataset_ref": safe_ref,
            "dataset_name": display_name,
            "dataset_type": "image_npz",
            "dataset_format": "npz",
            "rows": manifest["rows"],
            "features": manifest["features"],
            "classes": manifest["classes"],
            "description": description_txt,
        }
        self.persist_custom_dataset_metadata(username, out, out_path)
        self.save_custom_dataset_sidecar(
            out_path,
            {
                "dataset_name": display_name,
                "description": description_txt,
                "dataset_type": "custom_image_npz",
                "dataset_format": "npz",
                "storage_format": "npz",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        logger.info(
            "Saved custom NPZ dataset '%s' for user '%s' in %.3fs",
            safe_ref,
            username,
            time.perf_counter() - start,
        )
        return out

    def _save_tabular_dataset(
        self,
        *,
        username: str,
        display_name: str,
        source_path: str,
        description_txt: str,
        ext: str,
        out_dir: str,
    ) -> dict[str, Any]:
        """Persist one uploaded CSV/TSV dataset and normalized metadata.

        Args:
            username: Identifier of the uploading user.
            display_name: User-visible dataset label.
            source_path: Temporary uploaded-file path to validate and persist.
            description_txt: Optional user-provided dataset description.
            ext: Source filename extension used to infer the delimiter.
            out_dir: Destination directory for the persisted dataset file.

        Returns:
            Metadata payload describing the persisted dataset.
        """
        start = time.perf_counter()
        source_format = "tsv" if ext == ".tsv" else "csv"
        delimiter = "\t" if ext == ".tsv" else ","
        safe_ref = (
            f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{secrets.token_hex(4)}_{display_name}.csv"
        )
        out_path = os.path.join(out_dir, safe_ref)
        tmp_fd, tmp_out_path = tempfile.mkstemp(prefix=".dataset_upload_", suffix=".csv", dir=out_dir)
        os.close(tmp_fd)

        label_map: dict[str, int] = {}
        split_alias = {
            "train": "train",
            "training": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "dev": "val",
            "test": "test",
            "testing": "test",
        }
        split_counts = {"train": 0, "val": 0, "test": 0}
        row_count = 0
        feat_names: list[str] = []

        try:
            with open(
                source_path,
                "r",
                encoding="utf-8",
                newline="",
                buffering=self._IO_BUFFER_SIZE,
            ) as source_handle:
                reader = csv.reader(source_handle, delimiter=delimiter)
                try:
                    header = next(reader)
                except StopIteration:
                    raise ValueError("Dataset must include header and at least one data row")

                header = [str(cell or "").strip() for cell in header]
                if len(header) < 2:
                    raise ValueError("Dataset must have at least one feature column and one label column")
                if any(not column for column in header):
                    raise ValueError("Header contains empty column names")

                normalized_header = [column.lower() for column in header]
                label_candidates = {"label", "target", "class", "y"}
                label_idx = next(
                    (idx for idx, column in enumerate(normalized_header) if column in label_candidates),
                    len(header) - 1,
                )
                split_idx = next(
                    (idx for idx, column in enumerate(normalized_header) if column in {"split", "set", "partition"}),
                    -1,
                )
                if split_idx < 0:
                    raise ValueError("Dataset must include a split column with train/val/test values")

                feat_indices = [idx for idx in range(len(header)) if idx not in {label_idx, split_idx}]
                if not feat_indices:
                    raise ValueError("No feature columns found after selecting label column")

                feat_names = [header[idx] for idx in feat_indices]
                with open(
                    tmp_out_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                    buffering=self._IO_BUFFER_SIZE,
                ) as output_handle:
                    writer = csv.writer(output_handle)
                    writer.writerow(feat_names + ["label", "split"])

                    for line_no, row in enumerate(reader, start=2):
                        if len(row) != len(header):
                            raise ValueError(f"Row {line_no} has {len(row)} columns, expected {len(header)}")
                        try:
                            features = [float(str(row[idx]).strip()) for idx in feat_indices]
                        except Exception:
                            raise ValueError(f"Row {line_no} contains non-numeric feature values")

                        raw_label = str(row[label_idx]).strip()
                        if not raw_label:
                            raise ValueError(f"Row {line_no} has empty label")
                        raw_split = str(row[split_idx]).strip().lower()
                        split = split_alias.get(raw_split)
                        if split is None:
                            raise ValueError(
                                f"Row {line_no} has invalid split '{row[split_idx]}'. Use train/val/test"
                            )

                        if raw_label not in label_map:
                            label_map[raw_label] = len(label_map)
                        split_counts[split] += 1
                        row_count += 1
                        writer.writerow(features + [label_map[raw_label], split])
        except UnicodeDecodeError:
            raise ValueError("Dataset must be valid UTF-8 text")
        except csv.Error as exc:
            raise ValueError(f"Invalid delimited dataset content: {exc}")
        except ValueError:
            raise
        except Exception:
            raise ValueError("Failed to read dataset file")

        try:
            if row_count < 20:
                raise ValueError("Dataset too small: at least 20 rows are required")
            if len(label_map) < 2:
                raise ValueError("Dataset label column must contain at least 2 classes")
            if any(split_counts[key] == 0 for key in ("train", "val", "test")):
                raise ValueError("Dataset split must include non-empty train, val and test subsets")

            os.replace(tmp_out_path, out_path)
        except Exception:
            try:
                os.remove(tmp_out_path)
            except FileNotFoundError:
                pass
            except Exception:
                logger.warning("Failed to clean normalized dataset temp file '%s'", tmp_out_path, exc_info=True)
            raise

        out = {
            "dataset_ref": safe_ref,
            "dataset_name": display_name,
            "dataset_type": "tabular_csv",
            "dataset_format": source_format,
            "rows": row_count,
            "features": len(feat_names),
            "classes": len(label_map),
            "label_mapping": label_map,
            "description": description_txt,
        }
        self.persist_custom_dataset_metadata(username, out, out_path)
        self.save_custom_dataset_sidecar(
            out_path,
            {
                "dataset_name": display_name,
                "description": description_txt,
                "dataset_type": "custom_csv",
                "dataset_format": source_format,
                "storage_format": "csv",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        logger.info(
            "Saved custom %s dataset '%s' for user '%s' in %.3fs",
            source_format.upper(),
            safe_ref,
            username,
            time.perf_counter() - start,
        )
        return out

    def persist_custom_dataset_metadata(
        self,
        username: str,
        meta: dict[str, Any],
        abs_path: str,
    ) -> None:
        """Persist dataset metadata to DB without failing uploads on DB errors."""
        repo = self._custom_datasets_repo()
        if not (self._db_enabled() and repo is not None):
            return

        api_type = str(meta.get("dataset_type") or "")
        if api_type == "image_npz":
            db_type = "custom_image_npz"
        elif api_type == "image_folder":
            db_type = "custom_image_folder"
        else:
            db_type = "custom_csv"
        try:
            repo.upsert_dataset(
                {
                    "owner_identifier": username,
                    "dataset_ref": str(meta.get("dataset_ref") or ""),
                    "dataset_name": str(meta.get("dataset_name") or "custom_dataset"),
                    "dataset_type": db_type,
                    "storage_uri": os.path.abspath(abs_path),
                    "rows_count": meta.get("rows"),
                    "features_count": meta.get("features"),
                    "classes_count": meta.get("classes"),
                }
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist custom dataset metadata for '%s' owned by '%s': %s",
                str(meta.get("dataset_ref") or ""),
                username,
                exc,
            )
            return
