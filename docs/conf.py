"""Sphinx configuration for the unified Faster documentation tree.

This tree is the main documentation system for the repository. It covers
the product-facing Faster application, the bundled Job Manager execution
subsystem, and their shared MariaDB-backed persistence boundary.
"""

import os
import sys


DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))
JOB_MANAGER_ROOT = os.path.join(PROJECT_ROOT, "job_manager")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, JOB_MANAGER_ROOT)


def _preload_job_manager_runtime() -> None:
    """Import the Job Manager subsystem without a running database.

    The pool stub is scoped to this import only: ``web_backend`` performs real
    ``dbutils`` type checks and must keep seeing the genuine classes.
    """
    import importlib
    from unittest import mock

    from api.config import Config

    config = Config(os.path.join(JOB_MANAGER_ROOT, "config.yaml"))
    # ``api.logger`` attaches a RotatingFileHandler on import and creates the
    # log directory relative to the CWD, which would litter ``docs/`` on every
    # build. Documentation only needs the module imported, not its file sink.
    config._config.pop("log_file", None)
    modules = (
        "api.db",
        "api.auth",
        "api.job_db",
        "api.user_db",
        "api.logger",
        "api.notifier",
        "api.job_manager",
        "api.simulation",
        "api.manage_users",
        "api.api",
        "api.run",
    )
    with mock.patch("dbutils.pooled_db.PooledDB", new=mock.MagicMock()):
        for name in modules:
            importlib.import_module(name)


_preload_job_manager_runtime()

project = "Faster"
author = "HISTLab"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__",
    "show-inheritance": True,
}

autodoc_mock_imports = [
    "bcrypt",
    "cryptography",
    "dbutils",
    "deap",
    "fastapi",
    "fastapi.responses",
    "flask",
    "GPUtil",
    "imageio",
    "joblib",
    "jwt",
    "matplotlib",
    "matplotlib.colors",
    "matplotlib.pyplot",
    "matplotlib.ticker",
    "medmnist",
    "mysql",
    "mysql.connector",
    "numpy",
    "pandas",
    "pydantic",
    "psutil",
    "pymysql",
    "starlette",
    "starlette.background",
    "starlette.requests",
    "starlette.responses",
    "starlette.routing",
    "starlette.staticfiles",
    "requests",
    "scipy",
    "sklearn",
    "sklearn.metrics",
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.utils",
    "torch.utils.data",
    "torchvision",
    "torchvision.datasets",
    "torchvision.transforms",
    "torchmetrics",
    "tqdm",
    "uvicorn",
]

todo_include_todos = False

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
