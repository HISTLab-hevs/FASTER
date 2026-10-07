Contributor Map
===============

This page is a quick repository guide for contributors who need to know where
to look by domain before diving into module reference pages.

Frontend
--------

- ``wrapper/``: SPA entrypoint, templates, CSS, and JavaScript modules
- ``wrapper/js/modules/``: domain-oriented frontend behavior

Web backend
-----------

- ``web_backend/app.py``: FastAPI ASGI runtime entrypoint serving the SPA and backend routes
- ``web_backend/service.py``: facade over backend service domains
- ``web_backend/services/``: defaults, content, datasets, run lifecycle, scenarios, run artifacts
  and alerts domain modules behind the service facade
- ``web_backend/api_handlers/``: command dispatch and route-domain behavior
- ``web_backend/jm_client.py``: Faster-side client for the bundled Job Manager

Current runtime/dependency truth
--------------------------------

- ``pyproject.toml`` is the Faster packaging/dependency definition
- ``Dockerfile.web`` is the supported Faster runtime image and currently tracks
  the ``python:3.12-slim`` line

Persistence and auth
--------------------

- ``database/``: schema baseline, repository contracts, row models,
  and MariaDB adapters
- ``auth/``: authentication logic and the Faster-side user CLI
- ``docs/modules/database___init__.rst``: package-level entry page for the
  database layer in Sphinx
- ``docs/modules/database_repositories___init__.rst``: repository-level entry
  page for CRUD implementations and shared helpers

Training engine
---------------

- ``main.py``: training entrypoint executed by Job Manager
- ``utils/``: model, dataset, metrics, training, and evaluation helpers

Bundled Job Manager subsystem
-----------------------------

- ``job_manager/api/``: generic queue, auth, DB, notifier, simulation, and API
  modules
- ``job_manager/config.yaml``: worker command, app registration, and generic
  runtime settings

The repository is easiest to understand if you treat Faster and Job Manager as
two related layers in one codebase: the product-facing application and the
bundled execution subsystem it depends on. The repository documentation for
both layers lives under ``docs/``.
