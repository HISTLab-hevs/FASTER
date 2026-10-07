Architecture Overview
=====================

Faster is a multi-service repository with one product-facing application and
one bundled execution subsystem:

- ``faster-app`` is the web application that serves the SPA, validates user
  actions, owns product-level run configuration decisions, persists run
  metadata, and exposes the user-facing API surface.
- ``job-manager-api`` is the execution authority after a run is accepted. It
  queues jobs, transitions queued/running terminal lifecycle state, launches
  worker subprocesses, and samples runtime hardware usage.
- ``mariadb`` stores persisted users, runs, jobs, application settings,
  scenarios, alerts, and custom dataset metadata.
- ``reverse-proxy`` terminates HTTPS and routes browser requests to the web app
  and the rest of the Docker-first runtime.

Runtime Topology
----------------

The supported runtime is the four-service Docker Compose stack in
``docker-compose.yml``:

1. Browser traffic enters through ``reverse-proxy``.
2. The SPA under ``wrapper/`` talks only to the Faster web backend.
3. The Faster web backend persists user-facing state and submits background work
   to Job Manager over HTTP.
4. Job Manager executes the configured worker command and owns runtime
   lifecycle transitions after acceptance.
5. MariaDB and shared filesystem volumes store durable records and run-local
   artifacts.

Deployment and Python runtime truth
-----------------------------------

- The supported Python line for Faster is ``3.12.x``.
- ``pyproject.toml`` is the source of truth for Faster's Python
  version and dependency metadata.
- ``Dockerfile.web`` builds the runtime on ``python:3.12-slim`` and installs
  Faster directly from the repository metadata.

Subsystem Boundaries
--------------------

Faster-specific responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- SPA UI under ``wrapper/``
- request validation and auth/session handling
- run reservation and user-visible run metadata
- scenario/default/custom-dataset workflows backed by MariaDB metadata
- artifact export and history presentation
- authentication policy, JWT issuance, and bcrypt password verification

Job Manager responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- authenticated job submission and queue management
- worker-process execution and cancellation
- queued/running/completed/failed/stopped lifecycle transitions after
  acceptance
- queue/job persistence and runtime metrics endpoints in the shared MariaDB
  schema
- worker-facing config snapshot creation for accepted Faster runs

Shared runtime boundary
^^^^^^^^^^^^^^^^^^^^^^^

Faster uses Job Manager as the execution authority for queued and running work.
The only Faster-to-Job-Manager HTTP boundary is ``web_backend/jm_client.py``.
Frontend code and other Faster backend modules do not call Job Manager
directly.

Within Faster itself, ``TrainingServiceLayer`` remains the stable facade used by
API handlers, but it now composes narrower domain services internally:

- run lifecycle
- run artifacts
- alerts
- custom datasets
- content-facing defaults and scenarios

That internal split improves maintainability without changing the public Faster
API surface.

Metadata behavior
---------------------------

The supported runtime uses MariaDB as the primary metadata store and keeps the
filesystem focused on execution artifacts:

- MariaDB owns persisted users, run rows, lifecycle status, stop reasons,
  scenarios, defaults, alerts, jobs, and custom dataset metadata.
- Filesystem volumes remain authoritative only for run-local execution files
  such as config snapshots, logs, metrics, resource samples, uploaded dataset
  files, and exported result bundles.
- Run history, run ownership, display-name conflicts, status, and stop reason
  resolution are DB-backed metadata decisions rather than filesystem discovery.
- Faster reconciles its persisted run metadata against Job Manager state, but
  Job Manager does not replace Faster's own persisted run record contract.
- Authentication is DB-backed and bcrypt-only.
- User scenarios are DB-backed. External YAML uploads can seed a setup form,
  but saved scenarios live in MariaDB.

Current Faster Usage of Job Manager
-----------------------------------

Faster actively uses these Job Manager capabilities today:

- service-account login
- job submission
- job listing and role-scoped active-job snapshots
- job-status lookup
- job cancellation
- metrics retrieval
- worker execution of ``main.py`` via the configured job command
- queue-backed runtime sampling for active Faster runs

Run configuration
---------------------------

Faster owns the product-level run configuration. ``server_learning_rate`` is an
explicit required configuration value and is not inferred from
``learning_rate``. Custom model code must define
``build_model(n_channels, n_classes)`` and return a ``torch.nn.Module``.

The value-level checks (numeric ranges, allowed method, FedGP settings, FedProx
``mu``, custom-model code) live in ``utils/run_config_validation.py`` so both
the web API path and the standalone worker (``main.py``, launched by Job
Manager) reject the same out-of-range settings. The web-specific checks
(dataset support/refs, run display name, round-schedule normalization) stay in
``web_backend/api_handlers/run_workflows.py`` since they need web-only
collaborators.
