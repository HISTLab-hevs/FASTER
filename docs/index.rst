Faster Documentation
====================

This is the main documentation system for the Faster repository.
It covers the product-facing Faster orchestration service, the bundled Job
Manager execution subsystem, the shared MariaDB metadata model, and the
filesystem artifact boundary.

Repository runtime overview
---------------------------

The supported runtime is the Docker Compose stack defined in
``docker-compose.yml``:

- ``reverse-proxy`` terminates HTTPS and routes browser traffic.
- ``faster-app`` serves the SPA, the backend API bridge, and artifact exports.
- ``job-manager-api`` is the bundled generic execution subsystem used by Faster
  for queueing, subprocess execution, and runtime metrics.
- ``mariadb`` persists users, runs, jobs, defaults, scenarios, and dataset metadata.

The web backend is the user-facing orchestration layer. It validates incoming
commands, owns product-level run configuration decisions, persists run
metadata, and delegates accepted execution to Job Manager through
``web_backend/jm_client.py``. Job Manager owns runtime lifecycle transitions
after acceptance, launches ``main.py`` with the submitted config snapshot, and
writes run-local artifacts under the shared results volume.

MariaDB is the primary metadata database for users, bcrypt password hashes,
runs, jobs, saved defaults, scenarios, alerts, and custom dataset metadata. The
filesystem is used for large artifacts, logs, result files, uploaded datasets,
and required execution snapshots only.

.. toctree::
   :maxdepth: 2
   :caption: Architecture and Guides

   architecture_overview
   experiment_setup_guide
   frontend_architecture
   job_manager_subsystem
   contributor_map
   reference_index
