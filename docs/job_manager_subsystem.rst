Job Manager Subsystem
=====================

Job Manager is the execution authority bundled with Faster. It is documented
inside this Faster Sphinx site rather than as a separate parallel
documentation system.

Boundary With Faster
--------------------

Faster owns product-level orchestration:

- user-facing setup validation
- run naming and display-name conflict handling
- scenario, default, and custom-dataset workflows
- run reservation in MariaDB
- artifact export and history presentation

Job Manager owns execution after acceptance:

- authenticated job submission
- queueing and concurrency control
- worker subprocess launch and cancellation
- runtime lifecycle transitions for queued and running jobs
- runtime metrics and hardware sampling
- Job Manager job records in MariaDB

``web_backend/jm_client.py`` is the only Faster-to-Job-Manager HTTP boundary.
The SPA never calls Job Manager directly.

Execution Lifecycle
-------------------

1. The SPA submits a run to the Faster backend through ``POST /api``.
2. Faster validates the config, requires explicit ``server_learning_rate``, and
   reserves a ``runs`` row in MariaDB.
3. Faster writes the required config snapshot under ``results/<run_id>/`` and
   submits a YAML snapshot to Job Manager through ``JobManagerClient``.
4. Job Manager authenticates the service account, stores a queued job row, and
   places the job on its process-local queue.
5. The Job Manager background loop starts worker threads up to
   ``max_concurrent_jobs``.
6. For Faster jobs, Job Manager creates the worker-facing
   ``/app/results/<run_id>/config.yaml`` snapshot and sets execution-only
   ``save_path`` to the run artifact directory.
7. Job Manager launches the configured worker command, normally
   ``python3 -u /app/main.py --config "$JOB_INPUT_PATH" --save_path "$JOB_OUTPUT_DIR"``.
8. Job Manager updates status to ``running``, then to ``completed``,
   ``failed``, or ``stopped`` depending on subprocess outcome or cancellation.

Command-Line Interface
----------------------

``web_backend/job_cli.py`` drives the same submit/list/cancel/attach pipeline as
the SPA from the terminal, run inside the ``faster-app`` container via
``docker compose exec``. It reuses ``JobManagerClient`` and the backend handlers,
so submitted runs are validated, recorded in MariaDB, and queued identically.

Every command takes ``--user``; the role is resolved from the database and the
CLI enforces the **same rules as the dashboard**:

- a normal user may ``list`` the active queue and may ``cancel``/``attach`` to
  their own runs only;
- an admin sees and controls every job.

In ``list``, a normal user's own jobs appear in full, while other users' jobs are
masked to status and queue position only (no ``job_id``, ``run_id``, or owner) so
the user can see how many jobs sit ahead of theirs without learning their
identity. ``cancel``/``attach`` refuse with ``Forbidden`` if a non-admin targets a
job or run they do not own.

Because each CLI invocation is a fresh process, ``JobManagerClient`` supports an
on-disk service-token cache (``FL_JM_TOKEN_CACHE``) so repeated commands share one
login instead of re-authenticating and tripping the Job Manager's per-IP login
rate limit. The long-lived web app leaves this unset and relies on its in-memory
token cache.

Persistence
-----------

MariaDB is the primary store for Job Manager job rows and Faster run metadata. The
filesystem stores only execution material such as uploaded job snapshots,
``config.yaml``, logs, metrics, result files, resource usage samples, uploaded
datasets, and ZIP exports.

Job API Behavior
----------------

The active Job Manager HTTP API is implemented in ``job_manager/api/api.py``:

- ``POST /auth/login`` authenticates a DB-backed bcrypt user and returns a JWT.
- ``POST /submit-job`` accepts a multipart config/file upload and queues a job.
- ``GET /job-status/{job_id}`` returns one authorized job record.
- ``GET /jobs`` lists jobs visible to the authenticated user, with admins
  seeing all jobs.
- ``GET /jobs/user/{user_id}`` lists jobs for one user when the caller is that
  user or an admin.
- ``POST /jobs/{job_id}/cancel`` cancels queued jobs in MariaDB or signals the
  running subprocess.
- ``GET /metrics`` returns job counts, queue depth, active worker count, and
  worker capacity for admins.
- ``GET`` or ``POST /ping`` is an unauthenticated health check.

The subsystem also exposes retry, delete, bulk-delete, and direct result ZIP
endpoints for operator and API-level workflows. Faster's product UI does not
need to surface every generic endpoint for the endpoints to remain documented
as part of the bundled execution subsystem.

Startup and Bootstrap
---------------------

Docker Compose starts Job Manager as ``job-manager-api`` using
``uvicorn api.run:app``. Startup includes:

- loading ``job_manager/config.yaml`` and environment overrides
- registering the FastAPI router
- starting the ``JobManager`` daemon thread
- using the shared MariaDB schema initialized from
  ``database/schema/baseline.sql``

User Management
---------------

Job Manager authentication uses the shared MariaDB ``users`` table and bcrypt
hashes through ``job_manager/api/user_db.py``. Faster administrators normally
bootstrap accounts with ``python3 -m auth.user_cli create-admin`` from the
``faster-app`` container. The Job Manager management module remains documented
for subsystem-level operations.
