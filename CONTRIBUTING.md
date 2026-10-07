# Contributing

Thanks for looking. The most useful contributions here are small and specific: a
failing configuration, a sharper validation error, a fix with a test. FASTER is
a platform people run experiments on, so anything that changes what a stored run
means, such as the config schema, the metrics written to `metrics.json`, or the
shape of a results folder, breaks existing runs and needs to be discussed first.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.12 (`>=3.12,<3.13`, see
`pyproject.toml`). Always use `uv` against the locked `uv.lock`, never `pip`.

```bash
uv sync
uv run pytest                      # the whole suite, offline, no database
uv run make -C docs html           # the Sphinx reference site
```

The suite needs no database, no running stack and no network: the integration
tests drive in-memory repository fakes. If it wants any of those, something is
wrong with the change, not with your machine. Run a subset by path, for example
`uv run pytest tests/unit`.

Running the platform itself is a separate thing and needs Docker, see the README
quickstart. The stack is the supported runtime; a host-side `uv` environment is
for tests and docs only.

## What runs on a pull request

Nothing automatic yet: there is no CI workflow in this repository. Until there
is, the two commands above are the check, and you are the one running them. Say
so in the pull request, along with what you did not run. "Tests pass, docs not
rebuilt, not tried on a CPU-only host" is more useful than silence.

If you touched the Docker stack, the only honest check is the real one:

```bash
docker compose down -v
docker compose up -d --build
```

then sign in to the dashboard and submit a short run.

## Where things live

[`docs/contributor_map.rst`](docs/contributor_map.rst) is the map by domain and
is kept current; it is the first thing to read. The short version:

| Path | What it owns |
| --- | --- |
| `wrapper/` | The SPA: entrypoint, templates, CSS, JS modules. |
| `web_backend/` | FastAPI app, service facade, API handlers, Job Manager client. |
| `job_manager/` | Queueing, worker lifecycle, cancellation, runtime metrics. |
| `main.py` + `utils/` | The training worker: datasets, models, aggregation methods, FedGP. |
| `database/` + `auth/` | MariaDB adapters, repositories, authentication and the user CLI. |
| `docs/` | The Sphinx site. |
| `tests/` | `unit/` and `integration/`, `unittest` style, run through pytest. |

## Two rules that are not obvious

**Validation lives in `utils/run_config_validation.py`, once.** The web layer and
a direct `python main.py` run both have to reject the same configuration, so
value checks belong in that module and are called from both. A check added only
in the API is a check the CLI path does not have.

**A run must stay reproducible.** Seeding goes through `utils/seed.py`, and each
repeat derives its seed as `base_seed + repeat` so repetitions are independent
and re-runnable. If a change introduces a new source of randomness, it has to be
seeded from there, not from an ambient RNG.

## Changing the configuration schema

A new config key is touched in more places than it looks. Add it to
[`config.yaml`](config.yaml), validate it in `utils/run_config_validation.py`,
surface it in the setup flow in `wrapper/`, and document it in the README or
[`docs/experiment_setup_guide.rst`](docs/experiment_setup_guide.rst), in the
same commit. Then add a test that an invalid value is rejected: nearly every
configuration bug this project has had was a value that validation let through
and the worker crashed on several minutes later.

## Commits

`type: description`, lower case, imperative. `feat:`, `fix:`, `refactor:`,
`test:`, `docs:`, `chore:`.

Explain why in the body when the change is not obvious from the diff.

## Pull requests

Target `main`. Keep one concern per pull request; if you found three things,
three pull requests are easier to review and to revert.

Never commit `.env`, anything under `certs/`, or a database dump. The example
file is [`.env.example`](.env.example) and it is the only one that belongs in
the repository.

## Reporting a vulnerability

Do not open an issue. See [SECURITY.md](SECURITY.md).
