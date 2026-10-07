# Security policy

## Supported versions

FASTER is **1.0.x**. Fixes land on `main` and are released from there; there are
no maintained older lines, so a report against anything earlier will be answered
with an upgrade.

## Reporting a vulnerability

Report privately to **elia.pacioni@hevs.ch** or **valerio.crocetti@hevs.ch**.
Please do not open a public issue for a suspected vulnerability.

Useful in a report: what an attacker gains, the affected version or commit, and
the smallest reproduction you have. A configuration description is enough; you
do not need a working exploit.

What to expect:

- an acknowledgement within **10 working days**;
- an assessment, and a fix or a documented decision not to fix;
- credit in the changelog entry if you want it.

There is no bounty, and we do not run a CVE process.

## Scope

**In scope**, the deployed stack described in
[`docker-compose.yml`](docker-compose.yml):

- authentication and session handling (`auth/`): login, bcrypt password storage,
  the `FL_SECRET_KEY`-signed token, the password policy, and the admin/user role
  split enforced in `auth/authenticator.py`;
- the HTTP surface `web_backend/app.py` builds: the API routes under `/api`, the
  artifact export endpoints, and the SPA it serves;
- the `faster-app` to Job Manager bridge (`web_backend/jm_client.py`,
  `job_manager/api/`), including the `JM_JWT_SECRET`-signed service token and the
  ownership rules that stop one user controlling another user's run;
- dataset upload handling: archive extraction, path traversal, and the body-size
  gates (`FL_API_BODY_MAX_BYTES`, `FL_CUSTOM_DATASET_UPLOAD_MAX_BYTES`,
  `NGINX_CLIENT_MAX_BODY_SIZE`);
- SQL injection and authorization bypass in the repository layer (`database/`).

**Out of scope:**

- the placeholder credentials in [`.env.example`](.env.example). They are
  documented placeholders meant to be replaced before the first start, and the
  README says so;
- the self-signed certificate the quickstart generates into `certs/`. It is a
  local development fixture; a real deployment terminates TLS with a real
  certificate. No private key is shipped in this repository;
- findings that require an attacker to already hold valid admin credentials, or
  to already control the host, the Docker daemon, or the `.env` file;
- the Sphinx documentation build and the test suite.

## Known accepted risks

### Custom model code is executed, by design

In **Advanced** setup mode a user supplies Python that defines
`build_model(n_channels, n_classes)`. The worker runs it with `exec()` in the
training subprocess (`main.py`), because there is no other way to let a
researcher bring their own architecture.

`utils/run_config_validation.py` parses the code first and rejects imports of
`os`, `sys`, `subprocess`, `shutil`, `pathlib`, `socket` and `importlib`, and
direct calls to `eval`, `exec`, `compile` and `__import__`.

**This is a guard rail, not a sandbox.** It is an AST blocklist: it raises the
cost of an accident and of casual misuse, and it does not stop a determined
author. Attribute access, indirection through `torch`, and anything reached at
runtime rather than named in the source all go past it. Treat the ability to
submit a run as the ability to run code as the `faster-app` user inside the
container, and grant accounts accordingly. A bypass of the blocklist is
therefore not a vulnerability in itself; a way for an **unauthenticated** caller
to reach that path is.

### Everything behind the proxy trusts the private network

`faster-app`, `job-manager-api` and `mariadb` authenticate callers, but they
expect to sit on the Compose network with only `reverse-proxy` exposed.
Publishing any of their ports to a host interface removes a layer the design
assumes is there.

### Secrets live in `.env`

The stack reads every secret from `.env` on the host. It is excluded from the
repository by `.gitignore` and from the image by `.dockerignore`; keeping it that
way is a deployment responsibility. Rotating `FL_SECRET_KEY` or `JM_JWT_SECRET`
invalidates issued tokens, which is the intended effect.
