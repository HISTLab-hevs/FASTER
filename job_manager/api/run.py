"""Application entrypoint for the FastAPI job manager.

Responsibilities:
    - Initialize database schemas on startup.
    - Start the :class:`~api.job_manager.JobManager` background thread.
    - Register the API router.

Run with::

    uvicorn api.run:app --host 0.0.0.0 --port 5000
    # or:
    python -m api.run
"""

import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.api import router
from api.config import Config
from api.job_db import init_db
from api.job_manager import JobManager
from api.user_db import init_users_db

config = Config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application startup and shutdown lifecycle.

    On startup:
        1. Create/verify database tables.
        2. Launch the :class:`~api.job_manager.JobManager` daemon thread.

    Args:
        app: The FastAPI application instance (unused but required by
            the lifespan protocol).

    Yields:
        Control to the running application.
    """
    init_db()
    init_users_db()

    manager = JobManager(config)
    thread = threading.Thread(target=manager.run, daemon=True)
    thread.start()

    yield


app = FastAPI(
    title="Simple Job Manager",
    version="0.1.0",
    lifespan=lifespan,
)
"""The FastAPI application instance."""

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("api.run:app", host="0.0.0.0", port=5000, reload=config.get("debug", False))
