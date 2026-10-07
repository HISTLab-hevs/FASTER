"""Convenience entry point for running the job manager API.

Equivalent to::

    uvicorn api.run:app --host 0.0.0.0 --port 5000
"""

import uvicorn
from api.config import Config

if __name__ == "__main__":
    config = Config()
    uvicorn.run("api.run:app", host="0.0.0.0", port=5000, reload=config.get("debug", False))
