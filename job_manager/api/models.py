"""Pydantic models for API request/response schemas.

Defines the data contracts used by the FastAPI router for
request validation and response serialization.
"""

from pydantic import BaseModel


class LoginRequest(BaseModel):
    """Credentials for authentication.

    Attributes:
        username: The user's login name.
        password: The user's plain-text password.
    """

    username: str
    password: str


class LoginResponse(BaseModel):
    """Successful authentication response.

    Attributes:
        token: Signed JWT token string.
        username: The authenticated user's name.
    """

    token: str
    username: str


class JobSubmitResponse(BaseModel):
    """Response returned after a job is submitted.

    Attributes:
        job_id: Unique identifier assigned to the new job.
        status: Initial status (always ``"queued"``).
        job_type: The type label for the job.
    """

    job_id: str
    status: str
    job_type: str


class JobListResponse(BaseModel):
    """Paginated list of jobs.

    Attributes:
        offset: Zero-based pagination offset.
        limit: Maximum number of records returned.
        count: Actual number of records in this page.
        jobs: List of job record dicts.
    """

    offset: int
    limit: int
    count: int
    jobs: list[dict]


class UserJobListResponse(JobListResponse):
    """Paginated list of jobs filtered by user.

    Attributes:
        user_id: The user whose jobs are listed.
    """

    user_id: str


class BulkDeleteRequest(BaseModel):
    """Request body for bulk job deletion.

    Attributes:
        job_ids: List of job IDs to delete.
    """

    job_ids: list[str]
