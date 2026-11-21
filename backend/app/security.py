import hmac
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def require_api_key(provided_key: str | None = Security(api_key_header)):
    configured_key = os.environ.get("SKYGUARD_API_KEY")

    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured",
        )

    if not provided_key or not hmac.compare_digest(
        provided_key,
        configured_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )

    return True
