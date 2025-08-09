from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

api_key_header_auth = APIKeyHeader(name="Authorization", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header_auth)):
    """Checks for and validates the API key from the Authorization header."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="An API key is required. Please include it in the 'Authorization' header as 'Bearer <key>'."
        )

    # The key from the header will be 'Bearer <key>', so we split it.
    try:
        scheme, _, key = api_key.partition(' ')
        if scheme.lower() != 'bearer' or not key:
            raise ValueError("Invalid token format")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key format. Expected 'Bearer <key>'."
        )

    if key != settings.HACKRX_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return key
