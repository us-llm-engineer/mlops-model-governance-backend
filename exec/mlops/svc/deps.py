"""Dependencies for FastAPI routes including authentication and authorization."""

import hmac
from dataclasses import dataclass
from typing import Annotated, Callable

from fastapi import Depends, HTTPException, Header

from mlops.kernel import PolicyDenied


@dataclass(frozen=True)
class Principal:
    """Represents an authenticated user with name and role."""
    name: str
    role: str


def get_state():
    """Get the application state (to be overridden in tests and real app)."""
    raise NotImplementedError("get_state must be provided by the app")


def get_principal(
    authorization: Annotated[str | None, Header()] = None,
    state=Depends(get_state),
) -> Principal:
    """
    Extract and validate the principal from the Authorization header.

    Uses constant-time comparison for token validation.
    Raises HTTPException(401) if the token is invalid or missing.
    """
    # Parse the Authorization header
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing credentials",
            headers={"www-authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header",
            headers={"www-authenticate": "Bearer"},
        )

    token = parts[1]

    # Validate token format BEFORE comparison: reject if >512 chars or not printable ASCII
    if len(token) > 512 or not token.isascii():
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"www-authenticate": "Bearer"},
        )
    # Check each char is printable ASCII (0x21..0x7e)
    for char in token:
        code = ord(char)
        if code < 0x21 or code > 0x7e:
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
                headers={"www-authenticate": "Bearer"},
            )

    # Validate token using constant-time comparison
    tokens = state.settings.tokens

    # Check against all configured tokens with no early exit, comparing as bytes
    matched_principal = None
    token_bytes = token.encode("ascii")
    for configured_token, token_spec in tokens.items():
        try:
            config_token_bytes = configured_token.encode("ascii")
        except (UnicodeEncodeError, AttributeError):
            # Skip configured tokens that aren't ASCII
            continue
        if hmac.compare_digest(token_bytes, config_token_bytes):
            matched_principal = Principal(name=token_spec.name, role=token_spec.role)

    if matched_principal is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"www-authenticate": "Bearer"},
        )

    return matched_principal


def require_role(*allowed_roles: str) -> Callable[[Principal], Principal]:
    """
    Factory for a dependency that checks if the principal has one of the allowed roles.

    Args:
        *allowed_roles: The role(s) that are allowed to access the endpoint.

    Returns:
        A dependency function that validates the principal's role.

    Raises:
        HTTPException(403) if the principal's role is not in the allowed roles.
    """
    def check_role(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        if principal.role not in allowed_roles:
            raise PolicyDenied(f"Role '{principal.role}' is not authorized")
        return principal

    return check_role
