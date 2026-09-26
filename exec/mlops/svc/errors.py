"""Error handlers for MLOps FastAPI application."""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from mlops.kernel import (
    MlopsError,
    ValidationFailed,
    PolicyDenied,
    NotFound,
    Conflict,
    IntegrityError,
)

logger = logging.getLogger(__name__)


def install_error_handlers(app: FastAPI) -> None:
    """Install exception handlers for the FastAPI app."""

    # Handler for kernel MlopsError exceptions
    async def mlops_error_handler(request, exc: MlopsError):
        status_code_map = {
            "validation_failed": 400,
            "policy_denied": 403,
            "not_found": 404,
            "conflict": 409,
            "integrity_error": 500,
        }

        status_code = status_code_map.get(exc.code, 400)

        # Hide details for IntegrityError
        if isinstance(exc, IntegrityError):
            message = "integrity failure"
        else:
            message = str(exc)

        return JSONResponse(
            status_code=status_code,
            content={"error": {"code": exc.code, "message": message}},
        )

    # Handler for HTTPException
    async def http_exception_handler(request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "http_error", "message": exc.detail}},
            headers=exc.headers if exc.headers else None,
        )

    # Handler for RequestValidationError
    async def request_validation_error_handler(request, exc: RequestValidationError):
        # Extract field names only, never echo values
        field_names = set()
        for error in exc.errors():
            loc = error.get("loc", ())
            # Skip "body" and get the actual field name
            if len(loc) > 1:
                field_names.add(str(loc[1]))
            elif len(loc) == 1:
                field_names.add(str(loc[0]))

        if field_names:
            message = f"validation error in fields: {', '.join(sorted(field_names))}"
        else:
            message = "validation error"

        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation", "message": message}},
        )

    # Handler for generic Exception
    async def generic_exception_handler(request, exc: Exception):
        # Log the exception for debugging but don't expose details
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal", "message": "internal error"}},
        )

    # Handler for StarletteHTTPException (routes that don't exist, etc)
    from starlette.exceptions import HTTPException as StarletteHTTPException
    async def starlette_http_exception_handler(request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "http_error", "message": str(exc.detail)}},
            headers=exc.headers if exc.headers else None,
        )

    app.add_exception_handler(ValidationFailed, mlops_error_handler)
    app.add_exception_handler(PolicyDenied, mlops_error_handler)
    app.add_exception_handler(NotFound, mlops_error_handler)
    app.add_exception_handler(Conflict, mlops_error_handler)
    app.add_exception_handler(IntegrityError, mlops_error_handler)
    app.add_exception_handler(MlopsError, mlops_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
