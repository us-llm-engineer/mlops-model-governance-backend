"""FastAPI router for audit endpoints (GET /v1/audit/tail SSE, POST /v1/audit/verify)."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from mlops.kernel import ValidationFailed
from mlops.svc.deps import Principal, require_role
from mlops.svc.app import _Unprocessable
from mlops.svc.audit_stream import tail_events
from mlops.svc.routers.incidents import _parse_json_body_for_router

# A stalled client (connected, never reads) would otherwise pin its stream task forever.
_SSE_SEND_TIMEOUT_S = 30.0
_DEFAULT_MAX_TAILS = 8


class _CountedEventSource(EventSourceResponse):
    """EventSourceResponse that holds one slot of a shared, bounded gate while it runs.

    The slot is taken and released inside ``__call__`` (try/finally), so it is returned on
    normal end, client disconnect (task-group cancellation) and send timeouts alike; nothing
    that could skip the release (e.g. a generator ``finally`` that only runs on GC) is involved.
    """

    def __init__(self, *args, gate: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self._gate = gate

    async def __call__(self, scope, receive, send) -> None:
        gate = self._gate
        if gate["active"] >= gate["max"]:
            await JSONResponse(
                status_code=429,
                content={"error": {"code": "too_many_streams",
                                   "message": "too many concurrent audit tails"}},
                headers={"Retry-After": "5"},
            )(scope, receive, send)
            return
        gate["active"] += 1
        try:
            await super().__call__(scope, receive, send)
        finally:
            gate["active"] -= 1


def _valid_head(head) -> bool:
    """A head anchor must be complete: without ``anchor`` ChainVerifier skips the HMAC check,
    and without a head it skips truncation detection -- both would answer ``valid: true`` for a
    truncated export."""
    return (
        isinstance(head, dict)
        and isinstance(head.get("length"), int) and not isinstance(head.get("length"), bool)
        and isinstance(head.get("signature"), str)
        and isinstance(head.get("anchor"), str)
    )


def make_audit_router(audit, chain_verifier_cls, *, max_tails: int = _DEFAULT_MAX_TAILS) -> APIRouter:
    """Create a FastAPI router for audit endpoints.

    Args:
        audit: ChainedAuditStore instance
        chain_verifier_cls: A callable (with no args) that returns a verifier with
                          verify_export(export, head=None) -> bool

    Routes:
        GET /v1/audit/tail: SSE stream of audit entries (operator/admin only)
        POST /v1/audit/verify: verify chain integrity (operator/admin only)
    """
    if isinstance(max_tails, bool) or not isinstance(max_tails, int) or not 1 <= max_tails <= 1000:
        raise ValidationFailed("max_tails must be an int in [1, 1000]")
    router = APIRouter()
    gate = {"active": 0, "max": max_tails}

    @router.get("/v1/audit/tail")
    async def tail(
        principal: Annotated[Principal, Depends(require_role("operator", "admin"))],
    ):
        """Stream audit entries as Server-Sent Events.

        Authorization: operator/admin only (require_role enforces 403 for others).
        Content-Type: text/event-stream
        Each event is {"event": "audit", "data": <JSON string of entry>}
        """

        async def event_generator():
            """Wrap tail_events output: convert data to JSON string for SSE wire format.

            sse-starlette renders dict data with str() (Python repr), which is not valid JSON.
            The router must pass a JSON string instead, so the data: line is parseable JSON.
            """
            async for ev in tail_events(audit):
                # ev is {"event": "audit", "data": <entry dict>}
                # Convert data to JSON string
                yield {
                    "event": ev["event"],
                    "data": json.dumps(ev["data"], sort_keys=True)
                }

        return _CountedEventSource(
            event_generator(), gate=gate, send_timeout=_SSE_SEND_TIMEOUT_S,
        )

    @router.post("/v1/audit/verify")
    async def verify(
        request: Request,
        principal: Annotated[Principal, Depends(require_role("operator", "admin"))],
    ):
        """Verify the integrity of an audit chain export.

        Authorization: operator/admin only (require_role enforces 403 for others).

        Request body: {"export": [<entries>], "head": {<head dict>}}
        - export: required, list of audit entry dicts
        - head: required, dict with length, signature, anchor

        Response:
        - 200 {"valid": true} if export chain is valid
        - 200 {"valid": false, "reason": "<short code>"} if invalid
        - 422 if body is malformed (missing export, wrong types, etc.)

        The reason code is a short lowercase identifier (e.g., "chain_invalid"),
        never a raw exception or traceback.
        """
        # Same body pipeline as every other JSON endpoint (415 content-type, 413 size cap,
        # 422 malformed/too deep/non-finite); runs after the role check above.
        body = await _parse_json_body_for_router(request)

        # Validate body shape (422 for schema errors)
        if "export" not in body:
            raise _Unprocessable("validation error in fields: export")

        export = body.get("export")
        if not isinstance(export, list):
            raise _Unprocessable("validation error in fields: export")

        head = body.get("head")
        if not _valid_head(head):
            raise _Unprocessable("validation error in fields: head")

        # Call the verifier with the exported chain
        try:
            verifier = chain_verifier_cls()
            valid = verifier.verify_export(export, head=head)
        except Exception:
            # Verifier raised an exception (should not happen with ChainVerifier)
            # Treat as invalid, return a short code instead of exception text
            return {"valid": False, "reason": "verification_failed"}

        if valid:
            return {"valid": True}
        else:
            # Chain verification failed (corrupted, tampered, or truncated)
            return {"valid": False, "reason": "chain_invalid"}

    return router
