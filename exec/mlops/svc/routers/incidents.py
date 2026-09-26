"""FastAPI router for incident management endpoints."""

import json
import math
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Request

from mlops.incidents import IncidentManager
from mlops.svc.deps import Principal, require_role
from mlops.svc.schemas import IncidentModel
from mlops.svc.app import _Unprocessable, _UnsupportedMedia, _TooLarge


async def _parse_json_body_for_router(request: Request) -> dict:
    """Parse JSON body following app.py pattern (used by router endpoints).

    Checks content-type, size, then parses JSON with validation.
    """
    state = request.app.state.mlops
    content_type = request.headers.get("content-type", "").lower()
    if not content_type.startswith("application/json"):
        raise _UnsupportedMedia()

    max_bytes = state.settings.max_body_bytes
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise _TooLarge()
        except (ValueError, TypeError):
            pass

    chunks = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise _TooLarge()
        chunks.append(chunk)
    body_bytes = b"".join(chunks)

    if not body_bytes:
        return {}

    def reject_constant(x):
        raise ValueError("invalid constant")

    try:
        data = json.loads(body_bytes.decode("utf-8"), parse_constant=reject_constant)
    except RecursionError:
        raise _Unprocessable("Body nesting too deep")
    except (ValueError, UnicodeDecodeError):
        raise _Unprocessable("Body is not valid JSON")

    if not isinstance(data, dict):
        raise _Unprocessable("Body must be a JSON object")

    problem = _json_problem(data)
    if problem == "depth":
        raise _Unprocessable("Body nesting too deep")
    if problem == "nonfinite":
        raise _Unprocessable("Body contains a non-finite number")
    return data


def _json_problem(data, limit: int = 64) -> Optional[str]:
    """Iterative walk: returns 'depth' / 'nonfinite' for the first problem, else None."""
    stack = [(data, 1)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, dict):
            children = node.values()
        elif isinstance(node, list):
            children = node
        elif isinstance(node, float):
            if not math.isfinite(node):
                return "nonfinite"
            continue
        else:
            continue
        if depth > limit:
            return "depth"
        for c in children:
            if isinstance(c, (dict, list, float)):
                stack.append((c, depth + 1))
    return None


def make_incidents_router(manager: IncidentManager) -> APIRouter:
    """Create a FastAPI router for incident management.

    Routes:
    - POST /v1/incidents: open incident (operator/admin)
    - GET /v1/incidents/{incident_id}: get incident (viewer+)
    - POST /v1/incidents/{incident_id}/steps: complete step (operator/admin)
    - POST /v1/incidents/{incident_id}/resolve: resolve incident (operator/admin)
    """
    router = APIRouter()

    @router.post("/v1/incidents", status_code=201)
    async def open_incident(
        request: Request,
        principal: Annotated[Principal, Depends(require_role("operator", "admin"))],
    ):
        """Open a new incident."""
        body = await _parse_json_body_for_router(request)

        # Validate body shape (local validation -> 422)
        if "title" not in body or "signal" not in body:
            raise _Unprocessable("validation error in fields: signal, title")

        title = body.get("title")
        signal = body.get("signal")

        if not isinstance(title, str):
            raise _Unprocessable("validation error in fields: title")

        if not isinstance(signal, dict):
            raise _Unprocessable("validation error in fields: signal")

        # Call domain logic (may raise domain ValidationFailed -> 400)
        incident_id = manager.open(actor=principal.name, title=title, signal=signal)
        record = manager.get(incident_id)

        return {
            "incident_id": incident_id,
            "severity": record["severity"].name,
            "deadline": record["deadline"],
        }

    @router.get("/v1/incidents/{incident_id}")
    async def get_incident(
        incident_id: str,
        principal: Annotated[Principal, Depends(require_role("viewer", "operator", "admin"))],
    ):
        """Get an incident by ID."""
        record = manager.get(incident_id)
        model = IncidentModel.from_record(record)
        return model.model_dump()

    @router.post("/v1/incidents/{incident_id}/steps")
    async def complete_step(
        incident_id: str,
        request: Request,
        principal: Annotated[Principal, Depends(require_role("operator", "admin"))],
    ):
        """Complete a step in an incident."""
        body = await _parse_json_body_for_router(request)

        # Validate body shape (local validation -> 422)
        if "step" not in body or "note" not in body:
            raise _Unprocessable("validation error in fields: note, step")

        step = body.get("step")
        note = body.get("note")

        if not isinstance(step, str):
            raise _Unprocessable("validation error in fields: step")

        if not isinstance(note, str):
            raise _Unprocessable("validation error in fields: note")

        # Call domain logic (may raise domain ValidationFailed -> 400, Conflict -> 409)
        manager.complete_step(actor=principal.name, incident_id=incident_id, step=step, note=note)

        return {"status": "ok"}

    @router.post("/v1/incidents/{incident_id}/resolve")
    async def resolve_incident(
        incident_id: str,
        request: Request,
        principal: Annotated[Principal, Depends(require_role("operator", "admin"))],
    ):
        """Resolve an incident."""
        body = await _parse_json_body_for_router(request)

        # Validate body shape (local validation -> 422)
        if "postmortem" not in body:
            raise _Unprocessable("validation error in fields: postmortem")

        postmortem = body.get("postmortem")

        # postmortem can be None or a dict
        if postmortem is not None and not isinstance(postmortem, dict):
            raise _Unprocessable("validation error in fields: postmortem")

        # Call domain logic (may raise domain ValidationFailed -> 400, Conflict -> 409)
        manager.resolve(actor=principal.name, incident_id=incident_id, postmortem=postmortem)

        return {"status": "ok"}

    return router
