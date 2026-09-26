"""FastAPI router for policy history and decision log endpoints."""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, Request

from mlops.policy_store import PolicyStore
from mlops.svc.deps import Principal, require_role
from mlops.svc.schemas import PolicyBundleModel
from mlops.svc.app import _Unprocessable


def make_policy_history_router(store: PolicyStore = None) -> APIRouter:
    """Create a FastAPI router for policy history and decisions.

    Routes:
    - GET /v1/policy/versions: list policy versions (viewer+)
    - GET /v1/policy/versions/{version}: get policy version detail (viewer+)
    - GET /v1/policy/decisions: list policy decisions (operator/admin)

    If store is None (typical for extensions), it will be injected from app.state.mlops.policy_store.
    """
    router = APIRouter()

    @router.get("/v1/policy/versions")
    async def get_policy_versions(
        principal: Annotated[Principal, Depends(require_role("viewer", "operator", "admin"))],
        request: Request = None,
    ):
        """Get list of all policy versions."""
        # Get policy store from app state if not provided
        _store = store
        if _store is None:
            _store = request.app.state.mlops.policy_store

        versions = _store.history()
        return {"versions": versions}

    @router.get("/v1/policy/versions/{version}")
    async def get_policy_version(
        version: str,
        principal: Annotated[Principal, Depends(require_role("viewer", "operator", "admin"))],
        request: Request = None,
    ):
        """Get a specific policy version (version must be int >= 1)."""
        # Get policy store from app state if not provided
        _store = store
        if _store is None:
            _store = request.app.state.mlops.policy_store

        # Parse and validate version parameter (-> 422 on error)
        try:
            version_int = int(version)
        except (ValueError, TypeError):
            raise _Unprocessable("validation error in fields: version")

        if version_int < 1:
            raise _Unprocessable("validation error in fields: version")

        # Get the policy (may raise NotFound -> 404)
        policy_bundle = _store.get(version_int)
        model = PolicyBundleModel.from_domain(policy_bundle)
        result = model.model_dump()
        result["hash"] = policy_bundle.hash
        return result

    @router.get("/v1/policy/decisions")
    async def get_policy_decisions(
        principal: Annotated[Principal, Depends(require_role("operator", "admin"))],
        limit: Optional[int] = Query(50, ge=1, le=500),
        request: Request = None,
    ):
        """Get the last N policy decisions (limit 1-500, default 50)."""
        # Get policy store from app state if not provided
        _store = store
        if _store is None:
            _store = request.app.state.mlops.policy_store

        # Validate limit (Query validation handles this, but be explicit for clarity)
        if not isinstance(limit, int) or limit < 1 or limit > 500:
            raise _Unprocessable("validation error in fields: limit")

        decisions = _store.decision_log()
        last_n = decisions[-limit:] if limit else decisions

        # Convert decision objects to dicts if needed
        result_decisions = []
        for d in last_n:
            if hasattr(d, "__dict__"):
                result_decisions.append(d.__dict__)
            elif isinstance(d, dict):
                result_decisions.append(d)
            else:
                result_decisions.append({"raw": str(d)})

        return {"decisions": result_decisions}

    return router
