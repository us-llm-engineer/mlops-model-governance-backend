"""MlopsClient SDK: HTTP client for the MLOps control plane API."""

import json
import uuid
import urllib.parse
from typing import Any, Optional

import httpx
from tenacity import (
    Retrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)


class MlopsApiError(Exception):
    """API error response."""

    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

    def __repr__(self) -> str:
        return f"MlopsApiError(status={self.status}, code={self.code!r}, message={self.message!r})"

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


# The server sends an SSE keep-alive comment about every 15 s, so an idle tail is silent for that
# long. httpx's default 5 s read timeout would kill a healthy quiet stream; 45 s leaves margin for
# three missed pings while still detecting a dead peer.
_TAIL_TIMEOUT = httpx.Timeout(10.0, read=45.0)


def _reject_constant(name: str):
    raise ValueError("non-finite JSON constant")


class _Transient(Exception):
    """Marker for transient transport/network errors."""

    pass


class MlopsClient:
    """HTTP client for MLOps control plane API."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        token: str,
        retries: int = 3,
        backoff_s: float = 0.1,
    ):
        # Exactly one of base_url or client
        if (base_url is None and client is None) or (base_url is not None and client is not None):
            raise ValueError("exactly one of base_url or client is required")

        # Validate token
        if not isinstance(token, str) or not token or len(token) > 128:
            raise ValueError("token must be a non-empty string")

        self._token = token
        self._retries = retries
        self._backoff_s = backoff_s

        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(base_url=base_url)

    def __repr__(self) -> str:
        return f"MlopsClient(retries={self._retries}, backoff_s={self._backoff_s})"

    def __str__(self) -> str:
        return repr(self)

    def _validate_json_serializable(self, obj: Any) -> None:
        """Validate that an object can be JSON serialized without NaN/Infinity."""
        try:
            json.dumps(obj, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"not JSON serializable: {e}") from e

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
    ) -> dict:
        """Make an HTTP request with retries."""
        headers = {"Authorization": f"Bearer {self._token}"}

        # Generate idempotency key for POST requests
        idempotency_key = None
        if method == "POST":
            idempotency_key = uuid.uuid4().hex

        # Validate body before any request
        if body is not None:
            self._validate_json_serializable(body)

        last_error = None

        def do_request() -> dict:
            """Single attempt of the request."""
            nonlocal last_error

            req_headers = dict(headers)
            if idempotency_key is not None:
                req_headers["Idempotency-Key"] = idempotency_key

            try:
                response = self._client.request(method, path, json=body, headers=req_headers)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteError, httpx.PoolTimeout) as e:
                raise _Transient(str(e)) from e

            # Try to parse JSON
            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError):
                # Non-JSON response
                exc = MlopsApiError(response.status_code, "bad_response", "non-JSON response")
                last_error = exc
                # Check if transient status - if so, retry but save the error
                if response.status_code in (502, 503, 504):
                    raise _Transient(f"HTTP {response.status_code}") from exc
                raise exc

            # Check for error envelope
            if response.status_code >= 400:
                if isinstance(data, dict) and "error" in data:
                    err = data["error"]
                    if isinstance(err, dict):
                        code = err.get("code", "unknown")
                        message = err.get("message", "")
                        exc = MlopsApiError(response.status_code, code, message)
                        last_error = exc
                        # Check if transient status - if so, retry but save the error
                        if response.status_code in (502, 503, 504):
                            raise _Transient(f"HTTP {response.status_code}") from exc
                        raise exc
                exc = MlopsApiError(response.status_code, "bad_response", "malformed error response")
                last_error = exc
                if response.status_code in (502, 503, 504):
                    raise _Transient(f"HTTP {response.status_code}") from exc
                raise exc

            # Success
            return data

        # Retry loop
        try:
            retry_strategy = Retrying(
                stop=stop_after_attempt(self._retries + 1),
                wait=wait_exponential_jitter(initial=self._backoff_s, max=2),
                retry=retry_if_exception_type(_Transient),
                reraise=True,
            )
            return retry_strategy(do_request)
        except _Transient:
            # Exhausted retries on transient error
            if last_error is not None:
                # Raise the last actual error we encountered
                raise last_error
            # This was a transport error
            raise MlopsApiError(0, "transport", "transport error")
        except MlopsApiError:
            raise
        except RetryError as e:
            # This shouldn't happen with reraise=True, but just in case
            if last_error is not None:
                raise last_error
            raise MlopsApiError(0, "transport", "transport error") from e

    def _encode_segment(self, segment: str) -> str:
        """Percent-encode a path segment."""
        return urllib.parse.quote(segment, safe="")

    # API methods

    def health(self) -> dict:
        """GET /healthz."""
        return self._request("GET", "/healthz")

    def metrics_text(self) -> str:
        """GET /metrics (returns raw text)."""
        response = self._client.get("/metrics", headers={"Authorization": f"Bearer {self._token}"})
        if response.status_code >= 400:
            raise MlopsApiError(response.status_code, "error", "metrics request failed")
        return response.text

    def register_model(
        self,
        model_id: str,
        version_id: str,
        artifact_hash: str,
        dataset_version: Optional[str] = None,
    ) -> dict:
        """POST /v1/models."""
        body = {
            "model_id": model_id,
            "version_id": version_id,
            "artifact_hash": artifact_hash,
        }
        if dataset_version is not None:
            body["dataset_version"] = dataset_version
        return self._request("POST", "/v1/models", body)

    def register_dataset(
        self,
        name: str,
        content_hash: str,
        rows: int,
        schema: dict,
    ) -> dict:
        """POST /v1/datasets."""
        body = {
            "name": name,
            "content_hash": content_hash,
            "rows": rows,
            "dataset_schema": schema,
        }
        return self._request("POST", "/v1/datasets", body)

    def lineage_blast_radius(self, model_id: str, version_id: str) -> dict:
        """GET /v1/lineage/{model_id}/{version_id}/blast-radius."""
        path = f"/v1/lineage/{self._encode_segment(model_id)}/{self._encode_segment(version_id)}/blast-radius"
        return self._request("GET", path)

    def lineage_ancestors(self, model_id: str, version_id: str) -> dict:
        """GET /v1/lineage/{model_id}/{version_id}/ancestors."""
        path = f"/v1/lineage/{self._encode_segment(model_id)}/{self._encode_segment(version_id)}/ancestors"
        return self._request("GET", path)

    def validate_model(
        self,
        model_id: str,
        version_id: str,
        evidence: dict,
    ) -> dict:
        """POST /v1/models/{model_id}/versions/{version_id}/validate."""
        path = f"/v1/models/{self._encode_segment(model_id)}/versions/{self._encode_segment(version_id)}/validate"
        body = {"evidence": evidence}
        return self._request("POST", path, body)

    def transition(
        self,
        model_id: str,
        version_id: str,
        to_stage: str,
        context: Optional[dict] = None,
    ) -> dict:
        """POST /v1/models/{model_id}/versions/{version_id}/transition."""
        path = f"/v1/models/{self._encode_segment(model_id)}/versions/{self._encode_segment(version_id)}/transition"
        body = {"to_stage": to_stage}
        if context is not None:
            body["context"] = context
        return self._request("POST", path, body)

    def rollback(self, model_id: str) -> dict:
        """POST /v1/models/{model_id}/rollback."""
        path = f"/v1/models/{self._encode_segment(model_id)}/rollback"
        return self._request("POST", path, {})

    def get_model(self, model_id: str) -> dict:
        """GET /v1/models/{model_id}."""
        path = f"/v1/models/{self._encode_segment(model_id)}"
        return self._request("GET", path)

    def publish_policy(
        self,
        rules: list,
        name: str,
        version: int,
        note: Optional[str] = None,
    ) -> dict:
        """POST /v1/policy/publish."""
        body = {
            "rules": rules,
            "name": name,
            "version": version,
        }
        if note is not None:
            body["note"] = note
        return self._request("POST", "/v1/policy/publish", body)

    def activate_policy(
        self,
        version: int,
        human_approved_by: Optional[str] = None,
    ) -> dict:
        """POST /v1/policy/{version}/activate."""
        path = f"/v1/policy/{version}/activate"
        body = {}
        if human_approved_by is not None:
            body["human_approved_by"] = human_approved_by
        return self._request("POST", path, body)

    def decide(self, action: str, context: dict) -> dict:
        """POST /v1/policy/decide."""
        body = {"action": action, "context": context}
        return self._request("POST", "/v1/policy/decide", body)

    def active_policy(self) -> dict:
        """GET /v1/policy/active."""
        return self._request("GET", "/v1/policy/active")

    def lint(
        self,
        documents: list,
        image_allowlist: Optional[list] = None,
    ) -> dict:
        """POST /v1/lint."""
        body = {"documents": documents}
        if image_allowlist is not None:
            body["image_allowlist"] = image_allowlist
        return self._request("POST", "/v1/lint", body)

    def drift_check(
        self,
        reference: list,
        window: list,
    ) -> dict:
        """POST /v1/drift/check."""
        body = {"reference": reference, "window": window}
        return self._request("POST", "/v1/drift/check", body)

    def audit_head(self) -> dict:
        """GET /v1/audit/head."""
        return self._request("GET", "/v1/audit/head")

    def audit_export(self, limit: Optional[int] = None) -> dict:
        """GET /v1/audit/export."""
        params = {}
        if limit is not None:
            params["limit"] = str(limit)
        path = "/v1/audit/export"
        if params:
            path += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        return self._request("GET", path)

    def open_incident(self, title: str, signal: dict) -> dict:
        """POST /v1/incidents."""
        body = {"title": title, "signal": signal}
        return self._request("POST", "/v1/incidents", body)

    def get_incident(self, incident_id: str) -> dict:
        """GET /v1/incidents/{id}."""
        path = f"/v1/incidents/{self._encode_segment(incident_id)}"
        return self._request("GET", path)

    def complete_incident_step(self, incident_id: str, step: str, note: str) -> dict:
        """POST /v1/incidents/{id}/steps."""
        path = f"/v1/incidents/{self._encode_segment(incident_id)}/steps"
        body = {"step": step, "note": note}
        return self._request("POST", path, body)

    def resolve_incident(self, incident_id: str, postmortem: dict) -> dict:
        """POST /v1/incidents/{id}/resolve."""
        path = f"/v1/incidents/{self._encode_segment(incident_id)}/resolve"
        body = {"postmortem": postmortem}
        return self._request("POST", path, body)

    def policy_versions(self) -> dict:
        """GET /v1/policy/versions."""
        return self._request("GET", "/v1/policy/versions")

    def policy_version(self, version: int) -> dict:
        """GET /v1/policy/versions/{v}."""
        path = f"/v1/policy/versions/{version}"
        return self._request("GET", path)

    def policy_decisions(self, limit: int = 50) -> dict:
        """GET /v1/policy/decisions."""
        path = f"/v1/policy/decisions?limit={limit}"
        return self._request("GET", path)

    def audit_verify(self, export: list, head: dict) -> dict:
        """POST /v1/audit/verify."""
        body = {"export": export, "head": head}
        return self._request("POST", "/v1/audit/verify", body)

    def audit_tail(self):
        """GET /v1/audit/tail (streaming SSE).

        Yields parsed SSE data lines as dicts. This is a generator that does NOT
        go through the shared _request retry wrapper, since retrying a broken
        half-open stream is unsafe.
        """
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            with self._client.stream(
                "GET", "/v1/audit/tail", headers=headers, timeout=_TAIL_TIMEOUT
            ) as response:
                if response.status_code >= 400:
                    raise MlopsApiError(response.status_code, "error", "stream request failed")
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        try:
                            event = json.loads(data_str, parse_constant=_reject_constant)
                        except (ValueError, RecursionError):
                            raise MlopsApiError(0, "bad_response", "invalid JSON in SSE line") from None
                        if not isinstance(event, dict):
                            raise MlopsApiError(0, "bad_response", "SSE event is not an object")
                        yield event
        except httpx.HTTPError:
            # Read timeouts, dropped connections and protocol errors all end the stream; surface
            # one typed error (never the raw httpx text) so callers/CLI handle a single type.
            raise MlopsApiError(0, "transport", "audit stream interrupted") from None
