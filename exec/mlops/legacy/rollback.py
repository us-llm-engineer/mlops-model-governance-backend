"""
R2.2 -- Idempotent Rollback (GitOps-style).

Claim implemented here:
  C19 (source suite R2_2_S3): rollback (revert manifest + re-validate serving)
      is idempotent -- re-running rollback twice with no intervening promotion
      produces the same manifest state and no duplicate rollback actions.

The executor remembers, per environment, the version it last rolled back to.
A second call against an unchanged manifest is recorded as
``noop_idempotent`` and returns the same version, rather than reverting a
second time. The reverted path re-reads the manifest to confirm serving
converged to the previous version before reporting success.

Stdlib only.
"""

from dataclasses import dataclass
from typing import Dict, List

from mlops.promotion import EnvironmentManifest, now_iso

__all__ = [
    "RollbackRequest",
    "RollbackValidator",
    "RollbackExecutor",
]


@dataclass
class RollbackRequest:
    """A request to roll an environment back to its previous version."""

    environment: str
    requested_by: str


class RollbackValidator:
    """Decide whether an environment has a prior version to roll back to."""

    def can_rollback(self, manifest: EnvironmentManifest, environment: str) -> bool:
        """Return True when ``environment`` has at least two history entries."""
        return len(manifest.history(environment)) >= 2


class RollbackExecutor:
    """Execute idempotent rollbacks with an append-only audit log."""

    def __init__(self, manifest: EnvironmentManifest, validator: RollbackValidator) -> None:
        self.manifest = manifest
        self.validator = validator
        self.audit_log: List[dict] = []
        self._last_result_version: Dict[str, str] = {}

    def rollback(self, request: RollbackRequest) -> dict:
        """Roll ``request.environment`` back to its immediately prior version.

        Idempotent: a repeat call with no intervening change is a no-op. Raises
        :class:`ValueError` when there is no prior version, leaving the
        manifest untouched.
        """
        env = request.environment
        current = self.manifest.current_version(env)

        if env in self._last_result_version and current == self._last_result_version[env]:
            self.audit_log.append(
                {
                    "action": "rollback",
                    "environment": env,
                    "decision": "noop_idempotent",
                    "resulting_version": current,
                    "requested_by": request.requested_by,
                    "timestamp": now_iso(),
                }
            )
            return {"environment": env, "version": current, "action": "noop"}

        if not self.validator.can_rollback(self.manifest, env):
            self.audit_log.append(
                {
                    "action": "rollback",
                    "environment": env,
                    "decision": "rejected_no_prior_version",
                    "timestamp": now_iso(),
                }
            )
            raise ValueError(f"No prior version to roll back to for {env}")

        hist = self.manifest.history(env)
        previous_version = hist[-2][0]
        self.manifest.apply(env, previous_version, f"rollback:{request.requested_by}")
        revalidated = self.manifest.current_version(env) == previous_version
        self._last_result_version[env] = previous_version

        self.audit_log.append(
            {
                "action": "rollback",
                "environment": env,
                "decision": "reverted",
                "from_version": current,
                "to_version": previous_version,
                "revalidated": revalidated,
                "requested_by": request.requested_by,
                "timestamp": now_iso(),
            }
        )
        return {
            "environment": env,
            "version": previous_version,
            "action": "reverted",
            "revalidated": revalidated,
        }
