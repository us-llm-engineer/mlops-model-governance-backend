"""
S2.3 Model Stages: Model version registration, staging, and production gates.

This module provides:

* Stage: Enum for model version lifecycle (REGISTERED, STAGING, PRODUCTION, ARCHIVED)
* ModelVersionRecord: Immutable record of a model version with its stage
* ModelRegistry: Manage model versions and enforce state machine transitions

State machine:
  REGISTERED -> STAGING -> PRODUCTION -> ARCHIVED
  STAGING -> ARCHIVED
  ARCHIVED is terminal (never rollback target)
  PRODUCTION only reachable when validated=True and passes policy check

Stdlib only: dataclasses, enum, typing, copy, string.
"""

from __future__ import annotations

import copy
import string
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .kernel import Conflict, NotFound, ValidationFailed
from .supply_chain import verify_attestation, _is_hex64, _is_nonempty_str

__all__ = ["Stage", "ModelVersionRecord", "ModelRegistry"]


class Stage(str, Enum):
    """Model version lifecycle stage."""
    REGISTERED = "registered"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class ModelVersionRecord:
    """Immutable record of a model version at a point in time."""
    model_id: str
    version_id: str
    artifact_hash: str
    dataset_version: Optional[str]
    attestation_ok: bool
    validated: bool
    stage: Stage
    previous_production: Optional[str]


class ModelRegistry:
    """Register and manage model versions through a lifecycle state machine."""

    # Allowed state transitions
    _ALLOWED_EDGES = {
        Stage.REGISTERED: {Stage.STAGING},
        Stage.STAGING: {Stage.PRODUCTION, Stage.ARCHIVED},
        Stage.PRODUCTION: {Stage.ARCHIVED},
        Stage.ARCHIVED: set(),  # Terminal state
    }

    def __init__(
        self,
        audit: Any = None,
        decision_point: Any = None,
        datasets: Any = None,
        attestation_secret: Optional[bytes] = None,
        require_attestation: bool = False,
        trust_validated: bool = False,
        require_approver: Optional[Callable[[str], bool]] = None,
    ):
        """Initialize the registry.

        Args:
            audit: Audit store for logging decisions
            decision_point: Optional PolicyDecisionPoint for PRODUCTION gate
            datasets: Optional DatasetRegistry to validate dataset_version
            attestation_secret: Secret for attestation verification
            require_attestation: If True, require valid attestation for registration
            trust_validated: If True, accept validated=True at register; else reject it
            require_approver: Optional callable(actor)->bool to gate PRODUCTION/rollback
        """
        self.audit = audit
        self.decision_point = decision_point
        self.datasets = datasets
        self.attestation_secret = attestation_secret
        self.require_attestation = require_attestation
        self.trust_validated = trust_validated
        self.require_approver = require_approver

        # Store records: (model_id, version_id) -> record with current stage
        # Also maintain history in order
        self._records: Dict[tuple, ModelVersionRecord] = {}
        self._history: Dict[str, List[ModelVersionRecord]] = {}  # model_id -> list of records
        self._by_stage: Dict[tuple, str] = {}  # (model_id, stage) -> version_id of most recent in that stage

    def _commit(self, updates: Dict[str, Any], audit_action: str, audit_resource: str, audit_decision: str, audit_meta: Optional[Dict[str, Any]] = None) -> None:
        """Helper to enforce audit-before-commit ordering.

        Args:
            updates: Dict with keys like 'records', 'history', 'by_stage' containing updates to apply
            audit_action: Audit action name
            audit_resource: Audit resource (model_id or empty string for deny)
            audit_decision: 'allow' or 'deny'
            audit_meta: Optional audit metadata

        Raises:
            OSError or other: If audit.append() raises, state is NOT changed
        """
        # Audit BEFORE commit (if audit.append raises, nothing below runs)
        if self.audit is not None:
            self.audit.append(
                updates.get('actor', 'system'),
                audit_action,
                audit_resource,
                audit_decision,
                meta=audit_meta or {},
            )

        # Only after successful audit, commit state changes
        if 'records' in updates:
            self._records.update(updates['records'])
        if 'history' in updates:
            for model_id, records in updates['history'].items():
                if model_id not in self._history:
                    self._history[model_id] = []
                self._history[model_id].extend(records)
        # Delete old stage keys BEFORE adding new ones (order matters!)
        if 'by_stage_del' in updates:
            for key in updates['by_stage_del']:
                if key in self._by_stage:
                    del self._by_stage[key]
        if 'by_stage' in updates:
            self._by_stage.update(updates['by_stage'])

    def register(
        self,
        actor: str,
        model_id: str,
        version_id: str,
        artifact_hash: str,
        dataset_version: Optional[str] = None,
        attestation: Optional[Dict[str, Any]] = None,
        validated: bool = False,
    ) -> ModelVersionRecord:
        """Register a new model version.

        Args:
            actor: Principal performing registration (non-empty str)
            model_id: Model identifier (non-empty str, no control chars)
            version_id: Version identifier (non-empty str, no whitespace, no control chars)
            artifact_hash: SHA256 hex digest (64 lowercase hex chars)
            dataset_version: Optional dataset version ID (must be str, must exist if datasets is set)
            attestation: Optional attestation dict
            validated: Whether this version passed validation (must be bool, rejected if True without trust_validated)

        Returns:
            ModelVersionRecord with stage=REGISTERED

        Raises:
            ValidationFailed: Invalid inputs
            Conflict: Duplicate (model_id, version_id)
            NotFound: dataset_version doesn't exist
        """
        # Validate inputs BEFORE any state changes or audit calls
        if not _is_nonempty_str(actor):
            raise ValidationFailed(f"actor must be non-empty str, got {type(actor).__name__}")

        if not _is_nonempty_str(model_id):
            raise ValidationFailed(f"model_id must be non-empty str")
        if any(ord(c) < 32 or ord(c) >= 127 for c in model_id):
            raise ValidationFailed(f"model_id contains control characters")

        if not _is_nonempty_str(version_id):
            raise ValidationFailed(f"version_id must be non-empty str")
        if any(c.isspace() for c in version_id) or any(ord(c) < 32 or ord(c) >= 127 for c in version_id):
            raise ValidationFailed(f"version_id contains whitespace or control characters")

        # Validate artifact_hash
        if not _is_hex64(artifact_hash):
            raise ValidationFailed(f"Invalid artifact_hash")

        # Validate validated is bool
        if not isinstance(validated, bool):
            raise ValidationFailed(f"validated must be bool, got {type(validated).__name__}")

        # Check if validated=True requires trust_validated
        if validated and not self.trust_validated:
            raise ValidationFailed("register(validated=True) requires trust_validated=True")

        # Check for duplicate (model_id, version_id)
        key = (model_id, version_id)
        if key in self._records:
            raise Conflict(f"Model version already registered: {model_id}/{version_id}")

        # Validate dataset_version if provided
        if dataset_version is not None:
            if not isinstance(dataset_version, str):
                raise ValidationFailed(f"dataset_version must be str, got {type(dataset_version).__name__}")
            if self.datasets is not None:
                try:
                    self.datasets.get(dataset_version)
                except NotFound:
                    raise NotFound(f"Dataset version not found: {dataset_version}")
            elif dataset_version is not None:
                # datasets is None but dataset_version is not None
                raise ValidationFailed("dataset_version provided but datasets registry not set")

        # Verify attestation if provided or required
        attestation_ok = False
        if attestation is not None:
            if self.attestation_secret is None:
                raise ValidationFailed("attestation_secret not configured")
            if verify_attestation(attestation, self.attestation_secret, artifact_digest=artifact_hash):
                attestation_ok = True
            else:
                raise ValidationFailed("Attestation verification failed")
        elif self.require_attestation:
            raise ValidationFailed("Attestation required but not provided")

        # Build immutable record (no state changes yet)
        record = ModelVersionRecord(
            model_id=model_id,
            version_id=version_id,
            artifact_hash=artifact_hash,
            dataset_version=dataset_version,
            attestation_ok=attestation_ok,
            validated=validated,
            stage=Stage.REGISTERED,
            previous_production=None,
        )

        # Prepare updates (not yet applied)
        updates = {
            'actor': actor,
            'records': {key: record},
            'history': {model_id: [record]},
            'by_stage': {(model_id, Stage.REGISTERED): version_id},
        }

        # Commit with audit-before-state ordering
        self._commit(updates, "model.register", model_id, "allow", {"version_id": version_id})

        return record

    def validate(
        self,
        actor: str,
        model_id: str,
        version_id: str,
        evidence: Dict[str, Any],
    ) -> ModelVersionRecord:
        """Validate a model version based on evidence.

        Sets validated=True on the record.

        Args:
            actor: Principal performing validation (non-empty str)
            model_id: Model identifier
            version_id: Version identifier
            evidence: Validation evidence dict

        Returns:
            Updated ModelVersionRecord with validated=True

        Raises:
            ValidationFailed: Invalid actor or evidence
            NotFound: Version not found
        """
        # Validate inputs
        if not _is_nonempty_str(actor):
            raise ValidationFailed(f"actor must be non-empty str")
        if not isinstance(evidence, dict) or len(evidence) == 0:
            raise ValidationFailed(f"evidence must be non-empty dict")

        key = (model_id, version_id)
        if key not in self._records:
            # Audit the denial
            if self.audit is not None:
                self.audit.append(actor, "model.validate", model_id, "deny", meta={"reason": "not_found"})
            raise NotFound(f"Model version not found: {model_id}/{version_id}")

        # Build new record
        old_record = self._records[key]
        new_record = replace(old_record, validated=True)

        # Prepare updates
        updates = {
            'actor': actor,
            'records': {key: new_record},
            'history': {model_id: [new_record]},
        }

        # Commit with audit-before-state ordering
        self._commit(updates, "model.validate", model_id, "allow", {"version_id": version_id})

        return new_record

    def transition(
        self,
        actor: str,
        model_id: str,
        version_id: str,
        to_stage: Stage,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelVersionRecord:
        """Transition a model version to a new stage.

        Args:
            actor: Principal performing the transition (non-empty str)
            model_id: Model identifier
            version_id: Version identifier
            to_stage: Target stage (must be Stage enum)
            context: Optional policy context dict

        Returns:
            Updated ModelVersionRecord

        Raises:
            ValidationFailed: Invalid to_stage
            NotFound: Version not found
            Conflict: Invalid state transition or missing precondition
            PolicyDenied: Policy decision denied the transition
        """
        # Validate to_stage is Stage enum FIRST
        if not isinstance(to_stage, Stage):
            raise ValidationFailed(f"to_stage must be Stage enum, got {type(to_stage).__name__}")

        if not _is_nonempty_str(actor):
            raise ValidationFailed(f"actor must be non-empty str")

        key = (model_id, version_id)

        if key not in self._records:
            # Audit the denial
            if self.audit is not None:
                self.audit.append(actor, "model.transition", model_id, "deny", meta={"version_id": version_id, "reason": "not_found"})
            raise NotFound(f"Model version not found: {model_id}/{version_id}")

        old_record = self._records[key]
        from_stage = old_record.stage

        # Check if edge is allowed
        if to_stage not in self._ALLOWED_EDGES.get(from_stage, set()):
            # Audit the denial
            if self.audit is not None:
                self.audit.append(actor, "model.transition", model_id, "deny", meta={"version_id": version_id, "from": from_stage.value, "to": to_stage.value})
            raise Conflict(f"Illegal transition {from_stage.value} -> {to_stage.value}")

        # Check preconditions for PRODUCTION
        if to_stage == Stage.PRODUCTION:
            if not old_record.validated:
                # Audit the denial
                if self.audit is not None:
                    self.audit.append(actor, "model.transition", model_id, "deny", meta={"version_id": version_id, "reason": "not_validated"})
                raise Conflict("Cannot transition to PRODUCTION without validated=True")

            # Check require_approver if present
            if self.require_approver is not None:
                if not self.require_approver(actor):
                    if self.audit is not None:
                        self.audit.append(actor, "model.transition", model_id, "deny", meta={"version_id": version_id, "reason": "not_approver"})
                    from .kernel import PolicyDenied
                    raise PolicyDenied(f"Actor {actor} is not an approver")

            # Check policy decision point if present
            if self.decision_point is not None:
                policy_context = context or {}
                try:
                    self.decision_point.enforce("stage.production", policy_context, actor=actor)
                except Exception as e:
                    # Policy denied - it already wrote its own audit entry if it has one
                    # Write one registry entry for the denial only if we have a separate audit
                    if self.audit is not None and self.audit != self.decision_point.audit:
                        self.audit.append(actor, "model.transition", model_id, "deny", meta={"version_id": version_id, "reason": "policy_denied"})
                    raise

        # Build all state changes (not yet committed)
        new_record = replace(old_record, stage=to_stage, previous_production=None)
        updates = {
            'actor': actor,
            'records': {key: new_record},
            'history': {model_id: [new_record]},
            'by_stage': {(model_id, to_stage): version_id},
            'by_stage_del': [],
        }

        # Archive old production if transitioning to PRODUCTION
        if to_stage == Stage.PRODUCTION:
            current_prod_key = (model_id, Stage.PRODUCTION)
            if current_prod_key in self._by_stage:
                old_prod_version_id = self._by_stage[current_prod_key]
                old_prod_key = (model_id, old_prod_version_id)
                if old_prod_key in self._records and old_prod_version_id != version_id:
                    old_prod_record = self._records[old_prod_key]
                    archived_record = replace(old_prod_record, stage=Stage.ARCHIVED)
                    updates['records'][old_prod_key] = archived_record
                    updates['history'][model_id].insert(0, archived_record)
                    updates['by_stage_del'].append(current_prod_key)
                    # Index the archived version under ARCHIVED stage
                    updates['by_stage'][(model_id, Stage.ARCHIVED)] = old_prod_version_id

            # Set previous_production to old production version ID
            previous_prod = self._by_stage.get((model_id, Stage.PRODUCTION))
            if previous_prod is not None:
                updates['records'][key] = replace(new_record, previous_production=previous_prod)
                updates['history'][model_id][-1] = replace(new_record, previous_production=previous_prod)

        # Update stage index
        old_stage_key = (model_id, from_stage)
        if old_stage_key in self._by_stage and self._by_stage[old_stage_key] == version_id:
            updates['by_stage_del'].append(old_stage_key)
            # Find the most recent other version still in the old stage
            if model_id in self._history:
                # Iterate backwards through history to find the most recent version in from_stage
                for hist_rec in reversed(self._history[model_id]):
                    if hist_rec.version_id != version_id:
                        hist_key = (hist_rec.model_id, hist_rec.version_id)
                        if hist_key in self._records:
                            hist_current = self._records[hist_key]
                            if hist_current.stage == from_stage:
                                # Found a version still in the old stage, restore it
                                updates['by_stage'][old_stage_key] = hist_rec.version_id
                                break

        # Commit with audit-before-state ordering
        self._commit(updates, "model.transition", model_id, "allow", {"version_id": version_id, "to_stage": to_stage.value})

        return updates['records'][key]

    def rollback(
        self,
        actor: str,
        model_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelVersionRecord:
        """Rollback to the previous production version.

        Args:
            actor: Principal performing the rollback (non-empty str)
            model_id: Model identifier
            context: Optional policy context dict

        Returns:
            New current production record (the previous version)

        Raises:
            ValidationFailed: Invalid actor
            NotFound: Model not found
            Conflict: No previous production version exists, or policy denies
            PolicyDenied: Policy decision denied the rollback
        """
        if not _is_nonempty_str(actor):
            raise ValidationFailed(f"actor must be non-empty str")

        # Check if model exists at all
        model_exists = any(key[0] == model_id for key in self._records.keys())
        if not model_exists:
            if self.audit is not None:
                self.audit.append(actor, "model.rollback", model_id, "deny", meta={"reason": "model_not_found"})
            raise NotFound(f"Model not found: {model_id}")

        # Get current production version
        current_prod_key = (model_id, Stage.PRODUCTION)
        if current_prod_key not in self._by_stage:
            if self.audit is not None:
                self.audit.append(actor, "model.rollback", model_id, "deny", meta={"reason": "no_production"})
            raise Conflict("No current production version to rollback from")

        current_prod_version_id = self._by_stage[current_prod_key]
        current_prod_key_tuple = (model_id, current_prod_version_id)
        current_prod_record = self._records[current_prod_key_tuple]

        # Check if there's a previous production (ARCHIVED is terminal)
        if current_prod_record.previous_production is None:
            if self.audit is not None:
                self.audit.append(actor, "model.rollback", model_id, "deny", meta={"reason": "no_previous"})
            raise Conflict("No previous production version to rollback to")

        previous_prod_version_id = current_prod_record.previous_production
        previous_prod_key_tuple = (model_id, previous_prod_version_id)

        if previous_prod_key_tuple not in self._records:
            if self.audit is not None:
                self.audit.append(actor, "model.rollback", model_id, "deny", meta={"reason": "previous_not_found"})
            raise NotFound(f"Previous production version not found: {previous_prod_version_id}")

        previous_prod_record = self._records[previous_prod_key_tuple]

        # Check require_approver if present
        if self.require_approver is not None:
            if not self.require_approver(actor):
                if self.audit is not None:
                    self.audit.append(actor, "model.rollback", model_id, "deny", meta={"reason": "not_approver"})
                from .kernel import PolicyDenied
                raise PolicyDenied(f"Actor {actor} is not an approver")

        # Check policy if decision point is present
        if self.decision_point is not None:
            policy_context = context or {}
            try:
                self.decision_point.enforce("stage.production", policy_context, actor=actor)
            except Exception as e:
                # Policy denied - it already wrote its own audit entry
                if self.audit is not None and self.audit != self.decision_point.audit:
                    self.audit.append(actor, "model.rollback", model_id, "deny", meta={"reason": "policy_denied"})
                raise

        # Build all state changes (not yet committed)
        archived_record = replace(current_prod_record, stage=Stage.ARCHIVED)
        restored_record = replace(previous_prod_record, stage=Stage.PRODUCTION)

        updates = {
            'actor': actor,
            'records': {
                current_prod_key_tuple: archived_record,
                previous_prod_key_tuple: restored_record,
            },
            'history': {model_id: [archived_record, restored_record]},
            'by_stage': {
                (model_id, Stage.PRODUCTION): previous_prod_version_id,
                (model_id, Stage.ARCHIVED): current_prod_version_id,
            },
            'by_stage_del': [],
        }

        # Commit with audit-before-state ordering
        self._commit(updates, "model.rollback", model_id, "allow", {"from_version": current_prod_version_id, "to_version": previous_prod_version_id})

        return restored_record

    def current(self, model_id: str, stage: Stage) -> Optional[ModelVersionRecord]:
        """Get the current/most recent version of a model at a given stage.

        Args:
            model_id: Model identifier
            stage: Stage to query

        Returns:
            ModelVersionRecord or None if no version at that stage
        """
        stage_key = (model_id, stage)
        if stage_key not in self._by_stage:
            return None

        version_id = self._by_stage[stage_key]
        key = (model_id, version_id)
        return self._records.get(key)

    def history(self, model_id: str) -> List[ModelVersionRecord]:
        """Get all versions of a model in registration order.

        Args:
            model_id: Model identifier

        Returns:
            List of ModelVersionRecord showing current stages
        """
        if model_id not in self._history:
            return []

        # Return records showing their CURRENT stages
        result = []
        seen_keys = set()
        for hist_rec in self._history[model_id]:
            key = (hist_rec.model_id, hist_rec.version_id)
            if key not in seen_keys:
                seen_keys.add(key)
                # Get the current record for this version
                current_record = self._records.get(key, hist_rec)
                result.append(current_record)

        return result
