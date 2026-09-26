"""
S3.2 Incident Management: classification, lifecycle, escalation, resolution.

Provides incident severity classification, lifecycle management (open, steps,
escalation, resolution), and audit integration via ChainedAuditStore.
"""

import copy
import math
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from mlops.audit import ChainedAuditStore
from mlops.kernel import Clock, Conflict, NotFound, ValidationFailed


class Severity(Enum):
    """Incident severity with response time (in minutes)."""

    P0 = 15
    P1 = 60
    P2 = 240
    P3 = 1440

    @property
    def response_minutes(self) -> int:
        """Return the response time in minutes for this severity."""
        return self.value


CHECKLIST = ("detection", "impact_assessment", "recent_changes_review", "mitigation_options", "root_cause")

PM_KEYS = {"timeline", "root_cause", "user_impact", "preventive_measures", "monitoring_gap"}


def _validate_string(value: Any, name: str, allow_empty: bool = False) -> str:
    """Validate that value is a non-empty string with no control characters."""
    if not isinstance(value, str):
        raise ValidationFailed(f"{name} must be a string, got {type(value).__name__}")
    if not value and not allow_empty:
        raise ValidationFailed(f"{name} must be non-empty")
    # Check for control characters (ASCII 0-31 and 127)
    for ch in value:
        if ord(ch) < 32 or ord(ch) == 127:
            raise ValidationFailed(f"{name} contains control characters")
    return value


def _validate_number(value: Any, name: str, allow_bool: bool = False) -> float:
    """Validate that value is a finite real number (not bool, str, None, NaN, inf)."""
    if isinstance(value, bool):
        if not allow_bool:
            raise ValidationFailed(f"{name} cannot be a boolean")
        return float(value)
    if not isinstance(value, (int, float)):
        raise ValidationFailed(f"{name} must be numeric, got {type(value).__name__}")
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        raise ValidationFailed(f"{name} cannot be NaN or Inf")
    return fv


def classify(signal: Dict[str, Any]) -> Severity:
    """
    Classify an incident signal into a severity level.

    Rules (in priority order):
    1. serving_failure or null_predictions => P0
    2. accuracy_drop_pct > 10 => P1
    3. feature_psi > 0.3 or drift_level == "alert" => P2
    4. Any other reported condition (pipeline_delay, drift_level != "ok", accuracy > 0) => P3
    5. Empty signal or unknown keys => ValidationFailed

    Args:
        signal: dict with optional keys: serving_failure (bool), null_predictions (bool),
                accuracy_drop_pct (float >= 0), feature_psi (float >= 0),
                drift_level ("ok"|"warn"|"alert"), pipeline_delay (bool)

    Returns:
        Severity: P0, P1, P2, or P3

    Raises:
        ValidationFailed: if signal is invalid
    """
    if not isinstance(signal, dict):
        raise ValidationFailed("signal must be a dict")

    if not signal:
        raise ValidationFailed("signal cannot be empty")

    # Known keys
    known_keys = {"serving_failure", "null_predictions", "accuracy_drop_pct", "feature_psi", "drift_level", "pipeline_delay"}

    # Validate keys and collect values
    values = {}
    for key, value in signal.items():
        if key not in known_keys:
            raise ValidationFailed(f"unknown signal key: {key}")
        values[key] = value

    # Validate and extract values
    serving_failure = False
    null_predictions = False
    accuracy_drop_pct = None
    feature_psi = None
    drift_level = None
    pipeline_delay = False

    if "serving_failure" in values:
        v = values["serving_failure"]
        if not isinstance(v, bool):
            raise ValidationFailed("serving_failure must be bool")
        serving_failure = v

    if "null_predictions" in values:
        v = values["null_predictions"]
        if not isinstance(v, bool):
            raise ValidationFailed("null_predictions must be bool")
        null_predictions = v

    if "accuracy_drop_pct" in values:
        v = values["accuracy_drop_pct"]
        if isinstance(v, bool):
            raise ValidationFailed("accuracy_drop_pct cannot be bool")
        try:
            fv = _validate_number(v, "accuracy_drop_pct")
        except ValidationFailed:
            raise
        if fv < 0:
            raise ValidationFailed("accuracy_drop_pct cannot be negative")
        accuracy_drop_pct = fv

    if "feature_psi" in values:
        v = values["feature_psi"]
        if isinstance(v, bool):
            raise ValidationFailed("feature_psi cannot be bool")
        try:
            fv = _validate_number(v, "feature_psi")
        except ValidationFailed:
            raise
        if fv < 0:
            raise ValidationFailed("feature_psi cannot be negative")
        feature_psi = fv

    if "drift_level" in values:
        v = values["drift_level"]
        if not isinstance(v, str):
            raise ValidationFailed("drift_level must be string")
        if v not in ("ok", "warn", "alert"):
            raise ValidationFailed(f"drift_level must be 'ok', 'warn', or 'alert', got {v}")
        drift_level = v

    if "pipeline_delay" in values:
        v = values["pipeline_delay"]
        if not isinstance(v, bool):
            raise ValidationFailed("pipeline_delay must be bool")
        pipeline_delay = v

    # Apply classification rules
    if serving_failure or null_predictions:
        return Severity.P0

    if accuracy_drop_pct is not None and accuracy_drop_pct > 10:
        return Severity.P1

    if (feature_psi is not None and feature_psi > 0.3) or drift_level == "alert":
        return Severity.P2

    # Check if anything is reported
    if (accuracy_drop_pct is not None and accuracy_drop_pct > 0) or pipeline_delay or drift_level in ("warn",):
        return Severity.P3

    # If we got here, we have a signal dict but nothing that triggers classification
    # This shouldn't happen with proper signal validation, but if all values are 0 or ok:
    return Severity.P3


class IncidentManager:
    """Manages incident lifecycle: open, steps, escalation, resolution."""

    def __init__(self, audit: ChainedAuditStore, clock: Clock):
        """
        Initialize the incident manager.

        Args:
            audit: ChainedAuditStore for audit trail
            clock: Clock for timestamps
        """
        self._audit = audit
        self._clock = clock
        self._incidents: Dict[str, Dict[str, Any]] = {}
        self._next_id = 0

    def _new_id(self) -> str:
        """Generate a new incident ID."""
        self._next_id += 1
        return f"inc-{self._next_id}"

    def open(self, actor: str, title: str, signal: Dict[str, Any]) -> str:
        """
        Open a new incident.

        Args:
            actor: person opening the incident (non-empty str)
            title: incident title (non-empty str)
            signal: incident signal dict (validated by classify)

        Returns:
            incident_id: unique identifier

        Raises:
            ValidationFailed: if actor, title, or signal is invalid
        """
        # Validate actor and title before any state change
        actor = _validate_string(actor, "actor")
        title = _validate_string(title, "title")

        # Validate signal and get severity (this will raise ValidationFailed if invalid)
        severity = classify(signal)

        # Now create the incident
        iid = self._new_id()
        ts = self._clock.now()
        deadline = ts + severity.response_minutes * 60

        record = {
            "incident_id": iid,
            "actor": actor,
            "title": title,
            "signal": copy.deepcopy(signal),
            "severity": severity,
            "opened_ts": ts,
            "deadline": deadline,
            "status": "open",
            "steps_completed": [],
            "postmortem": None,
            "current_tier": 1,
        }

        self._incidents[iid] = record

        # Audit the open
        self._audit.append(actor, "incident.open", iid, "allow", ts=ts)

        return iid

    def get(self, incident_id: str) -> Dict[str, Any]:
        """
        Get a copy of an incident record.

        Args:
            incident_id: unique incident identifier

        Returns:
            dict: deep copy of the incident record

        Raises:
            NotFound: if incident does not exist
        """
        if incident_id not in self._incidents:
            raise NotFound(f"incident {incident_id} not found")
        return copy.deepcopy(self._incidents[incident_id])

    def complete_step(self, actor: str, incident_id: str, step: str, note: str) -> None:
        """
        Complete a step in the incident investigation.

        Steps must be completed in CHECKLIST order.

        Args:
            actor: person completing the step
            incident_id: unique incident identifier
            step: step name (must be in CHECKLIST)
            note: note on the step (non-empty str)

        Raises:
            ValidationFailed: if actor, step, or note is invalid
            NotFound: if incident does not exist
            Conflict: if step is out of order
        """
        # Validate inputs before any state change
        actor = _validate_string(actor, "actor")
        step = _validate_string(step, "step")
        note = _validate_string(note, "note")

        if incident_id not in self._incidents:
            raise NotFound(f"incident {incident_id} not found")

        record = self._incidents[incident_id]

        # Check that step is in CHECKLIST
        if step not in CHECKLIST:
            raise ValidationFailed(f"unknown step: {step}")

        # Check that step is next in order
        next_index = len(record["steps_completed"])
        if next_index >= len(CHECKLIST) or CHECKLIST[next_index] != step:
            raise Conflict(f"step {step} is out of order")

        # Perform the update
        ts = self._clock.now()
        record["steps_completed"].append(step)

        # Audit the step completion
        self._audit.append(actor, "incident.step", incident_id, "allow", ts=ts, meta={"step": step, "note": note})

    def list_open_incident_ids(self) -> List[str]:
        """Return ids of all incidents not yet resolved.

        Lets a periodic worker enumerate incidents to check for overdue
        escalation without reaching into the private _incidents dict.
        """
        return [iid for iid, rec in self._incidents.items() if rec["status"] != "resolved"]

    def escalate_if_overdue(self, incident_id: str) -> int:
        """
        Check if incident is overdue and escalate if needed.

        Returns the current on-call tier:
        - 1: primary (initial)
        - 2: escalated when elapsed > response_minutes * 60 seconds
        - 3: further escalated when elapsed > 2 * response_minutes * 60 seconds

        Audit entry is written only when tier RISES.

        Args:
            incident_id: unique incident identifier

        Returns:
            int: current on-call tier (1, 2, or 3)

        Raises:
            NotFound: if incident does not exist
        """
        if incident_id not in self._incidents:
            raise NotFound(f"incident {incident_id} not found")

        record = self._incidents[incident_id]

        # For resolved incidents, return the current tier without escalating
        if record["status"] == "resolved":
            return record["current_tier"]

        ts = self._clock.now()
        elapsed = ts - record["opened_ts"]
        response_seconds = record["severity"].response_minutes * 60

        new_tier = record["current_tier"]

        # Tier 2 when elapsed > response_minutes*60
        if elapsed > response_seconds and new_tier < 2:
            new_tier = 2

        # Tier 3 when elapsed > 2*response_minutes*60
        if elapsed > 2 * response_seconds and new_tier < 3:
            new_tier = 3

        # Audit only when tier rises
        if new_tier > record["current_tier"]:
            record["current_tier"] = new_tier
            self._audit.append("system", "incident.escalate", incident_id, "allow", ts=ts, meta={"tier": new_tier})

        return record["current_tier"]

    def resolve(self, actor: str, incident_id: str, postmortem: Optional[Dict[str, str]]) -> None:
        """
        Resolve an incident.

        All steps in CHECKLIST must be completed first.
        P0 and P1 incidents require a complete postmortem dict with keys:
          {timeline, root_cause, user_impact, preventive_measures, monitoring_gap}
        each with non-empty string values.
        P2 and P3 incidents may omit the postmortem.

        Args:
            actor: person resolving the incident
            incident_id: unique incident identifier
            postmortem: postmortem dict (required for P0/P1, optional for P2/P3)

        Raises:
            ValidationFailed: if actor or postmortem is invalid
            NotFound: if incident does not exist
            Conflict: if not all steps are complete or incident already resolved
        """
        # Validate actor before any state change
        actor = _validate_string(actor, "actor")

        if incident_id not in self._incidents:
            raise NotFound(f"incident {incident_id} not found")

        record = self._incidents[incident_id]

        # Check if already resolved
        if record["status"] == "resolved":
            raise Conflict(f"incident {incident_id} is already resolved")

        # Check that all steps are complete
        if len(record["steps_completed"]) < len(CHECKLIST):
            raise Conflict(f"not all steps are complete")

        # Validate postmortem if required (P0 and P1)
        if record["severity"] in (Severity.P0, Severity.P1):
            if postmortem is None:
                raise ValidationFailed("P0 and P1 incidents require a postmortem")
            if not isinstance(postmortem, dict):
                raise ValidationFailed("postmortem must be a dict")

            # Check all required keys are present
            if set(postmortem.keys()) != PM_KEYS:
                raise ValidationFailed(f"postmortem must have exactly keys {PM_KEYS}")

            # Validate each value is a non-empty string
            for key in PM_KEYS:
                if key not in postmortem:
                    raise ValidationFailed(f"postmortem missing key {key}")
                val = postmortem[key]
                if not isinstance(val, str):
                    raise ValidationFailed(f"postmortem[{key}] must be string, got {type(val).__name__}")
                if not val:
                    raise ValidationFailed(f"postmortem[{key}] must be non-empty")

            record["postmortem"] = copy.deepcopy(postmortem)
        elif postmortem is not None:
            # P2 and P3 may have postmortem but don't require it
            if isinstance(postmortem, dict):
                # Validate structure if provided
                if set(postmortem.keys()) != PM_KEYS:
                    raise ValidationFailed(f"postmortem must have exactly keys {PM_KEYS}")
                for key in PM_KEYS:
                    val = postmortem[key]
                    if not isinstance(val, str):
                        raise ValidationFailed(f"postmortem[{key}] must be string, got {type(val).__name__}")
                    if not val:
                        raise ValidationFailed(f"postmortem[{key}] must be non-empty")
                record["postmortem"] = copy.deepcopy(postmortem)

        # Mark as resolved and audit
        ts = self._clock.now()
        record["status"] = "resolved"
        self._audit.append(actor, "incident.resolve", incident_id, "allow", ts=ts)
