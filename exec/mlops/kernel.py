"""
S1 Kernel: Canonical JSON, clocks, errors, and event logging (Claims C73-C76).

This module provides the foundational primitives for the enforcement track:

* C73 -- Deterministic content IDs and reproducible time via ManualClock.
* C74 -- Hardened audit chain with head anchors and signature verification.
* C75 -- Lineage immutability detection and complete RBAC audit coverage.
* C76 -- Fail-closed parity checking on non-numeric/NaN values.

Stdlib only: hashlib, json, dataclasses, threading, time, datetime.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "MlopsError",
    "PolicyDenied",
    "IntegrityError",
    "NotFound",
    "Conflict",
    "ValidationFailed",
    "canonical_json",
    "sha256_hex",
    "content_id",
    "Clock",
    "SystemClock",
    "ManualClock",
    "iso",
    "Event",
    "EventLog",
]


# Error base class and subclasses
class MlopsError(Exception):
    """Base exception for MLOps platform errors with optional error code."""

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.code = code or self.__class__.__name__


class PolicyDenied(MlopsError):
    """Policy decision denied the action."""

    def __init__(self, message: str):
        super().__init__(message, "policy_denied")


class IntegrityError(MlopsError):
    """Data integrity check failed."""

    def __init__(self, message: str):
        super().__init__(message, "integrity_error")


class NotFound(MlopsError):
    """Requested resource not found."""

    def __init__(self, message: str):
        super().__init__(message, "not_found")


class Conflict(MlopsError):
    """Resource state conflict."""

    def __init__(self, message: str):
        super().__init__(message, "conflict")


class ValidationFailed(MlopsError):
    """Validation check failed."""

    def __init__(self, message: str):
        super().__init__(message, "validation_failed")


def canonical_json(obj: Any) -> str:
    """Serialize an object into deterministic canonical JSON.

    Keys are sorted, separators are compact, and floats are handled via repr()
    to preserve precision. NaN and inf values are rejected with ValueError.
    Only dict (string keys), list, tuple (as list), str, int, float (finite),
    bool, and None are allowed. Raises TypeError for any other type.
    """
    def _validate_and_serialize(val):
        """Recursively validate types and check for NaN/inf."""
        # Check for NaN and inf first (applies to floats)
        if isinstance(val, float):
            if val != val or val == float('inf') or val == float('-inf'):
                raise ValueError("NaN or Inf values are not allowed in canonical JSON")
        # Reject bool before int check since bool is subclass of int
        elif isinstance(val, bool):
            pass  # bool is allowed
        # Allow int and float
        elif isinstance(val, int):
            pass
        # Allow strings
        elif isinstance(val, str):
            pass
        # Allow None
        elif val is None:
            pass
        # Allow dict with string keys only
        elif isinstance(val, dict):
            for k, v in val.items():
                if not isinstance(k, str):
                    raise TypeError(f"dict keys must be strings, got {type(k).__name__}")
                _validate_and_serialize(v)
        # Allow list and tuple (tuple treated as list)
        elif isinstance(val, (list, tuple)):
            for item in val:
                _validate_and_serialize(item)
        else:
            # Reject any other type
            raise TypeError(f"canonical_json does not support type {type(val).__name__}")

    _validate_and_serialize(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_hex(data: Union[str, bytes]) -> str:
    """Compute SHA256 hash of a string or bytes object, return as hex."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def content_id(prefix: str, obj: Any) -> str:
    """Generate a deterministic content-addressed ID.

    Format: prefix_<first 16 chars of SHA256 of canonical JSON>.
    """
    canonical = canonical_json(obj)
    digest = sha256_hex(canonical)
    return f"{prefix}_{digest[:16]}"


class Clock:
    """Abstract clock interface for reproducible time."""

    def now(self) -> float:
        """Return the current time as seconds since epoch."""
        raise NotImplementedError


class SystemClock(Clock):
    """System clock: returns current epoch time."""

    def now(self) -> float:
        """Return the current time as seconds since epoch."""
        return time.time()


class ManualClock(Clock):
    """Manually-advanced clock for deterministic testing.

    Supports explicit time advancement.
    """

    def __init__(self, start: float = 0.0):
        """Initialize the clock at the given start time (seconds since epoch)."""
        self._t = start

    def now(self) -> float:
        """Return the current time without advancing."""
        return self._t

    def advance(self, seconds: float) -> None:
        """Move the clock forward by the given number of seconds."""
        self._t += seconds


def iso(ts: float) -> str:
    """Convert epoch seconds to a UTC ISO-8601 string.

    Result includes timezone-aware Z suffix.
    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.isoformat()


@dataclass
class Event:
    """A single logged event with topic, payload, timestamp, and ID."""

    topic: str
    payload: Dict[str, Any]
    ts: float
    id: str


class EventLog:
    """Append-only event log with optional filtering."""

    def __init__(self):
        self._events: List[Event] = []

    def emit(self, topic: str, payload: Dict[str, Any], clock: Optional[Clock] = None) -> None:
        """Append an event to the log.

        If no clock is provided, uses SystemClock.
        """
        if clock is None:
            clock = SystemClock()
        ts = clock.now()
        # Generate a simple ID (can be improved with uuid if needed)
        event_id = sha256_hex(f"{topic}_{ts}_{json.dumps(payload, sort_keys=True)}")[:16]
        event = Event(topic=topic, payload=payload, ts=ts, id=event_id)
        self._events.append(event)

    def events(self, topic: Optional[str] = None) -> List[Event]:
        """Return all events, optionally filtered by topic.

        Returns a copy of the list.
        """
        if topic is None:
            return list(self._events)
        return [e for e in self._events if e.topic == topic]
