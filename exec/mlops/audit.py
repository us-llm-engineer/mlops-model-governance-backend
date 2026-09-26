"""
R3.2 -- Audit & Evidence Export (C29, C30, C31, C32).

This module is the production lifting of four self-contained reference suites
into one importable surface:

* C29 / tests/R3_2_S1.py -- immutable, append-only, HMAC-SHA256 signed audit
  log. ``AuditEntry`` is a frozen dataclass and ``AuditStore`` exposes no
  update/delete/remove method, so history can only grow.
* C30 / tests/R3_2_S2.py -- signed JSON-lines export with ANDed filters and a
  record-level ``verify_signature``. ``AuditExporter`` emits one JSON line per
  matching entry, each carrying its signature under ``__signature__``.
* C31 / tests/R3_2_S3.py -- ``IndexedAuditStore`` keeps per-actor and
  per-action indexes updated on every append so a filtered query is
  O(matches), meeting the <100ms budget across 10k+ entries.
* C32 / tests/R3_2_S4.py -- ``ChainedAuditStore`` chains each signature over
  the previous entry's signature, and ``ChainVerifier`` recomputes the chain
  forward from genesis so one corrupted or missing entry invalidates every
  downstream signature.

UNIFICATION (append signature)
------------------------------
The two reference ``AuditStore.append`` shapes differ:

* R3_2_S1 calls ``append(actor, action, resource, decision,
  approval_trace="gate_run:g-1")`` -- the timestamp is auto-generated and
  ``approval_trace`` is a keyword argument.
* R3_2_S2 calls ``append(actor, action, resource, decision, timestamp,
  approval_trace="none")`` -- an explicit timestamp is passed as the fifth
  positional argument.

This module unifies them into::

    append(self, actor, action, resource, decision, timestamp=None,
           approval_trace="none")

where ``timestamp`` defaults to :func:`now_iso` when ``None``. This satisfies
both call shapes exactly: S1's keyword-only ``approval_trace`` binds by name,
and S2's fifth positional binds to ``timestamp``.

Stdlib only: hashlib, hmac, json, dataclasses, datetime.
"""

import copy
import hashlib
import hmac
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = [
    "now_iso",
    "canonical_payload",
    "AuditEntry",
    "AuditStore",
    "AuditExporter",
    "IndexedAuditStore",
    "ChainedAuditStore",
    "ChainVerifier",
]


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def canonical_payload(
    actor: str,
    action: str,
    resource: str,
    decision: str,
    approval_trace: str,
    timestamp: str,
) -> bytes:
    """Serialize an audit record into its canonical signed byte payload.

    Fields are sorted so the same logical record always produces the same
    bytes, independent of insertion order.
    """
    payload = {
        "timestamp": timestamp,
        "actor": actor,
        "action": action,
        "resource": resource,
        "decision": decision,
        "approval_trace": approval_trace,
    }
    return json.dumps(payload, sort_keys=True).encode()


@dataclass(frozen=True)
class AuditEntry:
    """An immutable, signed audit-log entry (C29).

    Frozen so an appended entry cannot be edited in place; the HMAC signature
    additionally binds to content so tampering with serialized state is
    detectable.
    """

    timestamp: str
    actor: str
    action: str
    resource: str
    decision: str
    approval_trace: str
    signature: str


class AuditStore:
    """Append-only in-memory audit store with HMAC-SHA256 signed entries.

    Immutability is enforced at the entry level (frozen dataclass) and at the
    store level: no update/delete/remove API is exposed, so history can only
    be appended to.
    """

    def __init__(self, secret: bytes):
        self._secret = secret
        self._entries: List[AuditEntry] = []

    def append(
        self,
        actor: str,
        action: str,
        resource: str,
        decision: str,
        timestamp: Optional[str] = None,
        approval_trace: str = "none",
    ) -> AuditEntry:
        """Sign and append a new entry, returning it.

        ``timestamp`` defaults to :func:`now_iso` when ``None``; passing it
        explicitly satisfies the explicit-timestamp call shape.
        """
        ts = timestamp if timestamp is not None else now_iso()
        message = canonical_payload(
            actor, action, resource, decision, approval_trace, ts
        )
        signature = hmac.new(self._secret, message, hashlib.sha256).hexdigest()
        entry = AuditEntry(
            ts, actor, action, resource, decision, approval_trace, signature
        )
        self._entries.append(entry)
        return entry

    def all(self) -> List[AuditEntry]:
        """Return a copy of all entries, in append order."""
        return list(self._entries)

    def verify_signature(self, entry: AuditEntry) -> bool:
        """Recompute the entry's HMAC and compare it in constant time."""
        message = canonical_payload(
            entry.actor,
            entry.action,
            entry.resource,
            entry.decision,
            entry.approval_trace,
            entry.timestamp,
        )
        expected = hmac.new(self._secret, message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, entry.signature)


class AuditExporter:
    """Filter and serialize audit entries into signed JSON-lines records (C30).

    Each emitted record carries its own HMAC signature under the
    ``__signature__`` key.
    """

    def __init__(self, store: AuditStore):
        self.store = store

    def export(
        self,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        model: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[str]:
        """Return one JSON line per entry matching every supplied filter.

        Filters are ANDed: ``actor``/``action`` compare for equality, ``model``
        matches ``entry.resource``, and ``start_time``/``end_time`` are
        inclusive string comparisons against the entry timestamp.
        """
        lines: List[str] = []
        for entry in self.store.all():
            if actor is not None and entry.actor != actor:
                continue
            if action is not None and entry.action != action:
                continue
            if model is not None and entry.resource != model:
                continue
            if start_time is not None and entry.timestamp < start_time:
                continue
            if end_time is not None and entry.timestamp > end_time:
                continue
            record = {
                "timestamp": entry.timestamp,
                "actor": entry.actor,
                "action": entry.action,
                "resource": entry.resource,
                "decision": entry.decision,
                "approval_trace": entry.approval_trace,
                "__signature__": entry.signature,
            }
            lines.append(json.dumps(record))
        return lines

    def verify_signature(self, record: Dict[str, Any]) -> bool:
        """Recompute the HMAC over a record's fields and compare it in constant time."""
        message = canonical_payload(
            record["actor"],
            record["action"],
            record["resource"],
            record["decision"],
            record["approval_trace"],
            record["timestamp"],
        )
        expected = hmac.new(self.store._secret, message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, record.get("__signature__", ""))


class IndexedAuditStore:
    """In-memory audit store indexed by actor and action (C31).

    Both indexes are updated on every append, so a filtered query is
    O(matches) rather than a full scan over all entries.
    """

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []
        self._by_actor: Dict[str, List[Dict[str, Any]]] = {}
        self._by_action: Dict[str, List[Dict[str, Any]]] = {}

    def append(
        self,
        actor: str,
        action: str,
        resource: str,
        decision: str,
    ) -> Dict[str, Any]:
        """Append an entry and update both indexes in the same call."""
        entry = {
            "seq": len(self._entries),
            "actor": actor,
            "action": action,
            "resource": resource,
            "decision": decision,
        }
        self._entries.append(entry)
        self._by_actor.setdefault(actor, []).append(entry)
        self._by_action.setdefault(action, []).append(entry)
        return entry

    def query(
        self,
        actor: Optional[str] = None,
        action: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return entries matching the supplied filters.

        Both filters given -> intersection via the actor index filtered by
        action; actor only -> actor index; action only -> action index; no
        filters -> all entries. Unknown filter values yield ``[]`` (never a
        ``KeyError``) because every lookup uses ``.get(..., [])``.
        """
        if actor is not None and action is not None:
            candidates = self._by_actor.get(actor, [])
            return [e for e in candidates if e["action"] == action]
        if actor is not None:
            return list(self._by_actor.get(actor, []))
        if action is not None:
            return list(self._by_action.get(action, []))
        return list(self._entries)


class ChainedAuditStore:
    """Audit store whose entries form a hash chain (C32, C74).

    Each signature is computed over the entry's own fields plus the previous
    entry's signature (``GENESIS`` for the first entry), so corrupting one
    entry breaks the linkage for every entry after it. Hardened to sign
    timestamp and metadata, and to provide a head() anchor for truncation
    detection.
    """

    GENESIS = "0" * 64

    def __init__(self, secret: bytes):
        self._secret = secret
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(
        self,
        actor: str,
        action: str,
        resource: str,
        decision: str,
        ts: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sign and append an entry chained over the previous signature.

        Args:
            actor, action, resource, decision: audit entry fields
            ts: optional timestamp (float epoch seconds); if None, uses current time
            meta: optional metadata dict; defaults to {}

        The signature covers all fields including ts and meta for integrity.
        Deep-copies meta so caller mutations don't affect the stored entry.
        """
        if ts is None:
            ts = datetime.now(timezone.utc).timestamp()
        if meta is None:
            meta = {}

        # Deep-copy meta to prevent caller mutations from affecting stored entry
        meta_copy = copy.deepcopy(meta)

        with self._lock:
            prev_sig = (
                self._entries[-1]["signature"] if self._entries else self.GENESIS
            )
            payload = {
                "actor": actor,
                "action": action,
                "resource": resource,
                "decision": decision,
                "ts": ts,
                "meta": meta_copy,
                "prev_sig": prev_sig,
            }
            message = json.dumps(payload, sort_keys=True).encode()
            signature = hmac.new(self._secret, message, hashlib.sha256).hexdigest()
            entry = dict(payload, signature=signature)
            self._entries.append(entry)
            return entry

    def head(self) -> Dict[str, Any]:
        """Return the chain head anchor: length, last signature, and signed anchor.

        The anchor is an HMAC-SHA256 over a canonical string of length and
        last signature, preventing forged or edited heads.

        Can be used with ChainVerifier.verify_export(records, head=...) to
        detect tail truncation and head forgery.
        """
        with self._lock:
            last_sig = self._entries[-1]["signature"] if self._entries else self.GENESIS
            length = len(self._entries)

            # Compute HMAC-SHA256 anchor over canonical string of length and signature
            anchor_payload = json.dumps(
                {"length": length, "signature": last_sig},
                sort_keys=True
            ).encode()
            anchor = hmac.new(self._secret, anchor_payload, hashlib.sha256).hexdigest()

            return {
                "length": length,
                "signature": last_sig,
                "anchor": anchor,
            }

    def export(self) -> List[Dict[str, Any]]:
        """Return deep copies of the chained entries, in append order.

        Deep-copies to prevent consumer mutations from affecting the chain.
        """
        with self._lock:
            return [copy.deepcopy(e) for e in self._entries]


class ChainVerifier:
    """Verify a chained export forward from genesis (C32, C74).

    Verification uses the *recomputed* expected signature -- not the possibly
    tampered stored ``signature`` field -- as the next entry's ``prev_sig``.
    This is what makes corruption cascade: once one entry's recomputation
    diverges, every entry after it was chained against a ``prev_sig`` that no
    longer matches. Hardened to support head anchor for truncation detection.
    """

    def __init__(self, secret: bytes):
        self._secret = secret

    def invalid_indices(self, export_records: List[Dict[str, Any]]) -> List[int]:
        """Return the indices whose recomputed signature does not match."""
        invalid: List[int] = []
        prev_sig = ChainedAuditStore.GENESIS
        for i, record in enumerate(export_records):
            payload = {
                "actor": record["actor"],
                "action": record["action"],
                "resource": record["resource"],
                "decision": record["decision"],
                "ts": record.get("ts"),
                "meta": record.get("meta", {}),
                "prev_sig": prev_sig,
            }
            message = json.dumps(payload, sort_keys=True).encode()
            expected = hmac.new(self._secret, message, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, record["signature"]):
                invalid.append(i)
            prev_sig = expected
        return invalid

    def verify_export(self, export_records: List[Dict[str, Any]],
                      head: Optional[Dict[str, Any]] = None) -> bool:
        """Verify the export chain, optionally against a head anchor.

        If head is provided, also checks that:
        - len(export_records) == head["length"]
        - last record's signature == head["signature"]
        - head["anchor"] is a valid HMAC-SHA256 signature over length and last signature

        This enables truncation detection: an attacker removing tail records
        will fail both length and signature checks. The anchor signature prevents
        forged or edited heads.
        """
        # Check for corrupted signatures in the chain
        if len(self.invalid_indices(export_records)) > 0:
            return False

        # If head anchor is provided, verify length, tail signature, and anchor
        if head is not None:
            if len(export_records) != head["length"]:
                return False
            if export_records and export_records[-1]["signature"] != head["signature"]:
                return False
            # Empty chain: verify head signature is GENESIS
            if not export_records and head["signature"] != ChainedAuditStore.GENESIS:
                return False

            # Verify the anchor signature (HMAC-SHA256 over canonical length+signature)
            if "anchor" in head:
                anchor_payload = json.dumps(
                    {"length": head["length"], "signature": head["signature"]},
                    sort_keys=True
                ).encode()
                expected_anchor = hmac.new(self._secret, anchor_payload, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(expected_anchor, head.get("anchor", "")):
                    return False

        return True
