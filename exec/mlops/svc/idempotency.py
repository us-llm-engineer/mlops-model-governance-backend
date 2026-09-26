"""Idempotency store for handling retries and duplicate requests."""

import copy
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from mlops.kernel import ValidationFailed


@dataclass
class IdempotencyEntry:
    """Stores a single idempotency entry."""
    request: Tuple[str, str, str]  # (method, path, body_hash)
    status: Optional[int] = None  # None while in-flight
    body: Optional[Any] = None
    expires_at: float = 0.0


class IdempotencyStore:
    """
    Thread-safe idempotency store for handling duplicate requests.

    Keys are scoped per principal name. Entries have a TTL and the store
    is bounded by max_entries, evicting oldest entries when full.
    """

    KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
    VALID_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}

    def __init__(self, clock, ttl_s: int, max_entries: int = 10_000):
        """
        Initialize the idempotency store.

        Args:
            clock: Clock object with now() method for tracking timestamps
            ttl_s: Time-to-live in seconds for cached entries
            max_entries: Maximum number of entries to store (oldest evicted when full)
        """
        self.clock = clock
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._lock = threading.Lock()
        # OrderedDict maintains insertion order for efficient front-based eviction
        self._store = OrderedDict()

    def _validate_key(self, key: Any) -> str:
        """Validate and normalize the idempotency key."""
        if not isinstance(key, str):
            raise ValidationFailed("Idempotency key must be a string")
        if not self.KEY_PATTERN.fullmatch(key):
            raise ValidationFailed("Idempotency key must be 8-128 chars matching [A-Za-z0-9_-]")
        return key

    def _validate_principal(self, principal: Any) -> str:
        """Validate the principal name."""
        if not isinstance(principal, str):
            raise ValidationFailed("Principal must be a non-empty string")
        if not principal or any(ord(c) < 32 for c in principal):
            raise ValidationFailed("Principal must be non-empty without control characters")
        return principal

    def _validate_method(self, method: Any) -> str:
        """Validate the HTTP method."""
        if not isinstance(method, str) or method not in self.VALID_METHODS:
            raise ValidationFailed("Method must be one of GET, POST, PUT, PATCH, DELETE")
        return method

    def _validate_path(self, path: Any) -> str:
        """Validate the request path."""
        if not isinstance(path, str) or not path.startswith("/") or len(path) > 1024:
            raise ValidationFailed("Path must start with / and be at most 1024 chars")
        return path

    def _validate_body_hash(self, body_hash: Any) -> str:
        """Validate the body hash."""
        if not isinstance(body_hash, str) or not body_hash or len(body_hash) > 128:
            raise ValidationFailed("Body hash must be non-empty string of at most 128 chars")
        return body_hash

    def _cleanup_expired(self, current_time: float) -> None:
        """Remove expired entries from the front (called with lock held)."""
        while self._store:
            first_entry = next(iter(self._store.values()))
            if first_entry.expires_at <= current_time:
                self._store.popitem(last=False)
            else:
                break

    def begin(
        self,
        key: Any,
        principal: Any,
        method: Any,
        path: Any,
        body_hash: Any,
    ) -> Tuple[str, Optional[Tuple[int, Any]]]:
        """
        Start processing a request with idempotency.

        Args:
            key: Idempotency key (validated)
            principal: Principal name (for scoping)
            method: HTTP method
            path: Request path
            body_hash: Hash of request body

        Returns:
            Tuple of:
            - status: "new", "replay", "conflict", or "in_flight"
            - cached: (status_code, body) for "replay", None otherwise
        """
        # Validate all inputs BEFORE any state change
        key = self._validate_key(key)
        principal = self._validate_principal(principal)
        method = self._validate_method(method)
        path = self._validate_path(path)
        body_hash = self._validate_body_hash(body_hash)

        request_id = (principal, key)
        current_time = self.clock.now()

        with self._lock:
            # Clean expired entries from the front (lazy, amortised O(1))
            self._cleanup_expired(current_time)

            if request_id in self._store:
                entry = self._store[request_id]
                # Check if the request matches
                if entry.request == (method, path, body_hash):
                    # Same request
                    if entry.status is None:
                        # In-flight
                        return ("in_flight", None)
                    else:
                        # Finished - return replay with a copy
                        cached = (entry.status, copy.deepcopy(entry.body))
                        return ("replay", cached)
                else:
                    # Different request with same key
                    return ("conflict", None)
            else:
                # New request - create entry
                expires_at = current_time + self.ttl_s
                entry = IdempotencyEntry(
                    request=(method, path, body_hash),
                    status=None,  # Mark as in-flight
                    body=None,
                    expires_at=expires_at,
                )
                self._store[request_id] = entry

                # Evict oldest entries if we exceed max_entries (from the front)
                while len(self._store) > self.max_entries:
                    self._store.popitem(last=False)

                return ("new", None)

    def finish(
        self,
        key: Any,
        principal: Any,
        method: Any,
        path: Any,
        body_hash: Any,
        status: int,
        body: Any,
    ) -> None:
        """
        Mark a request as finished and store the result.

        Args:
            key: Idempotency key
            principal: Principal name
            method: HTTP method
            path: Request path
            body_hash: Hash of request body
            status: HTTP status code
            body: Response body
        """
        # Validate all inputs BEFORE any state change
        key = self._validate_key(key)
        principal = self._validate_principal(principal)
        method = self._validate_method(method)
        path = self._validate_path(path)
        body_hash = self._validate_body_hash(body_hash)

        request_id = (principal, key)

        with self._lock:
            if request_id in self._store:
                entry = self._store[request_id]
                entry.status = status
                entry.body = copy.deepcopy(body)
                # Update expiry time but don't re-order in OrderedDict
                entry.expires_at = self.clock.now() + self.ttl_s

    def abort(
        self,
        key: Any,
        principal: Any,
        method: Any,
        path: Any,
        body_hash: Any,
    ) -> None:
        """
        Abort an in-flight request, clearing it from the store.

        Args:
            key: Idempotency key
            principal: Principal name
            method: HTTP method
            path: Request path
            body_hash: Hash of request body
        """
        # Validate all inputs BEFORE any state change
        key = self._validate_key(key)
        principal = self._validate_principal(principal)
        method = self._validate_method(method)
        path = self._validate_path(path)
        body_hash = self._validate_body_hash(body_hash)

        request_id = (principal, key)

        with self._lock:
            if request_id in self._store:
                del self._store[request_id]
