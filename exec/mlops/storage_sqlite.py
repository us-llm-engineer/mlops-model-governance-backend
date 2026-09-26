"""
S2.2 Durable Storage: SQLite WAL audit chain, docs store, lineage persistence.

Provides:
- open_db(path): Opens a SQLite connection with WAL mode and synchronous=FULL
- DurableAuditStore: Extends ChainedAuditStore with durable SQLite backend
- DurableDocs: Key-value docs store with integrity verification
- save_lineage/load_lineage: Round-trip LineageGraph with tamper detection
"""

import copy
import hashlib
import hmac
import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mlops.audit import ChainedAuditStore, ChainVerifier
from mlops.kernel import Conflict, IntegrityError, NotFound, canonical_json, sha256_hex
from mlops.lineage import LineageGraph, LineageGraphNode

SCHEMA_VERSION = "1.0"


def open_db(path: str) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and synchronous=FULL.

    Idempotently creates schema_version table and ensures WAL mode is active.
    """
    conn = sqlite3.connect(path, check_same_thread=False)

    # Enable WAL mode
    conn.execute("PRAGMA journal_mode = WAL")

    # Set synchronous=FULL for durability
    conn.execute("PRAGMA synchronous = FULL")

    # Set busy timeout to 5 seconds
    conn.execute("PRAGMA busy_timeout = 5000")

    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")

    # Create schema_version table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version TEXT
        )
    """)

    # Idempotently insert schema version (only if not already present)
    existing = conn.execute(
        "SELECT version FROM schema_version WHERE version = ?",
        (SCHEMA_VERSION,)
    ).fetchone()
    if existing is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

    conn.commit()
    return conn


class DurableAuditStore(ChainedAuditStore):
    """Durable audit store backed by SQLite with integrity verification.

    Extends ChainedAuditStore to persist entries and head anchor in a single
    atomic transaction. Detects corruption on load: edited rows, tail truncation,
    or anchor forgery raise IntegrityError.
    """

    def __init__(self, path: str, secret: bytes):
        super().__init__(secret)
        self._path = path
        self._conn = open_db(path)
        self._write_lock = threading.RLock()  # Use RLock for re-entrancy

        # Create audit tables
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_entries (
                seq INTEGER PRIMARY KEY,
                payload TEXT NOT NULL,
                signature TEXT NOT NULL
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_head (
                id INTEGER PRIMARY KEY CHECK(id=1),
                length INTEGER NOT NULL,
                signature TEXT NOT NULL,
                anchor TEXT NOT NULL
            )
        """)

        self._conn.commit()

        # Load and verify all entries
        self._load_and_verify()

    def _load_and_verify(self):
        """Load entries from disk, verify chain integrity, and verify head anchor."""
        # Load all entries
        rows = self._conn.execute(
            "SELECT seq, payload, signature FROM audit_entries ORDER BY seq"
        ).fetchall()

        # Rebuild the in-memory chain from stored entries
        verifier = ChainVerifier(self._secret)

        if rows:
            # Convert rows to entry dicts for verification
            entries_for_verify = []
            for seq, payload, sig in rows:
                try:
                    entry_dict = json.loads(payload)
                except (json.JSONDecodeError, ValueError) as e:
                    raise IntegrityError(f"Payload corruption at index {seq}: {e}")
                entry_dict["signature"] = sig
                entries_for_verify.append(entry_dict)

            # Check for signature corruption
            invalid_indices = verifier.invalid_indices(entries_for_verify)
            if invalid_indices:
                raise IntegrityError(f"Signature mismatch at index {invalid_indices[0]}")

            # Rebuild internal state with verified entries
            self._entries = entries_for_verify

        # Verify head anchor - FAIL-SAFE: non-empty entries requires valid head row
        stored_head = self._conn.execute(
            "SELECT length, signature, anchor FROM audit_head WHERE id=1"
        ).fetchone()

        if rows and not stored_head:
            # Non-empty entries table with no head row = corruption
            raise IntegrityError("audit_head row missing but entries exist")

        if stored_head:
            stored_len, stored_sig, stored_anchor = stored_head

            # Compute expected head
            expected_head = self.head()

            # Verify length matches
            if stored_len != expected_head["length"]:
                raise IntegrityError(
                    f"Head length mismatch: stored {stored_len}, computed {expected_head['length']}"
                )

            # Verify signature matches
            if stored_sig != expected_head["signature"]:
                raise IntegrityError(
                    f"Head signature mismatch at length {stored_len}"
                )

            # Verify anchor is valid
            if stored_anchor != expected_head["anchor"]:
                raise IntegrityError(
                    f"Head anchor mismatch at length {stored_len}"
                )

    @staticmethod
    def _check_entry(actor: Any, action: Any, resource: Any, decision: Any,
                     ts: Any, meta: Any) -> None:
        """Validate audit entry fields before writing.

        Raises TypeError or ValueError for invalid inputs.
        """
        # Validate actor
        if not isinstance(actor, str):
            raise TypeError(f"actor must be str, not {type(actor).__name__}")
        if len(actor) == 0:
            raise ValueError("actor cannot be empty")
        if any(ord(c) < 0x20 or ord(c) == 0x7f for c in actor):
            raise ValueError("actor cannot contain control characters")

        # Validate action
        if not isinstance(action, str):
            raise TypeError(f"action must be str, not {type(action).__name__}")
        if len(action) == 0:
            raise ValueError("action cannot be empty")
        if any(ord(c) < 0x20 or ord(c) == 0x7f for c in action):
            raise ValueError("action cannot contain control characters")

        # Validate resource
        if not isinstance(resource, str):
            raise TypeError(f"resource must be str, not {type(resource).__name__}")
        if len(resource) == 0:
            raise ValueError("resource cannot be empty")
        if any(ord(c) < 0x20 or ord(c) == 0x7f for c in resource):
            raise ValueError("resource cannot contain control characters")

        # Validate decision
        if not isinstance(decision, str):
            raise TypeError(f"decision must be str, not {type(decision).__name__}")
        if len(decision) == 0:
            raise ValueError("decision cannot be empty")
        if any(ord(c) < 0x20 or ord(c) == 0x7f for c in decision):
            raise ValueError("decision cannot contain control characters")

        # Validate ts (must be None or finite real number, not bool)
        if ts is not None:
            if isinstance(ts, bool):
                raise TypeError("ts must be a number, not bool")
            if not isinstance(ts, (int, float)):
                raise TypeError(f"ts must be None or a number, not {type(ts).__name__}")
            if isinstance(ts, float):
                if ts != ts or ts == float('inf') or ts == float('-inf'):
                    raise ValueError("ts must be a finite number")

        # Validate meta (must be dict with str keys or None)
        if meta is not None:
            if not isinstance(meta, dict):
                raise TypeError(f"meta must be a dict or None, not {type(meta).__name__}")
            for k, v in meta.items():
                if not isinstance(k, str):
                    raise TypeError(f"meta keys must be str, not {type(k).__name__}")
                # Reject NaN/inf in meta values
                if isinstance(v, float):
                    if v != v or v == float('inf') or v == float('-inf'):
                        raise ValueError("meta values cannot contain NaN or Inf")
                # Reject sets in meta values
                if isinstance(v, set):
                    raise TypeError("meta values cannot be sets")

    def append(
        self,
        actor: str,
        action: str,
        resource: str,
        decision: str,
        ts: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append entry to the chain and durably persist it.

        Writes both the audit row and head anchor in one atomic transaction
        before returning.

        Raises TypeError or ValueError for invalid inputs before any write.
        Raises Conflict or IntegrityError if another handle modified the DB.
        """
        # Validate all inputs before acquiring lock or writing
        self._check_entry(actor, action, resource, decision, ts, meta)

        with self._write_lock:
            # Re-read head from DB to detect stale handle
            stored_head = self._conn.execute(
                "SELECT length, signature FROM audit_head WHERE id=1"
            ).fetchone()

            db_length = stored_head[0] if stored_head else 0
            mem_length = len(self._entries)

            # If DB is ahead of memory, refresh from disk
            if db_length > mem_length:
                self._load_and_verify()

            # If DB and memory diverge (shouldn't happen after refresh), raise error
            if db_length != len(self._entries):
                raise Conflict("Database state diverged from memory after refresh")

            # Build the entry locally first
            if ts is None:
                ts = datetime.now(timezone.utc).timestamp()
            if meta is None:
                meta = {}

            meta_copy = copy.deepcopy(meta)

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
            local_entry = dict(payload, signature=signature)

            # Persist to disk in one transaction
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # Insert the new entry
                payload_json = json.dumps(local_entry, sort_keys=True)
                self._conn.execute(
                    "INSERT INTO audit_entries (seq, payload, signature) VALUES (?, ?, ?)",
                    (len(self._entries), payload_json, local_entry["signature"])
                )

                # Update head anchor
                new_head = {
                    "length": len(self._entries) + 1,
                    "signature": local_entry["signature"],
                }
                anchor_payload = json.dumps(new_head, sort_keys=True).encode()
                anchor = hmac.new(self._secret, anchor_payload, hashlib.sha256).hexdigest()

                self._conn.execute(
                    "INSERT OR REPLACE INTO audit_head (id, length, signature, anchor) VALUES (1, ?, ?, ?)",
                    (new_head["length"], new_head["signature"], anchor)
                )

                self._conn.commit()

                # Only update in-memory state after successful commit
                self._entries.append(local_entry)
                return local_entry

            except sqlite3.IntegrityError as e:
                self._conn.rollback()
                if "UNIQUE constraint failed" in str(e):
                    # Another handle wrote to the same seq; refresh and retry/raise
                    self._load_and_verify()
                    raise Conflict("Another handle appended concurrently")
                raise IntegrityError(f"Database error: {e}")
            except Exception:
                self._conn.rollback()
                raise

    def export(self) -> List[Dict[str, Any]]:
        """Return deep copies of the chained entries, in append order.

        Deep-copies to prevent consumer mutations from affecting the chain.
        """
        with self._write_lock:
            return [copy.deepcopy(e) for e in self._entries]

    def head(self) -> Dict[str, Any]:
        """Return the chain head anchor: length, last signature, and signed anchor.

        The anchor is an HMAC-SHA256 over a canonical string of length and
        last signature, preventing forged or edited heads.
        """
        with self._write_lock:
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

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()


class DurableDocs:
    """Key-value document store with canonical JSON and SHA256 integrity.

    Validates table names and verifies content hashes on read.
    """

    # Strict regex for table names: alphanumeric and underscore, max 64 chars
    _TABLE_NAME_REGEX = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]{0,63}')

    @staticmethod
    def _check_key(key: Any) -> None:
        """Validate a key: must be str, non-empty, <=256 chars, no control chars."""
        if not isinstance(key, str):
            raise TypeError(f"Key must be str, not {type(key).__name__}")
        if len(key) == 0:
            raise ValueError("Key cannot be empty")
        if len(key) > 256:
            raise ValueError(f"Key cannot exceed 256 characters (got {len(key)})")
        # Reject any control character (0x00-0x1F) or DEL (0x7F)
        if any(ord(c) < 0x20 or ord(c) == 0x7f for c in key):
            raise ValueError("Key cannot contain NUL or control characters")

    @staticmethod
    def _check_prefix(prefix: Any) -> None:
        """Validate a prefix: must be str, <=256 chars, no control chars."""
        if not isinstance(prefix, str):
            raise TypeError(f"Prefix must be str, not {type(prefix).__name__}")
        if len(prefix) > 256:
            raise ValueError(f"Prefix cannot exceed 256 characters (got {len(prefix)})")
        # Reject any control character (0x00-0x1F) or DEL (0x7F)
        if any(ord(c) < 0x20 or ord(c) == 0x7f for c in prefix):
            raise ValueError("Prefix cannot contain NUL or control characters")

    def __init__(self, path: str, table: str):
        """Initialize docs store with validated table name."""
        if not self._TABLE_NAME_REGEX.fullmatch(table):
            raise ValueError(f"Invalid table name: {table}")

        self._path = path
        self._table = table
        self._lock = threading.Lock()

        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA foreign_keys = ON")

        # Create docs table if needed
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{self._table}" (
                key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                hash TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _put_unlocked(self, key: str, obj: Any) -> None:
        """Internal put without lock (caller must hold lock or handle transaction)."""
        self._check_key(key)
        content_json = canonical_json(obj)
        content_hash = sha256_hex(content_json)
        self._conn.execute(
            f"INSERT OR REPLACE INTO \"{self._table}\" (key, content, hash) VALUES (?, ?, ?)",
            (key, content_json, content_hash)
        )

    def put(self, key: str, obj: Any, autocommit: bool = True) -> None:
        """Store an object with canonical JSON and hash verification.

        Raises TypeError if obj contains non-canonical types.
        Raises ValueError/TypeError for invalid keys.

        If autocommit=False, caller must commit the transaction.
        """
        with self._lock:
            self._put_unlocked(key, obj)
            if autocommit:
                self._conn.commit()

    def put_many(self, items: Dict[str, Any]) -> None:
        """Store multiple objects atomically in one transaction.

        Raises on any error before any writes are committed.
        """
        # Don't use nested locks - just call put without lock re-entrancy issues
        self._lock.acquire()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for key, obj in items.items():
                    # Validate the key
                    self._check_key(key)
                    # Call put through the proper validation path but with autocommit=False
                    # We have the lock so we can call directly
                    content_json = canonical_json(obj)
                    content_hash = sha256_hex(content_json)
                    self._conn.execute(
                        f"INSERT OR REPLACE INTO \"{self._table}\" (key, content, hash) VALUES (?, ?, ?)",
                        (key, content_json, content_hash)
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        finally:
            self._lock.release()


    def get(self, key: str) -> Any:
        """Retrieve an object by key, verifying its hash.

        Raises IntegrityError if hash doesn't match.
        Raises NotFound (or returns None) if key not found.
        Raises ValueError/TypeError for invalid keys.
        """
        self._check_key(key)
        with self._lock:
            row = self._conn.execute(
                f"SELECT content, hash FROM \"{self._table}\" WHERE key = ?",
                (key,)
            ).fetchone()

            if row is None:
                # Contract allows returning None or raising NotFound
                raise NotFound(f"Key not found: {key}")

            content_json, stored_hash = row
            computed_hash = sha256_hex(content_json)

            if computed_hash != stored_hash:
                raise IntegrityError(f"Hash mismatch for key {key}")

            return json.loads(content_json)

    def keys(self, prefix: str = "") -> List[str]:
        """Return all keys matching the prefix (case-sensitive, literal).

        Uses substring matching, not SQL LIKE wildcards.
        """
        self._check_prefix(prefix)
        with self._lock:
            if prefix:
                rows = self._conn.execute(
                    f"SELECT key FROM \"{self._table}\" WHERE substr(key, 1, ?) = ?",
                    (len(prefix), prefix)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT key FROM \"{self._table}\""
                ).fetchall()

            return [r[0] for r in rows]

    def delete(self, key: str) -> None:
        """Delete a key from the store.

        Raises ValueError/TypeError for invalid keys.
        """
        self._check_key(key)
        with self._lock:
            self._conn.execute(
                f"DELETE FROM \"{self._table}\" WHERE key = ?",
                (key,)
            )
            self._conn.commit()


def save_lineage(docs: DurableDocs, graph: LineageGraph) -> None:
    """Persist a LineageGraph atomically.

    Uses put_many for atomic transaction: if any write fails, all are rolled back
    ensuring the database remains in a consistent state.
    """
    # Build all items to save
    items = {}

    # Save nodes
    for node_id, node in graph.nodes.items():
        node_data = {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "content_hash": node.content_hash,
            "timestamp": node.timestamp,
            "properties": node.properties,
        }
        items[f"lineage:node:{node_id}"] = node_data

    # Save edges and immutable_content
    items["lineage:edges"] = graph.edges
    items["lineage:immutable_content"] = graph.immutable_content

    # Write all atomically using put_many which handles the transaction
    docs.put_many(items)


def load_lineage(docs: DurableDocs) -> LineageGraph:
    """Reconstruct a LineageGraph from DurableDocs with tamper detection."""
    graph = LineageGraph()

    # Load immutable_content with error handling
    try:
        immutable_content = docs.get("lineage:immutable_content")
        graph.immutable_content = immutable_content
    except NotFound:
        graph.immutable_content = {}

    # Load edges with error handling
    try:
        edges = docs.get("lineage:edges")
    except NotFound:
        edges = []

    # Find all nodes by scanning keys
    node_keys = [k for k in docs.keys("lineage:node:")]

    # Load each node and verify its content hash
    for key in node_keys:
        node_data = docs.get(key)
        node_id = node_data["node_id"]

        # Verify content hash by recomputing from stored fields
        content = json.dumps(
            {
                "type": node_data["node_type"],
                "properties": node_data["properties"],
                "timestamp": node_data["timestamp"],
            },
            sort_keys=True
        )
        computed_hash = sha256_hex(content)

        if computed_hash != node_data["content_hash"]:
            raise IntegrityError(
                f"Node {node_id} content hash mismatch: expected {node_data['content_hash']}, got {computed_hash}"
            )

        # Also verify against immutable_content if available
        if node_id in graph.immutable_content:
            if content != graph.immutable_content[node_id]:
                raise IntegrityError(
                    f"Node {node_id} content diverges from immutable_content"
                )

        # Reconstruct node
        node = LineageGraphNode(
            node_id=node_data["node_id"],
            node_type=node_data["node_type"],
            content_hash=node_data["content_hash"],
            timestamp=node_data["timestamp"],
            properties=node_data["properties"],
        )
        graph.nodes[node_id] = node

    # Verify all immutable_content nodes were loaded
    for node_id in graph.immutable_content.keys():
        if node_id not in graph.nodes:
            raise IntegrityError(f"Node {node_id} in immutable_content but not found in DB")

    # Restore edges
    graph.edges = edges

    return graph
