"""Audit stream generator for SSE tail endpoint.

Contract: the API contract section 3 (svc/audit_stream.tail_events).
"""

import asyncio
from typing import AsyncIterator, Dict, Any

from mlops.kernel import ValidationFailed


async def tail_events(
    audit,
    *,
    poll_interval_s: float = 0.5,
    limit: int = 1000
) -> AsyncIterator[Dict[str, Any]]:
    """Poll the audit store and yield new entries as SSE events.

    Args:
        audit: ChainedAuditStore instance
        poll_interval_s: float in (0, 60], how long to sleep between polls
        limit: int in [1, 100000], max entries to yield before stopping

    Yields:
        {"event": "audit", "data": <entry dict>}

    First poll delivers everything in the store; later polls deliver only
    entries appended since the last delivery (no entry is ever yielded twice).
    Never yields the secret or signing key (the store's export() already
    excludes it; this function does not add it back).
    """
    # Validate poll_interval_s: must be numeric in (0, 60], not bool
    # bool is a subclass of int in Python, so check it first
    if isinstance(poll_interval_s, bool) or not isinstance(poll_interval_s, (int, float)):
        raise ValidationFailed(f"poll_interval_s must be a float in (0, 60]")
    if poll_interval_s <= 0 or poll_interval_s > 60:
        raise ValidationFailed(f"poll_interval_s must be a float in (0, 60]")

    # Validate limit: must be int (not bool, not float) in [1, 100000]
    # bool is a subclass of int in Python, so check it first
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValidationFailed(f"limit must be an int in [1, 100000]")
    if limit < 1 or limit > 100000:
        raise ValidationFailed(f"limit must be an int in [1, 100000]")

    yielded = 0
    seen_idx = 0

    head = getattr(audit, "head", None)

    while yielded < limit:
        # Cheap growth check first: export() deep-copies EVERY entry under the store's lock, so
        # calling it every poll for every connection makes an idle tail O(n) CPU and blocks
        # appenders. head()["length"] is O(1); only export when something new exists.
        if callable(head) and head()["length"] == seen_idx:
            await asyncio.sleep(poll_interval_s)
            continue

        # Poll the store for current state
        current = audit.export()

        # Yield all new entries up to the limit
        while seen_idx < len(current) and yielded < limit:
            entry = current[seen_idx]
            yield {"event": "audit", "data": entry}
            seen_idx += 1
            yielded += 1

        # If we've hit the limit, stop
        if yielded >= limit:
            break

        # Otherwise sleep and poll again
        await asyncio.sleep(poll_interval_s)
