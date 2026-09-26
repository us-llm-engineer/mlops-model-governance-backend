"""F2.3 slim suite: SSE audit tail generator, unit-level (no HTTP).
Contract: the API contract section 3 (svc/audit_stream.tail_events).
Frozen BEFORE mlops.svc.audit_stream exists.

Semantics this suite locks in (the contract leaves the exact "since the last
poll" starting point implicit, so the decision is documented here per the
convention used by the rest of this suite family):

    "Since the last poll" has no prior poll on the very first call, so every
    entry already present in the store when the generator is first iterated
    is delivered as an initial catch-up batch, and only entries appended
    AFTER that first delivery are delivered on subsequent polls (no entry is
    ever delivered twice). This matches a live "tail" that shows you what's
    already there and then follows.

Timing: every test uses a tiny artificial poll_interval_s (0.01-0.02) and
bounded consumption (asyncio.wait_for on __anext__()) with a short per-item
timeout, so a bug that makes the generator stall fails fast with a timeout
error instead of hanging the suite. No real 0.5s+ sleeps are used anywhere.
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore
from mlops.kernel import ValidationFailed
from mlops.svc.audit_stream import tail_events

SECRET = b"f2-3-audit-stream-secret-00"
ITEM_TIMEOUT = 2.0


def make_store():
    return ChainedAuditStore(SECRET)


def append_n(store, n, actor="alice", start=0):
    for i in range(start, start + n):
        store.append(actor, f"action{i}", f"res{i}", "allow", meta={"i": i})


async def _next(agen, timeout=ITEM_TIMEOUT):
    """Bounded consumption: never block indefinitely on a stalled generator."""
    return await asyncio.wait_for(agen.__anext__(), timeout=timeout)


async def _drain(agen, count, timeout=ITEM_TIMEOUT):
    return [await _next(agen, timeout) for _ in range(count)]


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ (1) delivery + shape + ordering

def test_yields_preexisting_batch_matches_store_on_first_poll():
    store = make_store()
    append_n(store, 3)

    async def scenario():
        agen = tail_events(store, poll_interval_s=0.01, limit=100)
        return await _drain(agen, 3)

    items = run(scenario())
    exported = store.export()
    assert [it["data"] for it in items] == exported
    assert all(it["event"] == "audit" for it in items)


def test_item_shape_is_event_and_data_only():
    store = make_store()
    append_n(store, 1)

    async def scenario():
        agen = tail_events(store, poll_interval_s=0.01, limit=10)
        return await _next(agen)

    item = run(scenario())
    assert set(item) == {"event", "data"}
    assert item["event"] == "audit"
    assert item["data"] == store.export()[0]


def test_second_poll_yields_only_new_entries_not_repeats():
    store = make_store()
    append_n(store, 2, actor="alice")

    async def scenario():
        agen = tail_events(store, poll_interval_s=0.01, limit=100)
        first_batch = await _drain(agen, 2)
        append_n(store, 2, actor="bob", start=2)
        second_batch = await _drain(agen, 2)
        return first_batch, second_batch

    first_batch, second_batch = run(scenario())
    assert {it["data"]["actor"] for it in first_batch} == {"alice"}
    assert {it["data"]["actor"] for it in second_batch} == {"bob"}
    # no overlap: nothing from the first batch reappears in the second
    first_ts = {it["data"]["ts"] for it in first_batch}
    second_ts = {it["data"]["ts"] for it in second_batch}
    assert not (first_ts & second_ts)


def test_entries_yielded_in_append_order():
    store = make_store()
    append_n(store, 5)

    async def scenario():
        agen = tail_events(store, poll_interval_s=0.01, limit=100)
        return await _drain(agen, 5)

    items = run(scenario())
    actions = [it["data"]["action"] for it in items]
    assert actions == [f"action{i}" for i in range(5)]


# ------------------------------------------------------------------ (2) limit ceiling

def test_stops_after_small_limit_even_with_more_entries_appended():
    store = make_store()
    append_n(store, 5)

    async def scenario():
        agen = tail_events(store, poll_interval_s=0.01, limit=3)
        collected = []
        async for item in agen:
            collected.append(item)
        return collected

    items = run(asyncio.wait_for(scenario(), timeout=5.0))
    assert len(items) == 3
    actions = [it["data"]["action"] for it in items]
    assert actions == ["action0", "action1", "action2"]


def test_generator_raises_stopasynciteration_once_limit_reached():
    store = make_store()
    append_n(store, 4)

    async def scenario():
        agen = tail_events(store, poll_interval_s=0.01, limit=2)
        await _drain(agen, 2)
        with pytest.raises(StopAsyncIteration):
            await _next(agen, timeout=1.0)

    run(scenario())


def test_limit_one_yields_exactly_one_entry():
    store = make_store()
    append_n(store, 3)

    async def scenario():
        agen = tail_events(store, poll_interval_s=0.01, limit=1)
        item = await _next(agen)
        with pytest.raises(StopAsyncIteration):
            await _next(agen, timeout=1.0)
        return item

    item = run(scenario())
    assert item["data"]["action"] == "action0"


def test_limit_boundary_100000_is_accepted_not_validation_error():
    store = make_store()
    append_n(store, 1)

    async def scenario():
        agen = tail_events(store, poll_interval_s=0.01, limit=100000)
        return await _next(agen)

    item = run(scenario())
    assert item["event"] == "audit"


# ------------------------------------------------------------------ (3) poll_interval_s validation

@pytest.mark.parametrize("bad_interval", [0, 0.0, -1.0, 60.0001, 61, True, False, "0.5"])
def test_poll_interval_s_invalid_raises_validation_failed(bad_interval):
    store = make_store()
    append_n(store, 1)

    async def scenario():
        agen = tail_events(store, poll_interval_s=bad_interval, limit=10)
        await _next(agen, timeout=1.0)

    with pytest.raises(ValidationFailed):
        run(scenario())


def test_poll_interval_s_boundary_60_is_accepted():
    store = make_store()
    append_n(store, 1)

    async def scenario():
        agen = tail_events(store, poll_interval_s=60, limit=10)
        return await _next(agen)

    item = run(scenario())
    assert item["event"] == "audit"


# ------------------------------------------------------------------ (4) limit validation

@pytest.mark.parametrize("bad_limit", [0, -1, 100001, True, False, 1.5, "10"])
def test_limit_invalid_raises_validation_failed(bad_limit):
    store = make_store()
    append_n(store, 1)

    async def scenario():
        agen = tail_events(store, poll_interval_s=0.01, limit=bad_limit)
        await _next(agen, timeout=1.0)

    with pytest.raises(ValidationFailed):
        run(scenario())


# ------------------------------------------------------------------ (5) secret never leaks

def test_secret_never_appears_in_any_yielded_item():
    store = make_store()
    append_n(store, 6)
    secret_str = SECRET.decode()

    async def scenario():
        agen = tail_events(store, poll_interval_s=0.01, limit=100)
        return await _drain(agen, 6)

    items = run(scenario())
    assert len(items) == 6
    for item in items:
        rendered = str(item)
        assert secret_str not in rendered
        assert SECRET.hex() not in rendered
