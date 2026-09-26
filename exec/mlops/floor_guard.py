"""
S3.0 Runtime floor guard: windowed credit admission, queueing, and stability bounds.

This module provides:
* C85 StabilityError: exception for stability constraint violations
* C86 FloorConfig: frozen configuration for windowed credit allocation
* C87 TenantFloor: per-tenant admission control with window resets
* C88 AssuredFirstDispatcher: discrete-event simulation with priority queueing
       and doom-sound drop semantics

Stdlib only: math, random, heapq, typing.
"""

from __future__ import annotations

import copy
import heapq
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .kernel import MlopsError

__all__ = [
    "StabilityError",
    "FloorConfig",
    "AdmitResult",
    "TenantFloor",
    "Request",
    "Outcome",
    "AssuredFirstDispatcher",
    "check_preconditions",
    "require_guarantee",
    "sojourn_bound",
    "simulate_guard",
    "_real",
    "_count",
]


class StabilityError(MlopsError):
    """Raised when a stability or guarantee constraint cannot be satisfied."""

    def __init__(self, message: str):
        super().__init__(message, "stability")


def _real(val, name: str) -> float:
    """Validate and return a real finite number.

    Raises ValueError for any invalid input (non-number types, NaN, inf).
    """
    if isinstance(val, bool) or val is None or isinstance(val, str):
        raise ValueError(f"{name} must be a number, got {type(val).__name__}")
    if not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(val).__name__}")
    if isinstance(val, float):
        if math.isnan(val):
            raise ValueError(f"{name} must be finite, got NaN")
        if math.isinf(val):
            raise ValueError(f"{name} must be finite, got inf")
    try:
        return float(val)
    except (OverflowError, ValueError) as e:
        raise ValueError(f"{name} is too large: {e}")


def _count(val, name: str, lo: Optional[int] = 0, hi: Optional[int] = None) -> int:
    """Validate and return an integer count.

    Raises ValueError for any invalid input (non-int types, values outside range).
    """
    if isinstance(val, bool):
        raise ValueError(f"{name} must be int, got bool")
    if not isinstance(val, int):
        raise ValueError(f"{name} must be int, got {type(val).__name__}")
    if lo is not None and val < lo:
        raise ValueError(f"{name} must be >= {lo}, got {val}")
    if hi is not None and val > hi:
        raise ValueError(f"{name} must be <= {hi}, got {val}")
    return val


@dataclass(frozen=True)
class FloorConfig:
    """Windowed credit allocation configuration (frozen and validated).

    Parameters:
        credits: window credit budget (int >= 1, not bool, <= 10^6)
        window_s: window duration in seconds (float > 0, finite)
        batch_slots: parallel execution slots (int >= 1, <= 10^6)
        s_max: max service time per request (float > 0, finite)
        t0: reference epoch time (float, finite, default 0.0)
        excess: disposition of requests exceeding credits ("reject" or "demote")
        drop_lower_bound: sojourn bound for doom-sound drop (float >= 0, finite)
    """
    credits: int
    window_s: float
    batch_slots: int
    s_max: float
    t0: float = 0.0
    excess: str = "reject"
    drop_lower_bound: float = 0.0

    def __post_init__(self):
        # Validate credits: must be int >= 1, not bool, <= 10^6
        credits = _count(self.credits, "credits", lo=1, hi=10**6)

        # Validate window_s: must be float > 0, finite, and not too small
        window_s = _real(self.window_s, "window_s")
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        if window_s < 1e-100:
            raise ValueError(f"window_s must be >= 1e-100, got {window_s}")

        # Validate batch_slots: must be int >= 1, not bool, <= 10^6
        batch_slots = _count(self.batch_slots, "batch_slots", lo=1, hi=10**6)

        # Validate s_max: must be float > 0, finite, and not too large
        s_max = _real(self.s_max, "s_max")
        if s_max <= 0:
            raise ValueError(f"s_max must be > 0, got {s_max}")
        if s_max > 1e100:
            raise ValueError(f"s_max must be <= 1e100, got {s_max}")

        # Validate t0: must be finite and not too large
        t0 = _real(self.t0, "t0")
        if abs(t0) > 1e100:
            raise ValueError(f"t0 must be in [-1e100, 1e100], got {t0}")

        # Validate excess: must be "reject" or "demote"
        if self.excess not in ("reject", "demote"):
            raise ValueError(f"excess must be 'reject' or 'demote', got {self.excess}")

        # Validate drop_lower_bound: must be finite and >= 0
        drop_lower_bound = _real(self.drop_lower_bound, "drop_lower_bound")
        if drop_lower_bound < 0:
            raise ValueError(f"drop_lower_bound must be >= 0, got {drop_lower_bound}")


@dataclass
class AdmitResult:
    """Result of a floor admission attempt."""
    admitted: int
    demoted: int
    rejected: int
    window_index: int


class TenantFloor:
    """Per-tenant windowed credit floor with refill on window boundary."""

    def __init__(self, cfg: FloorConfig):
        self.cfg = cfg
        self.remaining_credits = cfg.credits
        self.last_window_index: Optional[int] = None

    def admit(self, now_ts: float, n: int) -> AdmitResult:
        """Admit up to n requests at timestamp now_ts.

        Returns AdmitResult with admitted, demoted, rejected counts and window_index.
        Raises ValueError/TypeError if now_ts or n is invalid.
        """
        # Validate now_ts
        now_ts = _real(now_ts, "now_ts")

        # Validate n: must be int >= 0, not bool
        n = _count(n, "n", lo=0)

        # Check for timestamp overflow before calculating window index
        ts_delta = abs(now_ts - self.cfg.t0)
        if ts_delta / self.cfg.window_s > 1e12:
            raise ValueError(f"timestamp {now_ts} is too far from t0={self.cfg.t0}")

        # Calculate window index
        window_index = int(math.floor((now_ts - self.cfg.t0) / self.cfg.window_s))

        # Check time backwards
        if self.last_window_index is not None and window_index < self.last_window_index:
            raise ValueError("time went backwards")

        # Refill if entering a new window
        if self.last_window_index is None or window_index > self.last_window_index:
            self.remaining_credits = self.cfg.credits
            self.last_window_index = window_index
        else:
            # Same window, update last seen if not yet set
            if self.last_window_index is None:
                self.last_window_index = window_index

        # Admit up to min(n, remaining)
        admitted = min(n, self.remaining_credits)
        self.remaining_credits -= admitted

        # Handle excess
        excess = n - admitted
        if self.cfg.excess == "reject":
            rejected = excess
            demoted = 0
        else:  # demote
            rejected = 0
            demoted = excess

        return AdmitResult(
            admitted=admitted,
            demoted=demoted,
            rejected=rejected,
            window_index=window_index,
        )


@dataclass
class Request:
    """A request with class, arrival time, service duration, and deadline."""
    id: str
    cls: str  # "assured" or "opportunistic"
    arrival_ts: float
    service_s: float
    deadline_ts: float


@dataclass
class Outcome:
    """Result of a dispatched request."""
    id: str
    cls: str
    start_ts: Optional[float]
    finish_ts: Optional[float]
    status: str  # "completed" or "dropped"
    missed: bool


class AssuredFirstDispatcher:
    """Discrete-event simulator with priority queue.

    Executes requests with C parallel slots, prioritizing assured class over
    opportunistic, FIFO by arrival within class. Implements doom-sound drop:
    a request is dropped if (dispatch_time + drop_lower_bound > deadline).
    """

    def __init__(self, cfg: FloorConfig):
        self.cfg = cfg

    def run(self, requests: List[Request]) -> List[Outcome]:
        """Simulate processing of requests with AssuredFirst priority.

        Validates all requests before simulation. Does not mutate inputs.
        Returns list of Outcome objects with start_ts, finish_ts, status, and missed.
        """
        # Validate and copy all requests (do not mutate inputs)
        validated_requests = []
        seen_ids: Set[str] = set()

        for i, req in enumerate(requests):
            # Validate id: non-empty str, unique
            if not isinstance(req.id, str) or not req.id:
                raise ValueError("Request id must be non-empty str")
            if req.id in seen_ids:
                raise ValueError(f"Duplicate request id: {req.id}")
            seen_ids.add(req.id)

            # Validate cls: must be in {"assured", "opportunistic"}
            if req.cls not in ("assured", "opportunistic"):
                raise ValueError(f"Request cls must be 'assured' or 'opportunistic', got {req.cls}")

            # Validate arrival_ts: real finite number
            arrival_ts = _real(req.arrival_ts, f"Request {req.id} arrival_ts")

            # Validate service_s: real finite number > 0
            service_s = _real(req.service_s, f"Request {req.id} service_s")
            if service_s <= 0:
                raise ValueError(f"Request {req.id} service_s must be > 0, got {service_s}")

            # Validate deadline_ts: real finite number
            deadline_ts = _real(req.deadline_ts, f"Request {req.id} deadline_ts")

            # Create a validated copy
            validated_req = Request(
                id=req.id,
                cls=req.cls,
                arrival_ts=arrival_ts,
                service_s=service_s,
                deadline_ts=deadline_ts,
            )
            validated_requests.append((i, validated_req))  # Track original index for uniqueness check

        # Extract assured and opportunistic requests
        assured_reqs = [req for idx, req in validated_requests if req.cls == "assured"]
        opportunistic_reqs = [req for idx, req in validated_requests if req.cls == "opportunistic"]

        # Sort by arrival_ts within each class to maintain FIFO
        assured_reqs.sort(key=lambda r: r.arrival_ts)
        opportunistic_reqs.sort(key=lambda r: r.arrival_ts)

        # Track outcomes
        outcomes_dict: Dict[str, Outcome] = {}

        # Track when each slot becomes free
        slot_free_at = [0.0] * self.cfg.batch_slots

        # Track which requests have been scheduled
        scheduled = set()

        # Build heaps for O(n log n) request selection by arrival time
        assured_heap = [(req.arrival_ts, i, req) for i, req in enumerate(assured_reqs)]
        opportunistic_heap = [(req.arrival_ts, i, req) for i, req in enumerate(opportunistic_reqs)]
        heapq.heapify(assured_heap)
        heapq.heapify(opportunistic_heap)

        # Indices into the heaps (requests already popped)
        assured_idx = 0
        opportunistic_idx = 0

        # Event loop: process each request in priority order
        while len(scheduled) < len(validated_requests):
            # Find the next slot that becomes free
            min_time = min(slot_free_at)
            slot_idx = slot_free_at.index(min_time)

            # Find the first unscheduled request that has arrived, prioritizing assured
            selected_req = None

            # Peek at assured heap to find next arrived request
            while assured_idx < len(assured_heap):
                _, _, req = assured_heap[assured_idx]
                if req.id not in scheduled:
                    if req.arrival_ts <= min_time:
                        selected_req = req
                        assured_idx += 1
                        break
                    else:
                        # This and all following assured requests haven't arrived yet
                        break
                assured_idx += 1

            # If no arrived assured, peek at opportunistic heap
            if selected_req is None:
                while opportunistic_idx < len(opportunistic_heap):
                    _, _, req = opportunistic_heap[opportunistic_idx]
                    if req.id not in scheduled:
                        if req.arrival_ts <= min_time:
                            selected_req = req
                            opportunistic_idx += 1
                            break
                        else:
                            # This and all following opportunistic requests haven't arrived yet
                            break
                    opportunistic_idx += 1

            # If still no arrived request, wait for next arrival
            if selected_req is None:
                next_assured_arrival = float('inf')
                for j in range(assured_idx, len(assured_heap)):
                    _, _, req = assured_heap[j]
                    if req.id not in scheduled:
                        next_assured_arrival = req.arrival_ts
                        break

                next_opportunistic_arrival = float('inf')
                for j in range(opportunistic_idx, len(opportunistic_heap)):
                    _, _, req = opportunistic_heap[j]
                    if req.id not in scheduled:
                        next_opportunistic_arrival = req.arrival_ts
                        break

                next_arrival = min(next_assured_arrival, next_opportunistic_arrival)

                if next_arrival == float('inf'):
                    break  # No more requests

                # Update slot time to next arrival
                slot_free_at[slot_idx] = next_arrival
                continue

            scheduled.add(selected_req.id)
            free_time = slot_free_at[slot_idx]
            dispatch_time = max(free_time, selected_req.arrival_ts)

            # Check doom-sound drop
            if dispatch_time + self.cfg.drop_lower_bound > selected_req.deadline_ts:
                # Dropped
                outcomes_dict[selected_req.id] = Outcome(
                    id=selected_req.id,
                    cls=selected_req.cls,
                    start_ts=None,
                    finish_ts=None,
                    status="dropped",
                    missed=True,
                )
            else:
                # Run
                start_ts = dispatch_time
                finish_ts = start_ts + selected_req.service_s
                missed = finish_ts > selected_req.deadline_ts
                outcomes_dict[selected_req.id] = Outcome(
                    id=selected_req.id,
                    cls=selected_req.cls,
                    start_ts=start_ts,
                    finish_ts=finish_ts,
                    status="completed",
                    missed=missed,
                )
                # Release slot at finish time
                slot_free_at[slot_idx] = finish_ts

        return list(outcomes_dict.values())


def check_preconditions(cfg: FloorConfig, assured_backlog: int) -> dict:
    """Check stability preconditions.

    Returns dict with:
        rho_a: admission load ratio B*s_max/(C*W)
        stable: rho_a <= 1
        backlog_ok: assured_backlog <= credits
        guarantee_void: not(stable and backlog_ok)
    """
    # Validate backlog
    assured_backlog = _count(assured_backlog, "assured_backlog", lo=0)

    B = cfg.credits
    s_max = cfg.s_max
    C = cfg.batch_slots
    W = cfg.window_s

    rho_a = (B * s_max) / (C * W)
    stable = rho_a <= 1
    backlog_ok = assured_backlog <= B
    guarantee_void = not (stable and backlog_ok)

    return {
        "rho_a": rho_a,
        "stable": stable,
        "backlog_ok": backlog_ok,
        "guarantee_void": guarantee_void,
    }


def require_guarantee(cfg: FloorConfig, assured_backlog: int) -> None:
    """Raise StabilityError if preconditions are not satisfied."""
    precond = check_preconditions(cfg, assured_backlog)
    if precond["guarantee_void"]:
        raise StabilityError("Guarantee void: system not stable or backlog exceeds budget")


def sojourn_bound(cfg: FloorConfig, assured_backlog: int = 0) -> Optional[float]:
    """Compute assured sojourn bound if preconditions hold, else None.

    Formula: s_max * (1 + ceil(B / C)) if stable and backlog_ok, else None.
    """
    precond = check_preconditions(cfg, assured_backlog)
    if precond["guarantee_void"]:
        return None
    return cfg.s_max * (1 + math.ceil(cfg.credits / cfg.batch_slots))


def simulate_guard(
    seed: int,
    n_windows: int,
    offered_assured: int,
    offered_opportunistic: int,
    cfg: FloorConfig,
    guarded: bool,
) -> dict:
    """Deterministic simulation of guarded vs. unguarded admission.

    With guarded=True: TenantFloor gates assured admissions, AssuredFirstDispatcher
    processes all requests.
    With guarded=False: Single FIFO queue with no floor, no priority.

    Returns dict with assured_miss_rate, opportunistic_miss_rate, and other stats.
    """
    # Validate arguments
    seed = _count(seed, "seed")  # Any int is valid (including negative)
    n_windows = _count(n_windows, "n_windows", lo=1)
    offered_assured = _count(offered_assured, "offered_assured", lo=0)
    offered_opportunistic = _count(offered_opportunistic, "offered_opportunistic", lo=0)

    # Bounds checks for performance
    if n_windows > 10000:
        raise ValueError(f"n_windows must be <= 10000, got {n_windows}")
    if offered_assured > 10000:
        raise ValueError(f"offered_assured must be <= 10000, got {offered_assured}")
    if offered_opportunistic > 10000:
        raise ValueError(f"offered_opportunistic must be <= 10000, got {offered_opportunistic}")

    if not isinstance(guarded, bool):
        raise ValueError(f"guarded must be bool, got {type(guarded).__name__}")

    if not isinstance(cfg, FloorConfig):
        raise ValueError(f"cfg must be FloorConfig, got {type(cfg).__name__}")

    rng = random.Random(seed)

    # Generate requests per window
    all_requests = []
    request_id_counter = 0

    for window_idx in range(n_windows):
        window_start = cfg.t0 + window_idx * cfg.window_s
        window_end = window_start + cfg.window_s

        # Assured requests uniformly distributed in window
        for _ in range(offered_assured):
            arrival_ts = window_start + rng.uniform(0, cfg.window_s)
            deadline_ts = arrival_ts + 1.25 * cfg.s_max * (1 + math.ceil(cfg.credits / cfg.batch_slots))
            all_requests.append(Request(
                id=f"req_{request_id_counter}",
                cls="assured",
                arrival_ts=arrival_ts,
                service_s=cfg.s_max,
                deadline_ts=deadline_ts,
            ))
            request_id_counter += 1

        # Opportunistic requests uniformly distributed in window
        for _ in range(offered_opportunistic):
            arrival_ts = window_start + rng.uniform(0, cfg.window_s)
            deadline_ts = arrival_ts + 1.25 * cfg.s_max * (1 + math.ceil(cfg.credits / cfg.batch_slots))
            all_requests.append(Request(
                id=f"req_{request_id_counter}",
                cls="opportunistic",
                arrival_ts=arrival_ts,
                service_s=cfg.s_max,
                deadline_ts=deadline_ts,
            ))
            request_id_counter += 1

    if guarded:
        # Guarded: use TenantFloor for admission, then AssuredFirstDispatcher
        floor = TenantFloor(cfg)
        admitted_reqs = []
        demoted_reqs = []
        rejected_reqs = []

        # Process each request through the floor
        for req in all_requests:
            if req.cls == "assured":
                result = floor.admit(req.arrival_ts, 1)
                if result.admitted > 0:
                    admitted_reqs.append(req)
                elif result.demoted > 0:
                    # Demote to opportunistic
                    demoted_reqs.append(Request(
                        id=req.id,
                        cls="opportunistic",
                        arrival_ts=req.arrival_ts,
                        service_s=req.service_s,
                        deadline_ts=req.deadline_ts,
                    ))
                else:
                    # Must be rejected
                    rejected_reqs.append(req)

        # Add non-demoted opportunistic requests
        opportunistic_reqs = [r for r in all_requests if r.cls == "opportunistic"]

        # Dispatch: admitted assured + demoted assured (as opportunistic) + original opportunistic
        dispatch_reqs = admitted_reqs + demoted_reqs + opportunistic_reqs
        outcomes = AssuredFirstDispatcher(cfg).run(dispatch_reqs)

        # Count misses
        assured_offered = sum(1 for r in all_requests if r.cls == "assured")
        assured_admitted = len(admitted_reqs)  # Admitted by the floor

        # Misses: rejected + demoted that miss + admitted that actually miss
        # Track demoted IDs to avoid double-counting
        demoted_ids = set(req.id for req in demoted_reqs)
        assured_missed = len(rejected_reqs)  # Rejected count as misses
        assured_missed += sum(1 for req in demoted_reqs if any(o.id == req.id and o.missed for o in outcomes))
        assured_missed += sum(1 for r in all_requests if r.cls == "assured" and r.id not in demoted_ids and
                             any(o.id == r.id and o.status == "completed" and o.missed for o in outcomes))

        opportunistic_offered = sum(1 for r in all_requests if r.cls == "opportunistic")
        orig_opp_ids = {r.id for r in opportunistic_reqs}
        opportunistic_missed = sum(1 for o in outcomes if o.id in orig_opp_ids and o.missed)

        assured_miss_rate = assured_missed / assured_offered if assured_offered > 0 else 0.0
        opportunistic_miss_rate = opportunistic_missed / opportunistic_offered if opportunistic_offered > 0 else 0.0
        assured_rejected = len(rejected_reqs)
        assured_demoted = len(demoted_reqs)

    else:
        # Unguarded: pure FIFO with no floor, no priority
        # Sort all by arrival time
        all_requests_copy = copy.deepcopy(all_requests)
        all_requests_copy.sort(key=lambda r: r.arrival_ts)

        # Track outcomes
        outcomes_dict: Dict[str, Outcome] = {}

        # Track when each slot becomes free
        slot_free_at = [0.0] * cfg.batch_slots

        # Process each request in FIFO order (no priority)
        for req in all_requests_copy:
            # Find the next slot that will be free
            earliest_free_idx = min(range(len(slot_free_at)), key=lambda i: slot_free_at[i])
            free_time = slot_free_at[earliest_free_idx]

            # Dispatch time is max(slot_free_time, arrival_time)
            dispatch_time = max(free_time, req.arrival_ts)

            # Check doom-sound drop (with drop_lower_bound=0 in unguarded)
            if dispatch_time + 0.0 > req.deadline_ts:
                # Dropped
                outcomes_dict[req.id] = Outcome(
                    id=req.id,
                    cls=req.cls,
                    start_ts=None,
                    finish_ts=None,
                    status="dropped",
                    missed=True,
                )
            else:
                # Run
                start_ts = dispatch_time
                finish_ts = start_ts + req.service_s
                missed = finish_ts > req.deadline_ts
                outcomes_dict[req.id] = Outcome(
                    id=req.id,
                    cls=req.cls,
                    start_ts=start_ts,
                    finish_ts=finish_ts,
                    status="completed",
                    missed=missed,
                )
                # Release slot at finish time
                slot_free_at[earliest_free_idx] = finish_ts

        outcomes = list(outcomes_dict.values())

        # Count misses (no priority in unguarded)
        assured_offered = sum(1 for r in all_requests if r.cls == "assured")
        assured_missed = sum(1 for r in outcomes if r.cls == "assured" and r.missed)
        opportunistic_offered = sum(1 for r in all_requests if r.cls == "opportunistic")
        opportunistic_missed = sum(1 for r in outcomes if r.cls == "opportunistic" and r.missed)

        assured_miss_rate = assured_missed / assured_offered if assured_offered > 0 else 0.0
        opportunistic_miss_rate = opportunistic_missed / opportunistic_offered if opportunistic_offered > 0 else 0.0
        # Unguarded has no floor-level rejection/demotion
        assured_rejected = 0
        assured_demoted = 0
        assured_admitted = sum(1 for r in outcomes if r.cls == "assured" and r.status == "completed")

    return {
        "simulated": True,
        "assured_miss_rate": assured_miss_rate,
        "opportunistic_miss_rate": opportunistic_miss_rate,
        "assured_offered": assured_offered,
        "assured_admitted": assured_admitted,
        "assured_rejected": assured_rejected,
        "assured_demoted": assured_demoted,
        "guarded": guarded,
    }
