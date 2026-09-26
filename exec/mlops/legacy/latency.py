"""
End-to-end latency and concurrent-operation surface (Claim C12).

Maps to requirement R1.3 (Platform API & CLI) and is lifted from the frozen
reference implementation in ``tests/R1_3_S4.py``.

Provides latency measurement, a concurrency-safe stub API backend and a
thread-pool testbed proving concurrent CLI operations neither block nor
deadlock. Standard library only.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Operation:
    """Represents a single CLI operation and its measured timing."""

    op_id: str
    operation: str
    start_time: float = None
    end_time: float = None
    success: bool = False
    result: str = ""

    def duration_ms(self) -> float:
        """Return the operation duration in milliseconds (0 if not complete)."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0


class LatencyMonitor:
    """Thread-safe monitor aggregating operation latencies."""

    def __init__(self) -> None:
        self.operations: List[Operation] = []
        self.lock = threading.RLock()

    def record_operation(self, op: Operation) -> None:
        """Record a completed operation under the monitor lock."""
        with self.lock:
            self.operations.append(op)

    def get_p95_latency(self) -> float:
        """Return the p95 duration across all recorded operations."""
        if not self.operations:
            return 0

        durations = sorted([op.duration_ms() for op in self.operations])
        p95_idx = int(len(durations) * 0.95)
        return durations[p95_idx]

    def get_max_latency(self) -> float:
        """Return the maximum recorded duration."""
        if not self.operations:
            return 0
        return max(op.duration_ms() for op in self.operations)

    def success_rate(self) -> float:
        """Return the fraction of recorded operations that succeeded."""
        if not self.operations:
            return 0
        successes = sum(1 for op in self.operations if op.success)
        return successes / len(self.operations)


class ApiBackend:
    """Concurrency-safe stub API backend counting and serving requests."""

    def __init__(self) -> None:
        self.request_count = 0
        self.lock = threading.RLock()

    def promote_model(self, model_id: str, from_env: str, to_env: str) -> Dict:
        """Promote a model, returning a JSON response with its request number."""
        with self.lock:
            self.request_count += 1
            req_num = self.request_count

        time.sleep(0.01)

        return {
            "status": 200,
            "body": json.dumps(
                {
                    "id": model_id,
                    "promoted": True,
                    "from": from_env,
                    "to": to_env,
                    "request_num": req_num,
                }
            ),
        }

    def list_models(self, env: str = None) -> Dict:
        """List models, incrementing the shared request counter."""
        with self.lock:
            self.request_count += 1

        time.sleep(0.01)

        return {
            "status": 200,
            "body": json.dumps({"items": [{"id": "m1", "name": "model1"}]}),
        }


class CliExecutor:
    """Execute CLI operations against an API backend with latency measurement."""

    def __init__(self, api_backend: ApiBackend) -> None:
        self.api = api_backend
        self.latency_monitor = LatencyMonitor()

    def promote_model_cli(
        self, op_id: str, model_id: str, from_env: str, to_env: str
    ) -> Operation:
        """Execute and time a promote command, recording the operation."""
        op = Operation(op_id=op_id, operation="promote")
        op.start_time = time.time()

        try:
            response = self.api.promote_model(model_id, from_env, to_env)

            time.sleep(0.001)

            op.end_time = time.time()
            op.success = response.get("status") == 200
            op.result = response.get("body", "")

        except Exception as e:
            op.end_time = time.time()
            op.success = False
            op.result = str(e)

        self.latency_monitor.record_operation(op)
        return op

    def list_models_cli(self, op_id: str, env: str = None) -> Operation:
        """Execute and time a list command, recording the operation."""
        op = Operation(op_id=op_id, operation="list")
        op.start_time = time.time()

        try:
            response = self.api.list_models(env)

            time.sleep(0.001)

            op.end_time = time.time()
            op.success = response.get("status") == 200
            op.result = response.get("body", "")

        except Exception as e:
            op.end_time = time.time()
            op.success = False
            op.result = str(e)

        self.latency_monitor.record_operation(op)
        return op


class ConcurrencyTestbed:
    """Run concurrent CLI operations through a shared API backend."""

    def __init__(self) -> None:
        self.api = ApiBackend()
        self.executor = CliExecutor(self.api)

    def run_concurrent_operations(
        self, num_operations: int, operation_type: str = "promote"
    ) -> List[Operation]:
        """Run ``num_operations`` concurrently and return the completed list."""
        operations: List[Operation] = []

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = []

            for i in range(num_operations):
                if operation_type == "promote":
                    future = pool.submit(
                        self.executor.promote_model_cli,
                        f"op_{i}",
                        f"model_{i % 5}",
                        "dev",
                        "prod",
                    )
                else:
                    future = pool.submit(
                        self.executor.list_models_cli,
                        f"op_{i}",
                        "prod",
                    )

                futures.append(future)

            for future in as_completed(futures):
                operations.append(future.result())

        return operations
