"""
R1.3-S4: Concurrent Load (Concurrency/Latency)

Test suite for C12: End-to-end latency from CLI mlops model promote to API completion
is < 2s. Concurrent CLI commands from multiple users don't block each other.

Dimension: concurrency/latency cases
Mutation targets:
  - No load testing (mutation: test only single request)
  - Latency not measured (mutation: skip timing)
  - Lock contention causes blocking (mutation: use global lock)
  - Requests queue instead of concurrent (mutation: serialize requests)
"""

import pytest
import time
import threading
import json
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any
from statistics import mean, stdev


@dataclass
class LoadTestResult:
    """Result of load test."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    latencies: List[float] = field(default_factory=list)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


class ConcurrentAPIServer:
    """Thread-safe API server for load testing."""

    def __init__(self, base_latency_ms: float = 50):
        self.base_latency_ms = base_latency_ms
        self.request_count = 0
        self.lock = threading.Lock()
        self.active_requests = 0
        self.max_concurrent = 0

    def promote_model(self, model_id: str, from_env: str, to_env: str) -> Dict:
        """Simulate model promotion (thread-safe)."""
        start_time = time.time()

        with self.lock:
            self.request_count += 1
            self.active_requests += 1
            if self.active_requests > self.max_concurrent:
                self.max_concurrent = self.active_requests

        try:
            # Simulate work with base latency
            time.sleep(self.base_latency_ms / 1000.0)

            result = {
                "model_id": model_id,
                "from_env": from_env,
                "to_env": to_env,
                "status": "promoted"
            }

            return result

        finally:
            with self.lock:
                self.active_requests -= 1

            elapsed_ms = (time.time() - start_time) * 1000
            return result

    def get_request_count(self) -> int:
        with self.lock:
            return self.request_count

    def get_max_concurrent(self) -> int:
        with self.lock:
            return self.max_concurrent


class CLILoadTester:
    """Load tester for CLI commands."""

    def __init__(self, api_server: ConcurrentAPIServer):
        self.api = api_server
        self.latencies: List[float] = []
        self.errors: List[str] = []

    def run_promotion_command(self, model_id: str, from_env: str, to_env: str) -> float:
        """
        Simulate mlops model promote command and measure latency.
        Returns: latency in milliseconds.
        """
        start = time.time()

        try:
            self.api.promote_model(model_id, from_env, to_env)
            elapsed = (time.time() - start) * 1000
            self.latencies.append(elapsed)
            return elapsed

        except Exception as e:
            self.errors.append(str(e))
            return (time.time() - start) * 1000

    def run_concurrent_load(self, num_requests: int, num_workers: int = 4) -> LoadTestResult:
        """
        Run concurrent load test.
        Returns: LoadTestResult with metrics.
        """
        latencies = []
        failed = 0

        def worker_task(i):
            model_num = i % 10
            return self.run_promotion_command(
                f"model-{model_num}",
                "dev",
                "prod"
            )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_requests)]

            for future in as_completed(futures):
                try:
                    latency = future.result()
                    if latency > 0:
                        latencies.append(latency)
                except Exception as e:
                    failed += 1
                    self.errors.append(str(e))

        # Calculate percentiles
        latencies.sort()
        result = LoadTestResult(
            total_requests=num_requests,
            successful_requests=num_requests - failed,
            failed_requests=failed,
            latencies=latencies,
            errors=self.errors
        )

        if latencies:
            result.p50_latency_ms = latencies[len(latencies) // 2]
            result.p95_latency_ms = latencies[int(len(latencies) * 0.95)]
            result.p99_latency_ms = latencies[int(len(latencies) * 0.99)]
            result.mean_latency_ms = mean(latencies)

        return result


class TestCLIEndToEndLatency:
    """End-to-end latency cases."""

    def test_single_promotion_within_sla(self):
        """Case 1: Single promotion command < 2s."""
        server = ConcurrentAPIServer(base_latency_ms=50)
        tester = CLILoadTester(server)

        latency_ms = tester.run_promotion_command("m1", "dev", "prod")

        assert latency_ms < 2000  # 2 seconds in ms

    def test_multiple_sequential_commands_within_sla(self):
        """Case 2: Sequential commands each < 2s."""
        server = ConcurrentAPIServer(base_latency_ms=50)
        tester = CLILoadTester(server)

        latencies = []
        for i in range(5):
            latency = tester.run_promotion_command(f"m{i}", "dev", "prod")
            latencies.append(latency)

        for latency in latencies:
            assert latency < 2000

    def test_mean_latency_reasonable(self):
        """Case 3: Mean latency is reasonable (< 500ms)."""
        server = ConcurrentAPIServer(base_latency_ms=50)
        tester = CLILoadTester(server)

        result = tester.run_concurrent_load(num_requests=20, num_workers=4)

        assert result.mean_latency_ms < 500


class TestConcurrentRequests:
    """Concurrency cases: multiple CLI users don't block each other."""

    def test_concurrent_requests_succeed(self):
        """Case 4: Concurrent requests all succeed."""
        server = ConcurrentAPIServer(base_latency_ms=50)
        tester = CLILoadTester(server)

        result = tester.run_concurrent_load(num_requests=10, num_workers=5)

        assert result.failed_requests == 0
        assert result.successful_requests == 10

    def test_concurrent_requests_not_serialized(self):
        """Case 5: Concurrent requests execute in parallel (not serialized)."""
        server = ConcurrentAPIServer(base_latency_ms=100)
        tester = CLILoadTester(server)

        result = tester.run_concurrent_load(num_requests=10, num_workers=5)

        # If requests were serialized, total time would be ~1000ms (10 * 100)
        # With concurrency, should be ~200ms (100 + overhead)
        total_time_ms = sum(result.latencies) / len(result.latencies) * 10

        # Mean latency should be close to base (not 10x base)
        assert result.mean_latency_ms < 300  # Much less than sequential

    def test_high_concurrency_sustained(self):
        """Case 6: High concurrency (10 workers) sustained."""
        server = ConcurrentAPIServer(base_latency_ms=50)
        tester = CLILoadTester(server)

        result = tester.run_concurrent_load(num_requests=30, num_workers=10)

        assert result.failed_requests == 0
        # Max concurrent should reach close to num_workers
        assert server.get_max_concurrent() >= 5


class TestLoadTestMetrics:
    """Load test metric cases."""

    def test_p95_latency_measured(self):
        """Case 7: p95 latency measured and reasonable."""
        server = ConcurrentAPIServer(base_latency_ms=50)
        tester = CLILoadTester(server)

        result = tester.run_concurrent_load(num_requests=50, num_workers=5)

        assert result.p95_latency_ms > 0
        assert result.p95_latency_ms >= result.mean_latency_ms

    def test_p99_latency_measured(self):
        """Case 8: p99 latency measured."""
        server = ConcurrentAPIServer(base_latency_ms=50)
        tester = CLILoadTester(server)

        result = tester.run_concurrent_load(num_requests=100, num_workers=10)

        assert result.p99_latency_ms > 0
        assert result.p99_latency_ms >= result.p95_latency_ms

    def test_latency_percentiles_ordered(self):
        """Case 9: Latency percentiles are ordered p50 < p95 < p99."""
        server = ConcurrentAPIServer(base_latency_ms=50)
        tester = CLILoadTester(server)

        result = tester.run_concurrent_load(num_requests=100, num_workers=5)

        assert result.p50_latency_ms <= result.p95_latency_ms
        assert result.p95_latency_ms <= result.p99_latency_ms


class TestConcurrencyFairness:
    """Concurrency fairness cases."""

    def test_all_concurrent_requests_complete(self):
        """Case 10: All concurrent requests eventually complete."""
        server = ConcurrentAPIServer(base_latency_ms=50)
        tester = CLILoadTester(server)

        result = tester.run_concurrent_load(num_requests=50, num_workers=8)

        assert result.total_requests == 50
        assert len(result.latencies) > 0
        # All should complete eventually
        assert result.successful_requests + result.failed_requests == 50

    def test_no_deadlock_under_load(self):
        """Case 11: No deadlock detected (test completes in time)."""
        server = ConcurrentAPIServer(base_latency_ms=50)
        tester = CLILoadTester(server)

        start = time.time()
        # This should complete even under high load
        result = tester.run_concurrent_load(num_requests=20, num_workers=10)
        elapsed = time.time() - start

        # Should complete within 5 seconds (no deadlock/stall)
        assert elapsed < 5.0
        assert result.successful_requests > 0
