"""
S1 Enforcement Model: Location-based gate bypass and latency (Claim C84).

Models enforcement at three locations (BUILD, REGISTRY, SERVING) with seeded
independent bypass injection.

LATENCY SEMANTICS:
  - BUILD and REGISTRY are RELEASE-PIPELINE latencies (incurred during build/push).
    These are NOT added to mean_added_latency_s (request-path latency).
    Reported separately as mean_release_latency_s.
  - SERVING is REQUEST-PATH latency: added per request to the served release.
    Only releases that reach serving (not blocked earlier, not bypassed-out of serving)
    incur serving latency. Reported as mean_added_latency_s = total serving latency
    incurred / total releases evaluated.

ESCAPE SEMANTICS:
  - A defective release escapes if it passes (or bypasses checks at) all locations.
  - A healthy release that is blocked counts as a false positive.
  - gate_fn(release) is called for every release at every location: True = block, False = allow.

Stdlib only: random, enum, dataclasses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List

__all__ = [
    "Location",
    "EnforcementModel",
]


class Location(Enum):
    """Enforcement location in the release pipeline."""

    BUILD = "build"
    REGISTRY = "registry"
    SERVING = "serving"


@dataclass
class EnforcementModel:
    """Models enforcement at multiple pipeline locations with bypass injection.

    A defective release escapes only if it bypasses or passes every enforcing
    location. BUILD and REGISTRY enforcement add release-pipeline latency (not
    request-path). SERVING enforcement adds request-path latency per served release.
    """

    locations: List[Location]
    bypass_prob: Dict[Location, float]
    latency_s: Dict[Location, float]
    seed: int

    def __post_init__(self):
        """Validate location configuration."""
        self._rng = random.Random(self.seed)
        for loc in self.locations:
            if loc not in self.bypass_prob:
                self.bypass_prob[loc] = 0.0
            if loc not in self.latency_s:
                self.latency_s[loc] = 0.0

    def evaluate(
        self,
        releases: List[Any],
        gate_fn: Callable[[Any], bool],
    ) -> Dict[str, Any]:
        """Evaluate releases at each location with independent bypass injection.

        Args:
            releases: List of release objects.
            gate_fn: Function that evaluates a release; returns True to block,
                     False to allow.

        Returns:
            Dict with keys:
                - escaped: number of defective releases that reached production
                - blocked: number of defective releases blocked at any location
                - mean_added_latency_s: serving latency incurred / total releases
                - mean_release_latency_s: build+registry latency incurred / total releases
                - false_positives: number of healthy releases blocked
                - bypass_events: count of bypass events per location
        """
        escaped = 0
        blocked = 0
        false_positives = 0
        serving_latency_s = 0.0
        release_latency_s = 0.0
        bypass_events = {loc: 0 for loc in self.locations}

        for release in releases:
            is_defective = getattr(release, 'defective', False)
            release_blocked = False
            release_reached_serving = True
            release_serving_latency = 0.0

            # Evaluate at each location in order
            for loc in self.locations:
                # Check for bypass (independent random event per location)
                bypass = self._rng.random() < self.bypass_prob[loc]
                if bypass:
                    bypass_events[loc] += 1
                    # Bypass: skip check and skip latency at this location
                    continue

                # No bypass: call gate_fn for every release
                try:
                    gate_blocks = gate_fn(release)
                except Exception:
                    # On error, fail-closed: block
                    gate_blocks = True

                if gate_blocks:
                    # This location blocks the release
                    release_blocked = True
                    if is_defective:
                        blocked += 1
                    else:
                        false_positives += 1
                    release_reached_serving = False
                    break

                # Gate allowed it at this location
                # Add latency based on location type
                if loc == Location.SERVING:
                    # SERVING latency is request-path: incurred per served request
                    release_serving_latency += self.latency_s[loc]
                else:
                    # BUILD/REGISTRY are release-pipeline latency
                    release_latency_s += self.latency_s[loc]

            # If release reached serving, add its serving latency to total
            if release_reached_serving:
                serving_latency_s += release_serving_latency
                if is_defective:
                    escaped += 1

        # Calculate means
        total_releases = len(releases)
        mean_added_latency = serving_latency_s / total_releases if total_releases > 0 else 0.0
        mean_release_latency = release_latency_s / total_releases if total_releases > 0 else 0.0

        return {
            "escaped": escaped,
            "blocked": blocked,
            "mean_added_latency_s": mean_added_latency,
            "mean_release_latency_s": mean_release_latency,
            "false_positives": false_positives,
            "bypass_events": bypass_events,
        }
