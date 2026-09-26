"""
X1.2 -- Offline/online retrieval from one definition (C41, C42, C43, C44, C47).

Point-in-time (as-of) offline retrieval and online latest lookup share exactly
one pure evaluator, :func:`evaluate_feature`, so the two paths are consistent by
construction. The store keeps, per ``(namespace, name)``, the rows it has
accepted, each tagged with a monotonically increasing ingestion sequence used to
break ties on identical ``event_ts`` fully deterministically.

Stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .features import ContractViolation, FeatureError
from .features import FeatureRegistry

__all__ = ["FeatureStore", "evaluate_feature"]


def evaluate_feature(
    definition: dict, rows: list, as_of_ts: int
) -> Any:
    """Evaluate ``definition`` over ``rows`` as of ``as_of_ts``.

    Pure and deterministic: neither ``definition`` nor ``rows`` is mutated.
    Rows may carry an internal ingestion sequence under ``"_seq"``; when absent
    the row's index in ``rows`` is used, so a direct call is repeatable.

    * ``"last"`` ignores ``window_seconds`` and returns the value of the row
      with the greatest ``event_ts <= as_of_ts``, ties broken by greatest
      sequence; ``None`` when there is no candidate.
    * ``"sum"``/``"count"`` consider rows in ``(as_of_ts - window_seconds,
      as_of_ts]`` when a window is declared, or ``event_ts <= as_of_ts``
      otherwise. ``None`` when no row matches.
    """
    agg = definition.get("agg")
    window = definition.get("window_seconds")

    if agg == "last":
        best_row = None
        best_key = None
        for index, row in enumerate(rows):
            ts = row.get("event_ts")
            if ts is None or ts > as_of_ts:
                continue
            seq = row.get("_seq")
            if seq is None:
                seq = index
            key = (ts, seq)
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
        if best_row is None:
            return None
        return best_row.get("value")

    if agg in ("sum", "count"):
        matched: List[Any] = []
        for row in rows:
            ts = row.get("event_ts")
            if ts is None or ts > as_of_ts:
                continue
            if window is not None and not ts > as_of_ts - window:
                continue
            matched.append(row.get("value"))
        if not matched:
            return None
        if agg == "count":
            return len(matched)
        total = 0
        for value in matched:
            total = total + value
        return total

    raise ValueError("unknown aggregation %r" % (agg,))


class FeatureStore:
    """Holds ingested rows and answers offline/online lookups via one evaluator.

    The store pins no version at construction: it resolves ``(namespace, name)``
    to the registry's newest published version on every ingest and every read.
    """

    def __init__(self, registry: FeatureRegistry, contract: Any = None) -> None:
        self._registry = registry
        self._contract = contract
        self._rows: Dict[tuple, List[dict]] = {}
        self._seq = 0

    def _definition(self, namespace: str, name: str) -> dict:
        try:
            version_id = self._registry.latest(namespace, name)
        except KeyError as exc:
            raise FeatureError("unknown feature %s/%s" % (namespace, name)) from exc
        return self._registry.get(version_id)

    def ingest(self, namespace: str, name: str, rows: list) -> Dict[str, int]:
        """Validate and store ``rows`` for a feature, returning ``{"stored": n}``.

        A missing feature raises :class:`~mlops.features.FeatureError`. When a
        contract is configured every row is checked before anything is stored,
        so a single violation rejects the whole batch atomically.
        """
        self._definition(namespace, name)

        if self._contract is not None:
            for row in rows:
                decided = self._contract.evaluate(row)
                if not decided.passed:
                    raise ContractViolation(
                        "data contract violated for %s/%s: %s"
                        % (namespace, name, decided.reason)
                    )

        bucket = self._rows.setdefault((namespace, name), [])
        for row in rows:
            self._seq += 1
            stored = dict(row)
            stored["_seq"] = self._seq
            bucket.append(stored)
        return {"stored": len(rows)}

    def get_offline(
        self, namespace: str, name: str, entity: str, as_of_ts: int
    ) -> Any:
        """Return the as-of value for ``entity`` from the stored rows."""
        definition = self._definition(namespace, name)
        entity_rows = [
            row
            for row in self._rows.get((namespace, name), [])
            if row.get("entity") == entity
        ]
        return evaluate_feature(definition, entity_rows, as_of_ts)

    def get_online(
        self, namespace: str, name: str, entity: str, now_ts: int
    ) -> Any:
        """Return the online value, delegating to ``get_offline`` exactly once."""
        return self.get_offline(namespace, name, entity, now_ts)
