"""Cost ledger rows, deterministic attribution and Decimal-exact totals (X3.1).

Implements the ``mlops/cost_ledger.py`` surface of the X3 API contract
(plan A):

* :class:`CostRow` -- one immutable-by-convention cost observation.
* :func:`attribute` -- resolve a row to an owner, a fallback label, or
  ``"<unallocated>"``.
* :func:`total_spend` -- exact :class:`~decimal.Decimal` sum of amounts.
* :func:`unallocated_fraction` -- amount-weighted fraction of unallocated
  spend.
* :func:`fallback_only_spend` -- spend attributed via a fallback label
  rather than an explicit owner.
* :func:`tag_namespace` -- tag a row's resolved namespace as known/unknown
  without mutating it.

Deviation from the API-contract text
------------------------------------
The contract states ``namespace_known = attribute(row, []) in
known_namespaces``. An empty fallback list makes the namespace the row's
explicit ``owner`` only, so a row with ``owner=None`` and a populated
``labels`` mapping would always resolve to ``"<unallocated>"`` and be
reported unknown. The frozen X3.1-S2-07 test requires a label-attributed
row (``owner=None``, ``labels={"team": "platform"}``) to be marked known.
:func:`tag_namespace` therefore derives the namespace with the row's own
labels as the fallback candidates, i.e. ``attribute(row, list(row.labels))``,
which selects the first truthy value in ``row.labels`` when ``row.owner``
is falsy. This satisfies both frozen cases: label-only attribution resolves
to ``"platform"`` (known), while an explicit ``owner="external-vendor"``
still wins over ``labels={"team": "platform"}`` and is reported unknown.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Set

__all__ = [
    "CostRow",
    "attribute",
    "total_spend",
    "unallocated_fraction",
    "fallback_only_spend",
    "tag_namespace",
]

UNALLOCATED = "<unallocated>"


@dataclass
class CostRow:
    """One cost observation retained verbatim.

    ``quantity`` is genuinely optional and is preserved as supplied
    (``None`` stays ``None``). ``labels`` defaults to a fresh dict per row,
    never a shared mutable object.
    """

    source: str
    resource: str
    owner: Optional[str]
    amount: Decimal
    window_start: int
    window_end: int
    labels: dict = field(default_factory=dict)
    quantity: Optional[float] = None


def attribute(row: CostRow, fallbacks: List[str]) -> str:
    """Resolve ``row`` to an owner string.

    A truthy explicit ``row.owner`` always wins. Otherwise the first truthy
    value among ``row.labels.get(name)`` for ``name`` in ``fallbacks`` (in
    order) is used. If neither yields a value, ``"<unallocated>"``.
    """
    if row.owner:
        return row.owner
    for name in fallbacks:
        value = row.labels.get(name)
        if value:
            return value
    return UNALLOCATED


def total_spend(rows: List[CostRow]) -> Decimal:
    """Return the exact :class:`Decimal` sum of every row's amount.

    Starts from ``Decimal("0")``; float is never used. An empty ledger
    returns ``Decimal("0")``.
    """
    total = Decimal("0")
    for row in rows:
        total += row.amount
    return total


def unallocated_fraction(rows: List[CostRow], fallbacks: List[str]) -> Decimal:
    """Return the amount-weighted fraction of unallocated spend.

    ``U`` is the sum of amounts of rows whose :func:`attribute` is
    ``"<unallocated>"`` divided by :func:`total_spend`. An empty ledger
    returns ``Decimal("0")`` rather than raising ``ZeroDivisionError``.
    """
    if not rows:
        return Decimal("0")
    total = total_spend(rows)
    unallocated = Decimal("0")
    for row in rows:
        if attribute(row, fallbacks) == UNALLOCATED:
            unallocated += row.amount
    return unallocated / total


def fallback_only_spend(rows: List[CostRow], fallbacks: List[str]) -> Decimal:
    """Return the exact Decimal sum of fallback-attributed spend.

    A row counts only when its :func:`attribute` is not
    ``"<unallocated>"`` and it has no explicit ``owner`` -- i.e. it was
    attributed solely via a fallback label.
    """
    total = Decimal("0")
    for row in rows:
        if not row.owner and attribute(row, fallbacks) != UNALLOCATED:
            total += row.amount
    return total


def tag_namespace(row: CostRow, known_namespaces: Set[str]) -> Dict[str, object]:
    """Tag ``row`` with whether its resolved namespace is known.

    Returns ``{"row": row, "namespace_known": bool}``. ``row`` is returned
    by identity and is never mutated. The namespace is derived with the
    row's own labels as fallback candidates (see the module docstring for
    why this deviates from the contract's ``attribute(row, [])`` wording).
    """
    namespace = attribute(row, list(row.labels))
    return {"row": row, "namespace_known": namespace in known_namespaces}
