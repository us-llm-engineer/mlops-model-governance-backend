"""F3.3: DriftWindowFrame for pandas-based rolling PSI and daily summaries.

Wraps a DataFrame of (timestamp, value) pairs and provides:
- rolling_psi: PSI computed over rolling windows
- to_daily_summary: Daily aggregations (count, mean, std, min, max)
- export_records: JSON-safe records with NaN/NaT replaced by None
"""

from __future__ import annotations

import math
from typing import List

import pandas as pd

from mlops.kernel import ValidationFailed
from mlops.drift import psi_from_samples

__all__ = [
    "DriftWindowFrame",
]


class DriftWindowFrame:
    """Wraps a DataFrame of (timestamp, value) pairs for drift analysis.

    Constructor validates:
    - >= 30 rows
    - All ts and value are finite (not NaN, inf, -inf)
    - ts is strictly non-decreasing (ties allowed, decreases rejected)
    """

    def __init__(self, pairs: List[tuple[float, float]]) -> None:
        """Initialize DriftWindowFrame from (timestamp, value) pairs.

        Args:
            pairs: Sequence of (ts, value) tuples

        Raises:
            ValidationFailed: if < 30 rows, non-finite values, or decreasing ts
        """
        if not isinstance(pairs, (list, tuple)):
            raise ValidationFailed("pairs must be a list or tuple of (ts, value) tuples")

        if len(pairs) < 30:
            raise ValidationFailed("DriftWindowFrame requires >= 30 rows")

        # Extract ts and value columns
        ts_list = []
        value_list = []

        for i, pair in enumerate(pairs):
            try:
                ts, value = pair
            except (TypeError, ValueError):
                raise ValidationFailed(f"pairs[{i}] is not a valid (ts, value) tuple")

            # Validate ts is finite
            if not isinstance(ts, (int, float)) or isinstance(ts, bool):
                raise ValidationFailed(f"pairs[{i}][0] (ts) must be numeric")
            if isinstance(ts, float) and not math.isfinite(ts):
                raise ValidationFailed(f"pairs[{i}][0] (ts) is not finite")

            # Validate value is finite
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValidationFailed(f"pairs[{i}][1] (value) must be numeric")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValidationFailed(f"pairs[{i}][1] (value) is not finite")

            ts_list.append(float(ts))
            value_list.append(float(value))

        # Validate ts is strictly non-decreasing (ties allowed, decreases rejected)
        for i in range(1, len(ts_list)):
            if ts_list[i] < ts_list[i - 1]:
                raise ValidationFailed(
                    f"ts at index {i} ({ts_list[i]}) is less than previous ({ts_list[i-1]}); "
                    "ts must be strictly non-decreasing"
                )

        # Build DataFrame
        self.df = pd.DataFrame({"ts": ts_list, "value": value_list})

    def rolling_psi(
        self, reference: List[float], window_size: int = 200, bins: int = 10
    ) -> pd.Series:
        """Compute rolling PSI using a window over the value column.

        Args:
            reference: Reference sample values
            window_size: Size of the rolling window (must be int in [1, 5000])
            bins: Number of bins for PSI computation

        Returns:
            pandas.Series of PSI values, same length as DataFrame
            First (window_size - 1) entries are NaN per pandas convention

        Raises:
            ValidationFailed: if window_size is invalid
        """
        # Validate window_size is an int, not a bool
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise ValidationFailed("window_size must be an integer, not bool")

        if window_size < 1 or window_size > 5000:
            raise ValidationFailed(
                f"window_size must be in range [1, 5000], got {window_size}"
            )

        # Define rolling PSI function
        def compute_psi(window_values):
            # window_values is a numpy array from rolling window
            window_list = list(window_values)
            return psi_from_samples(reference, window_list, bins=bins)

        # Apply rolling window
        psi_series = self.df["value"].rolling(window=window_size).apply(compute_psi)

        # Attach to DataFrame for export_records to pick up
        self.df["psi"] = psi_series

        return psi_series

    def to_daily_summary(self) -> pd.DataFrame:
        """Group by calendar date and compute daily statistics.

        ts values are epoch-second floats. Groups by calendar date and computes:
        - count: number of records
        - mean: mean of value
        - std: standard deviation (sample std, ddof=1)
        - min: minimum of value
        - max: maximum of value

        Returns:
            DataFrame with date-based index and columns [count, mean, std, min, max]
        """
        # Convert ts to datetime for grouping
        df = self.df.copy()
        df["date"] = pd.to_datetime(df["ts"], unit="s").dt.date

        # Group by date and aggregate
        summary = df.groupby("date")["value"].agg(
            count="count",
            mean="mean",
            std="std",  # default ddof=1 for sample std
            min="min",
            max="max",
        )

        return summary

    def export_records(self) -> List[dict]:
        """Export DataFrame records as JSON-safe dicts.

        Converts DataFrame to list of dicts, replacing any NaN/NaT with None
        so that json.dumps() never fails.

        Returns:
            List of dicts, each representing one row
        """
        # Make a copy to avoid modifying the original
        df = self.df.copy()

        # Replace NaN with None for json.dumps safety
        # TRAP: pandas 3.0's copy-on-write means we must .astype(object) BEFORE
        # calling .where(cond, None) on float columns, else NaN silently persists
        for col in df.columns:
            if df[col].dtype == "float64":
                df[col] = df[col].astype(object)
                df[col] = df[col].where(pd.notna(df[col]), None)
            else:
                # For other types (datetime, etc.), use mask
                df[col] = df[col].where(pd.notna(df[col]), None)

        # Convert to records
        records = df.to_dict("records")

        return records
