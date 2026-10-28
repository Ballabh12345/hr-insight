"""Population Stability Index (PSI) and basic drift helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    breakpoints = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(breakpoints) < 2:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)
    ref_pct = (ref_counts + 1) / (len(ref) + len(breakpoints) - 1)
    cur_pct = (cur_counts + 1) / (len(cur) + len(breakpoints) - 1)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref_counts = reference.astype(str).value_counts(normalize=True)
    cur_counts = current.astype(str).value_counts(normalize=True)
    categories = set(ref_counts.index) | set(cur_counts.index)
    total = 0.0
    for cat in categories:
        r = float(ref_counts.get(cat, 1e-6))
        c = float(cur_counts.get(cat, 1e-6))
        if r == 0:
            r = 1e-6
        if c == 0:
            c = 1e-6
        total += (c - r) * np.log(c / r)
    return float(total)


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two series."""
    if pd.api.types.is_numeric_dtype(reference) and pd.api.types.is_numeric_dtype(current):
        return _psi_numeric(reference, current, bins=bins)
    return _psi_categorical(reference, current)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """PSI for every column shared by the two frames."""
    shared = [c for c in reference.columns if c in current.columns]
    return {col: psi(reference[col], current[col]) for col in shared}