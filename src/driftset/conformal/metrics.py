# SPDX-License-Identifier: Apache-2.0
"""Coverage metrics computed directly from intervals and ground truth."""

from __future__ import annotations

import numpy as np


def _validate(intervals: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.asarray(intervals, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError(f"intervals must have shape (n, 2), got {intervals.shape}")
    if intervals.shape[0] != y_true.shape[0]:
        raise ValueError(f"intervals {intervals.shape[0]} and y_true {y_true.shape[0]} disagree")
    if intervals.shape[0] == 0:
        raise ValueError("empty intervals")
    if np.isnan(intervals).any() or np.isnan(y_true).any():
        raise ValueError(
            "NaN present in intervals or y_true; clean/drop missing rows before "
            "computing coverage (a silent 0.0 would understate coverage)"
        )
    return intervals, y_true


def empirical_coverage(intervals: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of labels falling within their (inclusive) interval."""
    intervals, y_true = _validate(intervals, y_true)
    lower, upper = intervals[:, 0], intervals[:, 1]
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def mean_interval_width(intervals: np.ndarray) -> float:
    """Mean upper-minus-lower width across intervals."""
    intervals = np.asarray(intervals, dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError(f"intervals must have shape (n, 2), got {intervals.shape}")
    return float(np.mean(intervals[:, 1] - intervals[:, 0]))
