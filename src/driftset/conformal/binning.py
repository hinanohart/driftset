# SPDX-License-Identifier: MIT
"""Helpers to build Mondrian (class-conditional) bins.

crepes performs Mondrian calibration when given a ``bins`` vector; these helpers
turn a continuous covariate or a categorical key into stable integer bin labels
that can be shared between the calibration and test sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior quantile edges of ``values`` for ``n_bins`` strata.

    Compute the edges on the calibration values, then reuse them on the test set
    via :func:`assign_bins` so both sides share the same strata.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    values = np.asarray(values, dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.quantile(values, quantiles)


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign integer bin labels in ``[0, len(edges)]`` from precomputed edges."""
    return np.digitize(np.asarray(values, dtype=float), np.asarray(edges, dtype=float))


def factorize_bins(keys: np.ndarray) -> np.ndarray:
    """Map categorical keys to integer codes (e.g. per-target Mondrian bins)."""
    codes, _ = pd.factorize(np.asarray(keys))
    return codes
