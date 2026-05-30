# SPDX-License-Identifier: MIT
"""Nonconformity adapter protocol.

An adapter binds a specific model's tabular output to the arrays the conformal
engines consume. The labelled/unlabelled split is deliberate and is the single
most common source of bugs in conformal pipelines:

* **Calibration** rows carry a ground-truth label, so a *true* nonconformity
  signal exists (the residual ``true - predicted``).
* **Inference** rows carry only the model's point prediction; no residual can
  be computed, only a calibrated interval applied.

Adapters therefore expose ``point_predictions`` (always available) and
``true_values`` (labelled rows only — raises :class:`MissingLabelsError`
otherwise). ``covariate_features`` feeds shift detection and ``group_keys``
feeds Mondrian binning.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


class MissingLabelsError(ValueError):
    """Raised when ground-truth values are requested for unlabelled rows."""


@runtime_checkable
class NonconformityAdapter(Protocol):
    """Structural contract every adapter satisfies.

    Implementations operate on a :class:`pandas.DataFrame` and return NumPy
    arrays so the conformal engines stay model-agnostic.
    """

    #: Stable identifier used in reports and provenance.
    name: str

    def point_predictions(self, frame: pd.DataFrame) -> np.ndarray:
        """Model point predictions, shape ``(n,)``. Always available."""
        ...

    def true_values(self, frame: pd.DataFrame) -> np.ndarray:
        """Ground-truth labels, shape ``(n,)``.

        Raises :class:`MissingLabelsError` for unlabelled inference frames.
        """
        ...

    def covariate_features(self, frame: pd.DataFrame) -> np.ndarray:
        """Features describing each row, shape ``(n, d)``, for shift detection."""
        ...

    def group_keys(self, frame: pd.DataFrame) -> np.ndarray | None:
        """Per-row Mondrian category, shape ``(n,)``, or ``None`` if ungrouped."""
        ...


def residuals(adapter: NonconformityAdapter, frame: pd.DataFrame) -> np.ndarray:
    """Calibration residuals ``true - predicted`` (crepes' sign convention).

    Requires labelled rows; propagates :class:`MissingLabelsError` from the
    adapter when labels are absent.
    """
    y = np.asarray(adapter.true_values(frame), dtype=float)
    y_hat = np.asarray(adapter.point_predictions(frame), dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError(f"true_values {y.shape} and point_predictions {y_hat.shape} disagree")
    return y - y_hat
