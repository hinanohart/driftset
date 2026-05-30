# SPDX-License-Identifier: MIT
"""Split conformal regression — a thin facade over :class:`crepes.ConformalRegressor`.

driftset does not re-implement the conformal mathematics; ``crepes`` (BSD-3)
provides the standard, normalized, and Mondrian variants. This facade adopts
crepes' ``true - predicted`` residual convention, validates shapes and
non-emptiness, and offers a small, typed surface that the adapters and reports
build on. Note the *standard* regressor produces symmetric intervals (absolute
residual quantile), so the residual sign only affects the normalized, Mondrian,
and weighted variants used later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from crepes import ConformalRegressor


@dataclass
class SplitConformalRegressor:
    """Split conformal regressor producing two-sided prediction intervals.

    Parameters
    ----------
    confidence:
        Target coverage in ``(0, 1)`` (e.g. ``0.9`` for 90% intervals).
    y_min / y_max:
        Optional clamps applied to interval endpoints (e.g. ``0`` and ``1`` for
        a confidence that lives on the unit interval).
    """

    confidence: float = 0.9
    y_min: float = -np.inf
    y_max: float = np.inf
    _engine: ConformalRegressor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")

    @property
    def is_fitted(self) -> bool:
        return self._engine is not None

    def calibrate(
        self,
        y_hat_cal: np.ndarray,
        y_cal: np.ndarray,
        *,
        sigmas: np.ndarray | None = None,
        bins: np.ndarray | None = None,
    ) -> SplitConformalRegressor:
        """Fit on a calibration set of predictions and ground-truth labels.

        ``bins`` selects Mondrian (class-conditional) calibration; ``sigmas``
        selects normalized (difficulty-scaled) intervals.
        """
        y_hat_cal = np.asarray(y_hat_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=float)
        if y_hat_cal.shape != y_cal.shape:
            raise ValueError(f"y_hat_cal {y_hat_cal.shape} and y_cal {y_cal.shape} disagree")
        if y_hat_cal.size == 0:
            raise ValueError("calibration set is empty")
        residuals_cal = y_cal - y_hat_cal
        self._engine = ConformalRegressor().fit(residuals_cal, sigmas=sigmas, bins=bins)
        return self

    def predict_interval(
        self,
        y_hat: np.ndarray,
        *,
        sigmas: np.ndarray | None = None,
        bins: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``(n, 2)`` lower/upper interval endpoints for ``y_hat``."""
        if self._engine is None:
            raise RuntimeError("calibrate() must be called before predict_interval()")
        y_hat = np.asarray(y_hat, dtype=float)
        return self._engine.predict_int(
            y_hat,
            sigmas=sigmas,
            bins=bins,
            confidence=self.confidence,
            y_min=self.y_min,
            y_max=self.y_max,
        )
