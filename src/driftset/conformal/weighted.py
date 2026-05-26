# SPDX-License-Identifier: Apache-2.0
"""Covariate-shift–weighted split conformal regression.

A thin facade over :class:`crepes_weighted.ConformalRegressor` (BSD-3). Weighted
conformal prediction (Tibshirani et al., 2019) restores marginal coverage under
**covariate shift** by reweighting calibration scores with the likelihood ratio
``w(x) = p_test(x) / p_cal(x)``. The same ``w(.)`` is evaluated on the
calibration points (at fit time) and on the test points (at predict time).

It corrects covariate shift only. If the *predictor's* error behaviour changes
(concept shift), no importance weight repairs it — :mod:`driftset.shift` can
flag that case but not fix it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from crepes_weighted import ConformalRegressor as WeightedConformalRegressor


@dataclass
class WeightedSplitConformalRegressor:
    """Weighted split conformal regressor for covariate-shift correction."""

    confidence: float = 0.9
    y_min: float = -np.inf
    y_max: float = np.inf
    _engine: WeightedConformalRegressor | None = field(default=None, init=False, repr=False)

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
        likelihood_ratios: np.ndarray,
        sigmas: np.ndarray | None = None,
        bins: np.ndarray | None = None,
    ) -> WeightedSplitConformalRegressor:
        """Fit on calibration predictions, labels, and per-point weights ``w(x_cal)``."""
        y_hat_cal = np.asarray(y_hat_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=float)
        weights = np.asarray(likelihood_ratios, dtype=float)
        if not (y_hat_cal.shape == y_cal.shape == weights.shape):
            raise ValueError(
                f"y_hat_cal {y_hat_cal.shape}, y_cal {y_cal.shape}, "
                f"likelihood_ratios {weights.shape} must all match"
            )
        if y_hat_cal.size == 0:
            raise ValueError("calibration set is empty")
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("likelihood_ratios must be finite and non-negative")
        self._engine = WeightedConformalRegressor().fit(
            y_cal - y_hat_cal, sigmas=sigmas, bins=bins, likelihood_ratios=weights
        )
        return self

    def predict_interval(
        self,
        y_hat: np.ndarray,
        *,
        likelihood_ratios: np.ndarray,
        sigmas: np.ndarray | None = None,
        bins: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``(n, 2)`` intervals using test-point weights ``w(x_test)``."""
        if self._engine is None:
            raise RuntimeError("calibrate() must be called before predict_interval()")
        y_hat = np.asarray(y_hat, dtype=float)
        weights = np.asarray(likelihood_ratios, dtype=float)
        if y_hat.shape != weights.shape:
            raise ValueError(f"y_hat {y_hat.shape} and likelihood_ratios {weights.shape} disagree")
        return self._engine.predict(
            y_hat,
            sigmas=sigmas,
            bins=bins,
            likelihood_ratios=weights,
            confidence=self.confidence,
            y_min=self.y_min,
            y_max=self.y_max,
        )
