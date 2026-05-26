# SPDX-License-Identifier: Apache-2.0
"""The coverage report card — driftset's headline output.

For a held-out test set it contrasts two intervals built from the *same*
calibration data:

* **calibrated** — the distribution-free split-conformal interval, which carries
  the marginal coverage guarantee;
* **naive** — a parametric Gaussian interval ``y_hat ± z · s`` where ``s`` is the
  calibration residual standard deviation. This is the interval you get if you
  *assume* the residuals are Gaussian instead of calibrating distribution-free.

The naive-vs-calibrated gap quantifies what the distribution-free calibration
buys you (or confirms the Gaussian assumption was already adequate). Every field
is measured from data; nothing here is hand-set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..adapters.base import NonconformityAdapter
from ..conformal.metrics import empirical_coverage, mean_interval_width
from ..conformal.split import SplitConformalRegressor


@dataclass(frozen=True)
class CoverageReport:
    """Measured coverage of calibrated vs naive intervals at one confidence."""

    adapter: str
    confidence: float
    n_calibration: int
    n_test: int
    target_coverage: float
    empirical_coverage: float
    naive_coverage: float
    coverage_gap: float
    naive_coverage_gap: float
    mean_interval_width: float
    naive_mean_width: float
    shift_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _naive_gaussian_intervals(
    y_hat: np.ndarray, residual_std: float, confidence: float
) -> np.ndarray:
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    half = z * residual_std
    return np.column_stack((y_hat - half, y_hat + half))


def compute_coverage_report(
    adapter: NonconformityAdapter,
    cal_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    confidence: float = 0.9,
    shift_score: float | None = None,
) -> CoverageReport:
    """Calibrate on ``cal_frame`` and measure coverage on ``test_frame``."""
    y_hat_cal = adapter.point_predictions(cal_frame)
    y_cal = adapter.true_values(cal_frame)
    y_hat_test = adapter.point_predictions(test_frame)
    y_test = adapter.true_values(test_frame)

    scr = SplitConformalRegressor(confidence=confidence).calibrate(y_hat_cal, y_cal)
    intervals = scr.predict_interval(y_hat_test)
    emp = empirical_coverage(intervals, y_test)
    width = mean_interval_width(intervals)

    residual_std = float(np.std(y_cal - y_hat_cal, ddof=1))
    naive_intervals = _naive_gaussian_intervals(y_hat_test, residual_std, confidence)
    naive_emp = empirical_coverage(naive_intervals, y_test)
    naive_width = mean_interval_width(naive_intervals)

    return CoverageReport(
        adapter=adapter.name,
        confidence=confidence,
        n_calibration=int(len(y_cal)),
        n_test=int(len(y_test)),
        target_coverage=confidence,
        empirical_coverage=emp,
        naive_coverage=naive_emp,
        coverage_gap=emp - confidence,
        naive_coverage_gap=naive_emp - confidence,
        mean_interval_width=width,
        naive_mean_width=naive_width,
        shift_score=shift_score,
    )
