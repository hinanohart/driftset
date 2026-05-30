# SPDX-License-Identifier: MIT
"""Core conformal correctness on synthetic data with analytically known coverage.

The synthetic generator emits a point prediction plus homoscedastic Gaussian
noise, so split conformal at confidence ``c`` must achieve empirical coverage
≈ ``c`` on a held-out test set (marginal coverage guarantee). This is the
ground-truth check that the adapter -> split-CP -> metrics path is sound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftset.adapters.base import (
    MissingLabelsError,
    NonconformityAdapter,
    residuals,
)
from driftset.adapters.tabular import TabularAdapter
from driftset.conformal.metrics import empirical_coverage, mean_interval_width
from driftset.conformal.split import SplitConformalRegressor


def _synthetic_frame(n: int, *, seed: int, noise_sd: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y_hat = 2.0 * x  # the "model" prediction
    y_true = y_hat + rng.normal(0.0, noise_sd, size=n)  # residual ~ N(0, noise_sd)
    return pd.DataFrame({"pred": y_hat, "truth": y_true, "x": x})


def _adapter() -> TabularAdapter:
    return TabularAdapter(
        name="synthetic",
        prediction_column="pred",
        truth_column="truth",
        feature_columns=("x",),
    )


def test_tabular_adapter_satisfies_protocol():
    assert isinstance(_adapter(), NonconformityAdapter)


def test_missing_labels_raises():
    adapter = TabularAdapter(name="inf", prediction_column="pred", truth_column=None)
    frame = pd.DataFrame({"pred": [1.0, 2.0]})
    with pytest.raises(MissingLabelsError):
        adapter.true_values(frame)


def test_residuals_follow_true_minus_pred_convention():
    adapter = _adapter()
    frame = pd.DataFrame({"pred": [1.0, 5.0], "truth": [1.5, 4.0], "x": [0.0, 0.0]})
    np.testing.assert_allclose(residuals(adapter, frame), [0.5, -1.0])


def test_predict_before_calibrate_raises():
    with pytest.raises(RuntimeError):
        SplitConformalRegressor(confidence=0.9).predict_interval(np.zeros(3))


@pytest.mark.parametrize("confidence", [0.80, 0.90, 0.95])
def test_split_conformal_marginal_coverage(confidence: float):
    adapter = _adapter()
    cal = _synthetic_frame(4000, seed=11)
    test = _synthetic_frame(4000, seed=99)

    scr = SplitConformalRegressor(confidence=confidence).calibrate(
        adapter.point_predictions(cal), adapter.true_values(cal)
    )
    intervals = scr.predict_interval(adapter.point_predictions(test))
    cov = empirical_coverage(intervals, adapter.true_values(test))

    # Marginal coverage guarantee with finite-sample slack (~sqrt(c(1-c)/n)).
    assert abs(cov - confidence) < 0.025, f"coverage {cov:.4f} vs target {confidence}"
    assert mean_interval_width(intervals) > 0.0


def test_clamping_respected():
    adapter = _adapter()
    cal = _synthetic_frame(2000, seed=3)
    scr = SplitConformalRegressor(confidence=0.9, y_min=-10.0, y_max=10.0).calibrate(
        adapter.point_predictions(cal), adapter.true_values(cal)
    )
    intervals = scr.predict_interval(np.array([0.0, 0.0]))
    assert np.all(intervals[:, 0] >= -10.0)
    assert np.all(intervals[:, 1] <= 10.0)


def test_covariate_features_default_uses_prediction_column():
    adapter = TabularAdapter(name="s", prediction_column="pred", truth_column="truth")
    frame = pd.DataFrame({"pred": [1.0, 2.0], "truth": [1.0, 2.0]})
    assert adapter.covariate_features(frame).shape == (2, 1)


def test_group_keys_none_and_present():
    assert _adapter().group_keys(_synthetic_frame(3, seed=1)) is None
    grouped = TabularAdapter(
        name="g", prediction_column="pred", truth_column="truth", group_column="g"
    )
    frame = pd.DataFrame({"pred": [1.0, 2.0], "truth": [1, 2], "g": ["a", "b"]})
    np.testing.assert_array_equal(grouped.group_keys(frame), np.array(["a", "b"]))


def test_prediction_and_truth_transforms_applied():
    adapter = TabularAdapter(
        name="t",
        prediction_column="pred",
        truth_column="truth",
        prediction_transform=lambda v: v / 100.0,
    )
    frame = pd.DataFrame({"pred": [50.0, 100.0], "truth": [0.4, 0.9]})
    np.testing.assert_allclose(adapter.point_predictions(frame), [0.5, 1.0])
    np.testing.assert_allclose(adapter.true_values(frame), [0.4, 0.9])


def test_missing_column_raises_keyerror():
    with pytest.raises(KeyError):
        _adapter().point_predictions(pd.DataFrame({"nope": [1.0]}))


def test_metrics_validation_errors():
    with pytest.raises(ValueError):
        empirical_coverage(np.zeros((3, 3)), np.zeros(3))
    with pytest.raises(ValueError):
        empirical_coverage(np.zeros((2, 2)), np.zeros(3))
    with pytest.raises(ValueError):
        empirical_coverage(np.zeros((0, 2)), np.zeros(0))
    with pytest.raises(ValueError):
        mean_interval_width(np.zeros((3, 3)))


def test_calibrate_shape_mismatch_and_empty():
    with pytest.raises(ValueError):
        SplitConformalRegressor().calibrate(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        SplitConformalRegressor().calibrate(np.zeros(0), np.zeros(0))


def test_invalid_confidence_rejected():
    with pytest.raises(ValueError):
        SplitConformalRegressor(confidence=1.5)


def test_coverage_metrics_reject_nan():
    with pytest.raises(ValueError):
        empirical_coverage(np.array([[0.0, np.nan]]), np.array([0.0]))
    with pytest.raises(ValueError):
        empirical_coverage(np.array([[0.0, 1.0]]), np.array([np.nan]))


class _MismatchedAdapter:
    name = "bad"

    def point_predictions(self, frame):
        return np.zeros(3)

    def true_values(self, frame):
        return np.zeros(4)

    def covariate_features(self, frame):
        return np.zeros((3, 1))

    def group_keys(self, frame):
        return None


def test_residuals_shape_mismatch_raises():
    with pytest.raises(ValueError):
        residuals(_MismatchedAdapter(), pd.DataFrame({"x": [1]}))
