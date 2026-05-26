# SPDX-License-Identifier: Apache-2.0
"""Moat layer: weighted CP, shift detection, and Mondrian binning.

The headline test builds a *controlled* covariate shift with a known likelihood
ratio, so it has an analytic oracle: under heteroscedastic noise and a shift
toward the noisy region, vanilla split CP under-covers and weighted CP (with the
true ratio) recovers near-nominal coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from driftset.conformal.binning import (
    assign_bins,
    factorize_bins,
    quantile_bin_edges,
)
from driftset.conformal.metrics import empirical_coverage
from driftset.conformal.split import SplitConformalRegressor
from driftset.conformal.weighted import WeightedSplitConformalRegressor
from driftset.shift.detector import ClassifierRatioEstimator, MMDDiagnostic


def _gen(n: int, a: float, b: float, seed: int):
    """Heteroscedastic data; x ~ Beta(a, b) controls the covariate distribution."""
    rng = np.random.default_rng(seed)
    x = np.clip(rng.beta(a, b, n), 0.05, 0.95)
    sd = 0.5 + 2.0 * x
    y_hat = 2.0 * x
    y = y_hat + rng.normal(0.0, sd)
    return x, y_hat, y


def test_weighted_cp_recovers_coverage_under_covariate_shift():
    # cal favours low x (low noise); test favours high x (high noise).
    xc, yhc, yc = _gen(6000, 1, 3, seed=1)
    xt, yht, yt = _gen(6000, 3, 1, seed=2)
    # true ratio p_test/p_cal for Beta(3,1)/Beta(1,3) = x^2 / (1-x)^2
    w_cal = xc**2 / (1.0 - xc) ** 2
    w_test = xt**2 / (1.0 - xt) ** 2

    vanilla = SplitConformalRegressor(0.9).calibrate(yhc, yc).predict_interval(yht)
    cov_vanilla = empirical_coverage(vanilla, yt)

    weighted = WeightedSplitConformalRegressor(0.9).calibrate(yhc, yc, likelihood_ratios=w_cal)
    cov_weighted = empirical_coverage(weighted.predict_interval(yht, likelihood_ratios=w_test), yt)

    assert cov_vanilla < 0.75  # vanilla badly under-covers under shift
    assert 0.87 <= cov_weighted <= 0.96  # weighted recovers near nominal
    assert abs(cov_weighted - 0.9) < abs(cov_vanilla - 0.9)


def test_weighted_validation():
    with pytest.raises(RuntimeError):
        WeightedSplitConformalRegressor().predict_interval(
            np.zeros(3), likelihood_ratios=np.ones(3)
        )
    with pytest.raises(ValueError):
        WeightedSplitConformalRegressor().calibrate(
            np.zeros(3), np.zeros(3), likelihood_ratios=np.ones(4)
        )
    with pytest.raises(ValueError):
        WeightedSplitConformalRegressor().calibrate(
            np.zeros(3), np.zeros(3), likelihood_ratios=np.array([-1.0, 1.0, 1.0])
        )


def test_weighted_bins_fail_loud():
    # crepes-weighted 0.1.3's weighted+binned path is upstream-broken; the facade
    # must refuse it loudly rather than surface a cryptic upstream crash.
    with pytest.raises(NotImplementedError):
        WeightedSplitConformalRegressor(0.9).calibrate(
            np.zeros(5), np.zeros(5), likelihood_ratios=np.ones(5), bins=np.zeros(5)
        )
    fitted = WeightedSplitConformalRegressor(0.9).calibrate(
        np.zeros(5), np.zeros(5), likelihood_ratios=np.ones(5)
    )
    with pytest.raises(NotImplementedError):
        fitted.predict_interval(np.zeros(5), likelihood_ratios=np.ones(5), bins=np.zeros(5))


def test_classifier_ratio_estimator_directionally_correct():
    xc, _, _ = _gen(6000, 1, 3, seed=1)
    xt, _, _ = _gen(6000, 3, 1, seed=2)
    w_true = xt**2 / (1.0 - xt) ** 2
    est = ClassifierRatioEstimator().fit(xc.reshape(-1, 1), xt.reshape(-1, 1))
    w_est = est.weights(xt.reshape(-1, 1))
    assert np.all(w_est >= 0) and np.all(np.isfinite(w_est))
    assert np.corrcoef(w_est, w_true)[0, 1] > 0.7


def test_classifier_ratio_no_shift_weights_near_one():
    xc, _, _ = _gen(4000, 2, 2, seed=3)
    xt, _, _ = _gen(4000, 2, 2, seed=4)
    est = ClassifierRatioEstimator().fit(xc.reshape(-1, 1), xt.reshape(-1, 1))
    assert 0.9 <= est.weights(xt.reshape(-1, 1)).mean() <= 1.1


def test_classifier_ratio_requires_fit():
    with pytest.raises(RuntimeError):
        ClassifierRatioEstimator().weights(np.zeros((3, 1)))


def test_mmd_discriminates_shift():
    xc, _, _ = _gen(2000, 1, 3, seed=1)
    xt_shift, _, _ = _gen(2000, 3, 1, seed=2)
    xt_same, _, _ = _gen(2000, 1, 3, seed=5)
    mmd = MMDDiagnostic()
    score_shift = mmd.shift_score(xc.reshape(-1, 1), xt_shift.reshape(-1, 1))
    score_same = mmd.shift_score(xc.reshape(-1, 1), xt_same.reshape(-1, 1))
    assert score_shift > score_same
    assert score_same < 0.05


def test_quantile_bins_and_factorize():
    values = np.arange(100, dtype=float)
    edges = quantile_bin_edges(values, n_bins=4)
    bins = assign_bins(values, edges)
    assert set(np.unique(bins)) == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        quantile_bin_edges(values, n_bins=1)
    codes = factorize_bins(np.array(["a", "b", "a", "c"]))
    assert codes.tolist() == [0, 1, 0, 2]


def test_weighted_extra_validation():
    with pytest.raises(ValueError):
        WeightedSplitConformalRegressor(confidence=1.5)
    scr = WeightedSplitConformalRegressor(0.9).calibrate(
        np.zeros(5), np.zeros(5), likelihood_ratios=np.ones(5)
    )
    assert scr.is_fitted
    with pytest.raises(ValueError):
        scr.predict_interval(np.zeros(3), likelihood_ratios=np.ones(4))
    with pytest.raises(ValueError):
        WeightedSplitConformalRegressor().calibrate(
            np.zeros(0), np.zeros(0), likelihood_ratios=np.zeros(0)
        )


def test_estimator_input_validation():
    with pytest.raises(ValueError):
        ClassifierRatioEstimator(clip=0.9)
    est = ClassifierRatioEstimator()
    with pytest.raises(ValueError):
        est.fit(np.zeros(3), np.zeros((3, 1)))  # 1-D features
    with pytest.raises(ValueError):
        est.fit(np.zeros((3, 2)), np.zeros((3, 3)))  # dimension mismatch
    with pytest.raises(ValueError):
        est.fit(np.zeros((0, 2)), np.zeros((3, 2)))  # empty


def test_mondrian_per_bin_coverage():
    # two groups with very different noise; Mondrian calibrates each separately.
    rng = np.random.default_rng(0)
    n = 4000
    group = rng.integers(0, 2, size=2 * n)
    sd = np.where(group == 0, 0.5, 3.0)
    y_hat = np.zeros(2 * n)
    y = y_hat + rng.normal(0.0, sd)
    cal_idx, test_idx = slice(0, n), slice(n, 2 * n)

    scr = SplitConformalRegressor(0.9).calibrate(y_hat[cal_idx], y[cal_idx], bins=group[cal_idx])
    intervals = scr.predict_interval(y_hat[test_idx], bins=group[test_idx])
    g_test = group[test_idx]
    for g in (0, 1):
        cov_g = empirical_coverage(intervals[g_test == g], y[test_idx][g_test == g])
        assert abs(cov_g - 0.9) < 0.05
