#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Measure coverage under an induced covariate shift on the affinity benchmark.

Usage::

    uv run python scripts/run_shift_coverage.py

Builds a biased calibration/test split that shifts the test set toward
*low-confidence* Boltz-2 compounds (which tend to have larger errors), then
contrasts vanilla split conformal against covariate-shift–weighted conformal
using domain-classifier importance weights. Writes ``reports/shift_coverage.json``.
CPU-only; reuses the cached Zenodo download.

Honesty notes baked into the report: weighted conformal trades calibration bias
for variance, so with skewed estimated weights a few intervals can hit maximum
(infinite) size. We therefore report the *finite fraction* and *median* width
(never infinity), plus the effective calibration sample size ``n_eff``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from driftset.adapters.affinity import boltz2_affinity_adapter
from driftset.conformal.metrics import empirical_coverage
from driftset.conformal.split import SplitConformalRegressor
from driftset.conformal.weighted import WeightedSplitConformalRegressor
from driftset.datasets import zenodo_affinity as zaff
from driftset.shift.detector import ClassifierRatioEstimator, MMDDiagnostic

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
CONFIDENCES = (0.80, 0.90, 0.95)
SEED = 20260527
SHIFT_STRENGTH = 1.5


def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.sum(weights**2))


def _finite_fraction(intervals: np.ndarray) -> float:
    return float(np.mean(np.isfinite(intervals).all(axis=1)))


def _median_finite_width(intervals: np.ndarray) -> float | None:
    widths = intervals[:, 1] - intervals[:, 0]
    widths = widths[np.isfinite(widths)]
    return float(np.median(widths)) if widths.size else None


def _biased_split(frame, rng):
    """Assign rows to test with prob rising as Boltz-2 confidence falls."""
    conf = frame[zaff.CONFIDENCE_COLUMN].to_numpy(dtype=float)
    z = (conf - conf.mean()) / (conf.std() + 1e-12)
    p_test = 1.0 / (1.0 + np.exp(SHIFT_STRENGTH * z))  # low confidence -> high p_test
    is_test = rng.uniform(size=len(frame)) < p_test
    return frame[~is_test].reset_index(drop=True), frame[is_test].reset_index(drop=True)


def main() -> None:
    frame = zaff.load()
    frame = frame.dropna(subset=list(zaff.FEATURE_COLUMNS)).reset_index(drop=True)
    adapter = boltz2_affinity_adapter()
    rng = np.random.default_rng(SEED)

    cal, test = _biased_split(frame, rng)
    cal_feat = adapter.covariate_features(cal)
    test_feat = adapter.covariate_features(test)
    yhc, yc = adapter.point_predictions(cal), adapter.true_values(cal)
    yht, yt = adapter.point_predictions(test), adapter.true_values(test)

    estimator = ClassifierRatioEstimator().fit(cal_feat, test_feat)
    w_cal = estimator.weights(cal_feat)
    w_test = estimator.weights(test_feat)
    shift_score = MMDDiagnostic().shift_score(cal_feat, test_feat)

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # crepes max-size warning is reported as finite_fraction
        for c in CONFIDENCES:
            vanilla = SplitConformalRegressor(c).calibrate(yhc, yc).predict_interval(yht)
            weighted = WeightedSplitConformalRegressor(c).calibrate(
                yhc, yc, likelihood_ratios=w_cal
            )
            wt = weighted.predict_interval(yht, likelihood_ratios=w_test)
            results.append(
                {
                    "confidence": c,
                    "vanilla_coverage": empirical_coverage(vanilla, yt),
                    "weighted_coverage": empirical_coverage(wt, yt),
                    "vanilla_median_width": _median_finite_width(vanilla),
                    "weighted_median_width": _median_finite_width(wt),
                    "weighted_finite_fraction": _finite_fraction(wt),
                }
            )

    payload = {
        "dataset": zaff.provenance(frame),
        "shift": {
            "kind": "covariate shift on Boltz-2 Confidence_score (test biased low)",
            "strength": SHIFT_STRENGTH,
            "seed": SEED,
            "n_calibration": int(len(cal)),
            "n_test": int(len(test)),
            "mmd_shift_score": shift_score,
            "effective_calibration_n": _effective_sample_size(w_cal),
            "weight_source": "domain-classifier density ratio (ClassifierRatioEstimator)",
        },
        "results": results,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    # allow_nan=False: never serialize the invalid JSON token ``Infinity``; the
    # widths are already median-of-finite (or None), so this only guards regressions.
    (REPORTS_DIR / "shift_coverage.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )

    print(
        f"MMD shift score = {shift_score:.4f}  n_cal={len(cal)} n_test={len(test)} "
        f"n_eff_cal={_effective_sample_size(w_cal):.0f}"
    )
    for r in results:
        # median width is None when no interval is finite (fully degenerate);
        # format defensively so the print never crashes in that regime.
        vw = f"{r['vanilla_median_width']:.2f}" if r["vanilla_median_width"] is not None else "n/a"
        ww = (
            f"{r['weighted_median_width']:.2f}" if r["weighted_median_width"] is not None else "n/a"
        )
        print(
            f"  conf={r['confidence']:.2f}  vanilla={r['vanilla_coverage']:.4f}  "
            f"weighted={r['weighted_coverage']:.4f}  "
            f"finite={r['weighted_finite_fraction']:.3f}  "
            f"med_width {vw} -> {ww}"
        )


if __name__ == "__main__":
    main()
