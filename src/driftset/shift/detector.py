# SPDX-License-Identifier: MIT
"""Distribution-shift estimators.

* :class:`ClassifierRatioEstimator` — the primary importance-weight source. A
  domain classifier separates calibration (label 0) from test (label 1); the
  density ratio is ``w(x) = [p/(1-p)] * (n_cal/n_test)`` where ``p`` is the
  classifier's test-class probability. These weights feed
  :class:`~driftset.conformal.weighted.WeightedSplitConformalRegressor`.
* :class:`MMDDiagnostic` — a scalar shift score (squared MMD with an RBF kernel
  and the median-heuristic bandwidth). Diagnostic only; it produces no weights.

Both standardize features first so heterogeneous descriptor scales do not
dominate.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ClassifierRatioEstimator:
    """Domain-classifier density-ratio estimator for covariate-shift weights."""

    def __init__(self, classifier=None, *, clip: float = 1e-6) -> None:
        if not 0.0 < clip < 0.5:
            raise ValueError(f"clip must be in (0, 0.5), got {clip}")
        self._clip = clip
        self._classifier = classifier
        self._scaler: StandardScaler | None = None
        self._prior_ratio: float | None = None

    def fit(self, cal_features: np.ndarray, test_features: np.ndarray) -> ClassifierRatioEstimator:
        cal_features = np.asarray(cal_features, dtype=float)
        test_features = np.asarray(test_features, dtype=float)
        if cal_features.ndim != 2 or test_features.ndim != 2:
            raise ValueError("features must be 2-D (n, d)")
        if cal_features.shape[1] != test_features.shape[1]:
            raise ValueError("calibration and test features must share dimension")
        n_cal, n_test = len(cal_features), len(test_features)
        if n_cal == 0 or n_test == 0:
            raise ValueError("both feature sets must be non-empty")

        x = np.vstack([cal_features, test_features])
        y = np.concatenate([np.zeros(n_cal), np.ones(n_test)])
        self._scaler = StandardScaler().fit(x)
        clf = self._classifier or LogisticRegression(max_iter=1000)
        clf.fit(self._scaler.transform(x), y)
        self._classifier = clf
        self._prior_ratio = n_cal / n_test
        return self

    def weights(self, features: np.ndarray) -> np.ndarray:
        """Importance weights ``w(x) = [p/(1-p)] * (n_cal/n_test)``."""
        if self._scaler is None or self._prior_ratio is None:
            raise RuntimeError("fit() must be called before weights()")
        features = np.asarray(features, dtype=float)
        proba = self._classifier.predict_proba(self._scaler.transform(features))[:, 1]
        proba = np.clip(proba, self._clip, 1.0 - self._clip)
        return (proba / (1.0 - proba)) * self._prior_ratio


class MMDDiagnostic:
    """Squared maximum mean discrepancy (RBF kernel, median-heuristic bandwidth)."""

    def __init__(self, *, max_samples: int = 500, random_state: int = 0) -> None:
        self._max_samples = max_samples
        self._rng = np.random.default_rng(random_state)

    def _subsample(self, a: np.ndarray) -> np.ndarray:
        if len(a) <= self._max_samples:
            return a
        idx = self._rng.choice(len(a), self._max_samples, replace=False)
        return a[idx]

    def shift_score(self, cal_features: np.ndarray, test_features: np.ndarray) -> float:
        cal_features = np.asarray(cal_features, dtype=float)
        test_features = np.asarray(test_features, dtype=float)
        scaler = StandardScaler().fit(np.vstack([cal_features, test_features]))
        a = self._subsample(scaler.transform(cal_features))
        b = self._subsample(scaler.transform(test_features))

        def sq_dists(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            return np.sum(p**2, 1)[:, None] + np.sum(q**2, 1)[None, :] - 2.0 * p @ q.T

        d_ab = sq_dists(a, b)
        # median-heuristic bandwidth from pooled pairwise squared distances
        pooled = np.vstack([a, b])
        med = np.median(sq_dists(pooled, pooled))
        gamma = 1.0 / med if med > 0 else 1.0
        k_aa = np.exp(-gamma * sq_dists(a, a))
        k_bb = np.exp(-gamma * sq_dists(b, b))
        k_ab = np.exp(-gamma * d_ab)
        mmd2 = float(k_aa.mean() + k_bb.mean() - 2.0 * k_ab.mean())
        return max(mmd2, 0.0)
