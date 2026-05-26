# SPDX-License-Identifier: Apache-2.0
"""Risk-aware decisions from calibrated intervals.

Given prediction intervals and a decision threshold (e.g. an activity cutoff on
pAffinity), each item is called **above** / **below** the threshold only when its
whole interval lies on one side; otherwise it **abstains**. Because the interval
carries the conformal coverage guarantee, the error rate among *decided* items is
controlled at the chosen confidence level, with the abstention rate as the cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ABOVE = "above"
BELOW = "below"
ABSTAIN = "abstain"


@dataclass(frozen=True)
class ThresholdDecision:
    """Per-item decisions against a threshold plus summary rates."""

    labels: np.ndarray
    threshold: float
    abstain_rate: float
    decided_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "abstain_rate": self.abstain_rate,
            "decided_rate": self.decided_rate,
            "n": int(self.labels.shape[0]),
        }


def threshold_decisions(intervals: np.ndarray, threshold: float) -> ThresholdDecision:
    """Label each interval ``above`` / ``below`` / ``abstain`` w.r.t. ``threshold``."""
    intervals = np.asarray(intervals, dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError(f"intervals must have shape (n, 2), got {intervals.shape}")
    lower, upper = intervals[:, 0], intervals[:, 1]
    labels = np.where(lower > threshold, ABOVE, np.where(upper < threshold, BELOW, ABSTAIN))
    abstain_rate = float(np.mean(labels == ABSTAIN))
    return ThresholdDecision(
        labels=labels,
        threshold=float(threshold),
        abstain_rate=abstain_rate,
        decided_rate=1.0 - abstain_rate,
    )


def evaluate_threshold_decisions(
    intervals: np.ndarray, y_true: np.ndarray, threshold: float
) -> dict[str, object]:
    """Abstention rate and error rate among decided items.

    The decision error is the fraction of decided items whose called side does
    not match where the ground truth actually falls relative to ``threshold``.
    Ground truth exactly equal to ``threshold`` (measure zero on a continuous
    scale) is treated as not-above.
    """
    decision = threshold_decisions(intervals, threshold)
    y_true = np.asarray(y_true, dtype=float)
    if y_true.shape[0] != decision.labels.shape[0]:
        raise ValueError("intervals and y_true length disagree")
    decided = decision.labels != ABSTAIN
    n_decided = int(np.sum(decided))
    if n_decided == 0:
        decision_error: float | None = None
    else:
        called_above = decision.labels[decided] == ABOVE
        truth_above = y_true[decided] > threshold
        decision_error = float(np.mean(called_above != truth_above))
    return {
        "threshold": float(threshold),
        "abstain_rate": decision.abstain_rate,
        "decided_rate": decision.decided_rate,
        "n_decided": n_decided,
        "decision_error": decision_error,
    }
