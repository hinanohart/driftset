# SPDX-License-Identifier: MIT
"""pLDDT adapter (code path) and the threshold Decision API, tested offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftset.adapters.base import NonconformityAdapter
from driftset.adapters.structure import (
    LDDT_COLUMN,
    PLDDT_COLUMN,
    alphafold_plddt_adapter,
)
from driftset.decision.api import (
    ABOVE,
    ABSTAIN,
    BELOW,
    evaluate_threshold_decisions,
    threshold_decisions,
)
from driftset.decision.report import compute_coverage_report


def _plddt_frame(n: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plddt = rng.uniform(50.0, 100.0, size=n)
    lddt = np.clip(plddt / 100.0 + rng.normal(0.0, 0.05, size=n), 0.0, 1.0)
    return pd.DataFrame({PLDDT_COLUMN: plddt, LDDT_COLUMN: lddt})


def test_plddt_adapter_is_protocol_and_rescales():
    adapter = alphafold_plddt_adapter()
    assert isinstance(adapter, NonconformityAdapter)
    frame = pd.DataFrame({PLDDT_COLUMN: [50.0, 100.0], LDDT_COLUMN: [0.4, 0.95]})
    np.testing.assert_allclose(adapter.point_predictions(frame), [0.5, 1.0])
    np.testing.assert_allclose(adapter.true_values(frame), [0.4, 0.95])


def test_plddt_coverage_report_on_synthetic():
    adapter = alphafold_plddt_adapter()
    cal = _plddt_frame(3000, seed=1)
    test = _plddt_frame(3000, seed=2)
    report = compute_coverage_report(adapter, cal, test, confidence=0.9)
    assert report.adapter == "alphafold-plddt"
    assert abs(report.empirical_coverage - 0.9) < 0.03


def test_threshold_decisions_labels():
    intervals = np.array([[7.5, 8.0], [5.0, 6.0], [6.5, 7.5]])
    decision = threshold_decisions(intervals, threshold=7.0)
    assert decision.labels.tolist() == [ABOVE, BELOW, ABSTAIN]
    assert decision.abstain_rate == pytest.approx(1 / 3)
    assert decision.decided_rate == pytest.approx(2 / 3)
    assert decision.to_dict()["n"] == 3


def test_threshold_decisions_validation():
    with pytest.raises(ValueError):
        threshold_decisions(np.zeros((3, 3)), threshold=1.0)


def test_evaluate_threshold_decisions_error_and_abstain():
    # decided items: [7.5,8.0] called above (truth 7.8 above -> correct),
    # [5,6] called below (truth 5.5 below -> correct); [6.5,7.5] abstains.
    intervals = np.array([[7.5, 8.0], [5.0, 6.0], [6.5, 7.5]])
    y_true = np.array([7.8, 5.5, 7.1])
    out = evaluate_threshold_decisions(intervals, y_true, threshold=7.0)
    assert out["n_decided"] == 2
    assert out["abstain_rate"] == pytest.approx(1 / 3)
    assert out["decision_error"] == pytest.approx(0.0)


def test_evaluate_threshold_decisions_all_abstain():
    intervals = np.array([[6.5, 7.5], [6.0, 8.0]])
    y_true = np.array([7.1, 7.2])
    out = evaluate_threshold_decisions(intervals, y_true, threshold=7.0)
    assert out["n_decided"] == 0
    assert out["decision_error"] is None


def test_evaluate_threshold_decisions_length_mismatch():
    with pytest.raises(ValueError):
        evaluate_threshold_decisions(np.zeros((3, 2)), np.zeros(2), threshold=1.0)
