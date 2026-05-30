# SPDX-License-Identifier: MIT
"""Affinity adapter + coverage report card, tested offline.

These tests never touch the network: they build a small DataFrame with the same
column names as Zenodo Data S6 and exercise the adapter wiring and the report
maths. The real measured numbers live in ``reports/affinity_coverage.json``,
produced by ``scripts/run_affinity_coverage.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftset.adapters.affinity import boltz2_affinity_adapter
from driftset.adapters.base import NonconformityAdapter
from driftset.datasets import zenodo_affinity as zaff
from driftset.decision.report import CoverageReport, compute_coverage_report


def _s6_like_frame(n: int, *, seed: int, noise_sd: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pred = rng.uniform(4.0, 9.0, size=n)
    truth = pred + rng.normal(0.0, noise_sd, size=n)
    return pd.DataFrame(
        {
            zaff.PREDICTION_COLUMN: pred,
            zaff.TRUTH_COLUMN: truth,
            zaff.CONFIDENCE_COLUMN: rng.uniform(0.5, 1.0, size=n),
            zaff.TARGET_COLUMN: rng.choice(["P1", "P2", "P3"], size=n),
            "MW": rng.uniform(200, 600, size=n),
            "Num_RB": rng.integers(0, 10, size=n),
            "Num_HBD": rng.integers(0, 5, size=n),
            "Num_HBA": rng.integers(0, 10, size=n),
            "Num_HA": rng.integers(10, 40, size=n),
            "Compound_structural_similarity": rng.uniform(0.0, 1.0, size=n),
        }
    )


def test_adapter_is_protocol_and_wired_to_s6_columns():
    adapter = boltz2_affinity_adapter()
    assert isinstance(adapter, NonconformityAdapter)
    assert adapter.name == "boltz2-affinity"
    assert adapter.prediction_column == zaff.PREDICTION_COLUMN
    assert adapter.truth_column == zaff.TRUTH_COLUMN
    assert adapter.group_column == zaff.TARGET_COLUMN


def test_adapter_extracts_predictions_truth_and_features():
    adapter = boltz2_affinity_adapter()
    frame = _s6_like_frame(20, seed=1)
    assert adapter.point_predictions(frame).shape == (20,)
    assert adapter.true_values(frame).shape == (20,)
    assert adapter.covariate_features(frame).shape == (20, len(zaff.FEATURE_COLUMNS))
    assert adapter.group_keys(frame).shape == (20,)


@pytest.mark.parametrize("confidence", [0.80, 0.90, 0.95])
def test_coverage_report_hits_target_on_synthetic(confidence: float):
    adapter = boltz2_affinity_adapter()
    cal = _s6_like_frame(3000, seed=7)
    test = _s6_like_frame(3000, seed=8)
    report = compute_coverage_report(adapter, cal, test, confidence=confidence)

    assert isinstance(report, CoverageReport)
    assert report.adapter == "boltz2-affinity"
    assert report.target_coverage == confidence
    assert abs(report.empirical_coverage - confidence) < 0.03
    assert report.mean_interval_width > 0.0
    assert 0.0 <= report.naive_coverage <= 1.0
    # gap definitions are self-consistent
    assert report.coverage_gap == pytest.approx(report.empirical_coverage - confidence)


def test_coverage_report_to_dict_is_serialisable():
    adapter = boltz2_affinity_adapter()
    frame = _s6_like_frame(500, seed=3)
    report = compute_coverage_report(adapter, frame, frame, confidence=0.9)
    d = report.to_dict()
    assert set(d) >= {
        "adapter",
        "confidence",
        "empirical_coverage",
        "naive_coverage",
        "coverage_gap",
        "mean_interval_width",
    }
    assert d["shift_score"] is None


def test_provenance_record_has_doi_and_sha():
    prov = zaff.provenance()
    assert prov["doi"] == "10.5281/zenodo.18669539"
    assert len(str(prov["sha256"])) == 64
