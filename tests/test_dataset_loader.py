# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the Zenodo affinity loader (no network).

A tiny CSV fixture with the Data S6 column names is placed in a temp cache so
``download``/``load`` exercise their real code paths without fetching anything.
"""

from __future__ import annotations

import pandas as pd
import pytest

from driftset.datasets import zenodo_affinity as zaff


def _write_fixture(cache_dir) -> None:
    frame = pd.DataFrame(
        {
            zaff.PREDICTION_COLUMN: [6.0, 7.0, 8.0, 5.0],
            zaff.TRUTH_COLUMN: [6.1, 6.8, float("nan"), 5.2],
            zaff.CONFIDENCE_COLUMN: [0.9, 0.8, 0.7, 0.95],
            zaff.TARGET_COLUMN: ["P1", "P1", "P2", "P2"],
            "MW": [400.0, 420.0, 380.0, 360.0],
            "Num_RB": [5, 6, 4, 3],
            "Num_HBD": [1, 2, 1, 0],
            "Num_HBA": [6, 7, 5, 4],
            "Num_HA": [27, 29, 25, 22],
            "Compound_structural_similarity": [0.5, 0.6, 0.4, 0.7],
        }
    )
    (cache_dir / zaff.FILE_NAME).write_text(frame.to_csv(index=False))


def test_sha256_of_roundtrip(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"driftset")
    digest = zaff.sha256_of(p)
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_load_drops_nan_rows(tmp_path):
    _write_fixture(tmp_path)
    frame = zaff.load(cache_dir=tmp_path, verify_sha=False, dropna=True)
    # one row had NaN truth -> dropped
    assert len(frame) == 3
    assert not frame[zaff.TRUTH_COLUMN].isna().any()


def test_load_keeps_nan_when_dropna_false(tmp_path):
    _write_fixture(tmp_path)
    frame = zaff.load(cache_dir=tmp_path, verify_sha=False, dropna=False)
    assert len(frame) == 4


def test_verify_sha_mismatch_raises(tmp_path):
    _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        zaff.download(cache_dir=tmp_path, verify_sha=True)


def test_non_https_url_refused(tmp_path, monkeypatch):
    # empty cache so the download path runs; force a non-https URL.
    monkeypatch.setattr(zaff, "FILE_URL", "ftp://evil/Data.csv")
    with pytest.raises(ValueError, match="non-https"):
        zaff.download(cache_dir=tmp_path, verify_sha=False)


def test_default_cache_dir_under_home():
    assert zaff.default_cache_dir().name == "driftset"


def test_provenance_includes_rowcount_when_frame_given(tmp_path):
    _write_fixture(tmp_path)
    frame = zaff.load(cache_dir=tmp_path, verify_sha=False)
    prov = zaff.provenance(frame)
    assert prov["n_rows_used"] == 3
    assert prov["license"] == "CC-BY-4.0"
