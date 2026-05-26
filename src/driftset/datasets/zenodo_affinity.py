# SPDX-License-Identifier: Apache-2.0
"""Loader for the public ChEMBL-derived Boltz-2 affinity benchmark.

Source: "Supplemental data of ChEMBL-Derived Benchmark Dataset and Computational
Results of Boltz-2-Based Binding Affinity Prediction", Zenodo
DOI 10.5281/zenodo.18669539, license CC-BY-4.0. We use ``Data S6`` which pairs
representative experimental pChEMBL values (ground truth) with Boltz-2 predicted
affinity values, so coverage can be measured with **no GPU and no model
re-inference** — only a CSV download.

The data is downloaded on demand and cached; it is never vendored into the
driftset repository (its CC-BY-4.0 licence binds the data, not this Apache code).
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

DOI = "10.5281/zenodo.18669539"
RECORD_ID = "18669539"
FILE_NAME = "Data_S6_benchmark_dataset_for_predictive_performance.csv"
FILE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files/{FILE_NAME}/content"
LICENSE = "CC-BY-4.0"

#: SHA-256 of the upstream CSV, pinned for reproducibility.
EXPECTED_SHA256 = "6d603067fd4a278c2dd82a232da91ede933567916f2101e7ed438a5f2f86be7a"

#: Column roles within Data S6.
PREDICTION_COLUMN = "Predicted_affinity_value"
TRUTH_COLUMN = "pChEMBL_value_median"
CONFIDENCE_COLUMN = "Confidence_score"
TARGET_COLUMN = "UniProt_ID"
FEATURE_COLUMNS = (
    "MW",
    "Num_RB",
    "Num_HBD",
    "Num_HBA",
    "Num_HA",
    "Compound_structural_similarity",
    "Confidence_score",
)


def default_cache_dir() -> Path:
    """Per-user cache directory for downloaded calibration data."""
    return Path.home() / ".cache" / "driftset"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(cache_dir: Path | None = None, *, verify_sha: bool = True) -> Path:
    """Download Data S6 into the cache (idempotent) and return its path.

    Raises ``ValueError`` if ``verify_sha`` is set and the downloaded file's
    SHA-256 does not match :data:`EXPECTED_SHA256`.
    """
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / FILE_NAME
    if not target.exists():
        # FILE_URL is a hardcoded https Zenodo constant, never user input; the
        # explicit scheme guard rules out file:// / ftp:// before the fetch.
        if not FILE_URL.startswith("https://"):
            raise ValueError(f"refusing non-https download URL: {FILE_URL!r}")
        # Fetch into a temp sibling then atomically rename, so an interrupted
        # download never leaves a corrupt file occupying the cached path.
        tmp = target.with_name(target.name + ".part")  # pragma: no cover
        urllib.request.urlretrieve(FILE_URL, tmp)  # nosemgrep  # noqa: S310  # pragma: no cover
        tmp.replace(target)  # pragma: no cover
    if verify_sha:
        actual = sha256_of(target)
        if actual != EXPECTED_SHA256:
            raise ValueError(
                f"SHA-256 mismatch for {FILE_NAME}: expected {EXPECTED_SHA256}, "
                f"got {actual}. Delete {target} and re-download."
            )
    return target


def load(
    cache_dir: Path | None = None,
    *,
    verify_sha: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """Return Data S6 as a DataFrame.

    With ``dropna`` (default) rows missing the prediction or ground-truth column
    are removed, because downstream coverage metrics intentionally refuse NaN.
    """
    path = download(cache_dir, verify_sha=verify_sha)
    frame = pd.read_csv(path)
    if dropna:
        frame = frame.dropna(subset=[PREDICTION_COLUMN, TRUTH_COLUMN]).reset_index(drop=True)
    return frame


def provenance(frame: pd.DataFrame | None = None) -> dict[str, object]:
    """Provenance record for reports (DOI, file, SHA, URL, row count)."""
    record: dict[str, object] = {
        "doi": DOI,
        "file": FILE_NAME,
        "url": FILE_URL,
        "license": LICENSE,
        "sha256": EXPECTED_SHA256,
        "prediction_column": PREDICTION_COLUMN,
        "truth_column": TRUTH_COLUMN,
    }
    if frame is not None:
        record["n_rows_used"] = int(len(frame))
    return record
