<!-- SPDX-License-Identifier: Apache-2.0 -->
# driftset

**Reproducible, GPU-free conformal coverage harness for scientific foundation models under distribution shift.**

`driftset` answers one practical question for users of scientific foundation
models: *when the model reports a confidence (Boltz-2 binding affinity,
AlphaFold pLDDT), how often is the truth actually inside the prediction interval
— and what happens to that coverage when the test distribution drifts away from
the calibration distribution?*

It does this with split conformal prediction and covariate-shift–weighted
conformal prediction over **public, precomputed predictions** (no GPU, no model
re-inference required), and prints a **coverage report card**: target vs.
empirical coverage, the naive-vs-calibrated gap, mean interval width, and a
shift score.

> [!IMPORTANT]
> **Scope, honestly.** driftset does **not** introduce a new shift-correction
> method. The conformal machinery is provided by the BSD-3 libraries
> [`crepes`](https://github.com/henrikbostrom/crepes) and
> [`crepes-weighted`](https://github.com/jefjonkers/crepes-weighted); driftset
> contributes the adapter protocol, the scientific-FM bindings, the
> reproducible public-data pipeline, and the measured coverage report card.
> Weighted conformal prediction corrects **covariate** shift (a change in the
> input distribution) only; it does **not** correct **concept** shift (a change
> in the predictor's error behaviour, e.g. confidently-wrong fold-switchers).
> driftset flags the latter via a shift score but cannot repair it.

## Why this exists

Confidence scores from scientific foundation models are widely used as
decision gates, yet they are known to be miscalibrated and to degrade under
distribution shift. Prior work (e.g. CalPro, arXiv:2601.07201) studies
shift-robust conformal coverage for protein structure but ships no public code.
driftset is the engineering complement: a small, installable, public-data
package that *measures* coverage and makes the measurement reproducible.

## Install

```bash
pip install driftset        # once published
# or, from a clone:
uv sync --extra dev
```

## Quickstart

```python
from driftset.adapters.affinity import boltz2_affinity_adapter
from driftset.datasets import zenodo_affinity
from driftset.decision.report import compute_coverage_report

frame = zenodo_affinity.load()              # downloads + caches public Data S6 (CC-BY-4.0)
adapter = boltz2_affinity_adapter()
cal, test = frame.iloc[: len(frame) // 2], frame.iloc[len(frame) // 2 :]
report = compute_coverage_report(adapter, cal, test, confidence=0.90)
print(report.empirical_coverage, report.mean_interval_width)
```

### Coverage report card — Boltz-2 binding affinity

Measured on the public ChEMBL-derived Boltz-2 benchmark (random iid split,
n_cal = n_test = 4609). Every number is reproduced by
`scripts/run_affinity_coverage.py` and committed to
[`reports/affinity_coverage.json`](reports/affinity_coverage.json) /
[`reports/PROVENANCE.md`](reports/PROVENANCE.md).

| Target coverage | Conformal (calibrated) | Naive Gaussian | Conformal width (pAffinity) |
|---|---|---|---|
| 0.80 | 0.7928 | 0.7958 | 2.997 |
| 0.90 | 0.9028 | 0.8974 | 3.940 |
| 0.95 | 0.9525 | 0.9419 | 4.927 |

Reading it honestly: distribution-free conformal lands on the nominal target at
every level; the Gaussian-assumption baseline is close but under-covers in the
upper tail (0.942 vs the 0.95 target), because the affinity residuals are
slightly heavier-tailed than Gaussian. The gap here is modest — the larger
payoff appears under distribution shift, where the naive split itself degrades
(see the shift layer).

## Datasets

| Adapter | Source | License | Ground truth |
|---|---|---|---|
| Boltz-2 affinity | ChEMBL-derived Boltz-2 benchmark, Zenodo DOI [10.5281/zenodo.18669539](https://doi.org/10.5281/zenodo.18669539) | CC-BY-4.0 | experimental pChEMBL |
| AlphaFold pLDDT | AlphaFold DB pLDDT + CASP/CAMEO lDDT | CC-BY-4.0 | experimental lDDT |

The Boltz-2 affinity adapter is **measured today** (table above). The AlphaFold
pLDDT adapter is built on the same protocol; whether its coverage card ships
measured or is deferred to a later release depends on turnkey paired
pLDDT/lDDT data and is stated explicitly rather than faked. Calibration datasets
are downloaded on demand and are **not** vendored into the repository.

## License

[Apache-2.0](LICENSE). Calibration **data** retains its own upstream license
(CC-BY-4.0 / CC-BY-SA where noted); those licenses bind the data, not this code.
