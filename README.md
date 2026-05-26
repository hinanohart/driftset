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

<!-- Quickstart code + the measured coverage table are populated during the
     build from reports/*.json artifacts; no numbers are hand-written here. -->

_Results table and runnable quickstart are added in the build steps that
produce `reports/affinity_coverage.json` and `reports/PROVENANCE.md`. Every
number in this README is traceable to a committed artifact under `reports/`._

## Datasets

| Adapter | Source | License | Ground truth |
|---|---|---|---|
| Boltz-2 affinity | ChEMBL-derived Boltz-2 benchmark, Zenodo DOI [10.5281/zenodo.18669539](https://doi.org/10.5281/zenodo.18669539) | CC-BY-4.0 | experimental pChEMBL |
| AlphaFold pLDDT | AlphaFold DB pLDDT + CASP/CAMEO lDDT | CC-BY-4.0 | experimental lDDT |

driftset ships code for both adapters. Calibration datasets are downloaded
on demand and are **not** vendored into the repository.

## License

[Apache-2.0](LICENSE). Calibration **data** retains its own upstream license
(CC-BY-4.0 / CC-BY-SA where noted); those licenses bind the data, not this code.
