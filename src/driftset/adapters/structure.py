# SPDX-License-Identifier: Apache-2.0
"""AlphaFold pLDDT confidence adapter.

The nonconformity signal is ``|pLDDT/100 - lDDT|``: the model reports a per-residue
confidence ``pLDDT`` on a 0-100 scale that is meant to estimate the experimental
``lDDT`` (0-1), so calibrating residuals on the unit scale yields prediction
intervals for the true lDDT.

Scope honesty (v0.1): this adapter ships as **code** with a synthetic test. A
*measured* coverage card is deferred to a later release because no turnkey
pre-paired pLDDT/lDDT table is publicly distributed — pLDDT lives in the
B-factor column of AlphaFold structures, and lDDT must be computed against
experimental references (e.g. CAMEO/CASP targets). To assemble a calibration
frame yourself:

1. Download AlphaFold predictions and the matching experimental structures
   (CAMEO provides continuous releases; CC-BY-4.0).
2. Compute per-residue/per-model lDDT-Cα against the references.
3. Build a DataFrame with a ``plddt`` column (0-100) and an ``lddt`` column
   (0-1), then pass it to :func:`alphafold_plddt_adapter` and the same
   :func:`~driftset.decision.report.compute_coverage_report` used for affinity.
"""

from __future__ import annotations

from .tabular import TabularAdapter

PLDDT_COLUMN = "plddt"
LDDT_COLUMN = "lddt"


def alphafold_plddt_adapter() -> TabularAdapter:
    """Return the pLDDT adapter expecting ``plddt`` (0-100) and ``lddt`` (0-1).

    ``pLDDT`` is rescaled to the unit interval so residuals are ``lDDT -
    pLDDT/100`` and intervals land on the lDDT scale.
    """
    return TabularAdapter(
        name="alphafold-plddt",
        prediction_column=PLDDT_COLUMN,
        truth_column=LDDT_COLUMN,
        feature_columns=(PLDDT_COLUMN,),
        prediction_transform=lambda v: v / 100.0,
    )
