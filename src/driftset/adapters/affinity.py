# SPDX-License-Identifier: Apache-2.0
"""Boltz-2 binding-affinity adapter.

The nonconformity signal is the affinity residual ``experimental_pChEMBL -
predicted_affinity`` (so the calibrated interval is on the pAffinity scale and
its half-width is interpretable in pAffinity units). This is a thin
configuration of :class:`~driftset.adapters.tabular.TabularAdapter` over the
public Data S6 columns; there is no model-specific maths to re-implement.
"""

from __future__ import annotations

from ..datasets import zenodo_affinity as zaff
from .tabular import TabularAdapter


def boltz2_affinity_adapter() -> TabularAdapter:
    """Return the adapter bound to the Data S6 column roles.

    * prediction = ``Predicted_affinity_value`` (Boltz-2)
    * ground truth = ``pChEMBL_value_median`` (experimental)
    * covariates = molecular descriptors + Boltz-2 confidence (for shift detection)
    * Mondrian group = ``UniProt_ID`` (per-target calibration)
    """
    return TabularAdapter(
        name="boltz2-affinity",
        prediction_column=zaff.PREDICTION_COLUMN,
        truth_column=zaff.TRUTH_COLUMN,
        feature_columns=zaff.FEATURE_COLUMNS,
        group_column=zaff.TARGET_COLUMN,
    )
