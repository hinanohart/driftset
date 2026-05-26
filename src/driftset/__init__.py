# SPDX-License-Identifier: Apache-2.0
"""driftset — reproducible, GPU-free conformal coverage harness for scientific
foundation models under distribution shift.

driftset does not invent a new shift-correction algorithm. It provides a
pip-installable, public-data-reproducible reference implementation that unifies
multiple scientific foundation models (Boltz-2 binding affinity, AlphaFold
pLDDT) behind one nonconformity-adapter protocol, and reports a measured
naive-vs-calibrated coverage gap (the "coverage report card").
"""

from .adapters.affinity import boltz2_affinity_adapter
from .adapters.base import MissingLabelsError, NonconformityAdapter
from .adapters.structure import alphafold_plddt_adapter
from .adapters.tabular import TabularAdapter
from .conformal.metrics import empirical_coverage, mean_interval_width
from .conformal.split import SplitConformalRegressor
from .conformal.weighted import WeightedSplitConformalRegressor
from .decision.api import (
    ThresholdDecision,
    evaluate_threshold_decisions,
    threshold_decisions,
)
from .decision.report import CoverageReport, compute_coverage_report
from .shift.detector import ClassifierRatioEstimator, MMDDiagnostic

__version__ = "0.1.0a1"

__all__ = [
    "__version__",
    "NonconformityAdapter",
    "MissingLabelsError",
    "TabularAdapter",
    "boltz2_affinity_adapter",
    "alphafold_plddt_adapter",
    "SplitConformalRegressor",
    "WeightedSplitConformalRegressor",
    "empirical_coverage",
    "mean_interval_width",
    "ClassifierRatioEstimator",
    "MMDDiagnostic",
    "CoverageReport",
    "compute_coverage_report",
    "ThresholdDecision",
    "threshold_decisions",
    "evaluate_threshold_decisions",
]
