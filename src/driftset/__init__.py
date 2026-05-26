# SPDX-License-Identifier: Apache-2.0
"""driftset — reproducible, GPU-free conformal coverage harness for scientific
foundation models under distribution shift.

driftset does not invent a new shift-correction algorithm. It provides a
pip-installable, public-data-reproducible reference implementation that unifies
multiple scientific foundation models (Boltz-2 binding affinity, AlphaFold
pLDDT) behind one nonconformity-adapter protocol, and reports a measured
naive-vs-calibrated coverage gap (the "coverage report card").
"""

__version__ = "0.1.0a1"

__all__ = ["__version__"]
