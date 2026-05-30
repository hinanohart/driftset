# SPDX-License-Identifier: MIT
"""Smoke test: the package imports and exposes a version."""

import driftset


def test_version_present():
    assert isinstance(driftset.__version__, str)
    assert driftset.__version__ == "0.1.0a1"
