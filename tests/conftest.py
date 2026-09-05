"""Shared fixtures. Nothing here needs a data file or the lab repository."""

from __future__ import annotations

import pytest

from synthetic import write_synthetic_uimf


@pytest.fixture
def synthetic_uimf(tmp_path):
    """A two-frame synthetic UIMF file, and the `SyntheticFile` describing it.

    Small enough to write per test: a couple of dozen scans of a few hundred points.
    """
    return write_synthetic_uimf(tmp_path / "synthetic.uimf", frames=2, scans=16, bins=4096)
