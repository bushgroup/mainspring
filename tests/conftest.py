"""Shared fixtures. The synthetic ones need no data file and no lab repository.

Real files are a separate matter and a separate fixture: PNNL's excerpts are fetched
into `external/` and never committed, and the lab's own sample lives in the private
repository, so every test that wants one is parametrised over whatever this clone
happens to have and skips when that is nothing. A bare public clone runs the whole
suite green on synthetic files alone.
"""

from __future__ import annotations

import glob
import os

import pytest

from synthetic import write_synthetic_uimf


def real_uimf_paths() -> list[str]:
    """Every `.uimf` this clone can see: PNNL's excerpts, plus `MAINSPRING_SMOKE_UIMF`.

    `tools/fetch_testdata.py` puts the excerpts in place; the environment variable names
    one more file, which is how a lab-side session points the suite at real acquisitions
    without any path to them living in this repository.
    """
    from mainspring import EXTERNAL_DIR

    paths = sorted(glob.glob(os.path.join(EXTERNAL_DIR, "pnnl-testdata", "*.uimf")))
    smoke = os.environ.get("MAINSPRING_SMOKE_UIMF")
    if smoke and os.path.isfile(smoke):
        paths.append(os.path.abspath(smoke))
    return paths


@pytest.fixture
def synthetic_uimf(tmp_path):
    """A two-frame synthetic UIMF file, and the `SyntheticFile` describing it.

    Small enough to write per test: a couple of dozen scans of a few hundred points.
    """
    return write_synthetic_uimf(tmp_path / "synthetic.uimf", frames=2, scans=16, bins=4096)


@pytest.fixture(
    params=real_uimf_paths() or [None],
    ids=lambda path: os.path.basename(path) if path else "no-real-file",
)
def real_uimf(request):
    """One real file per parametrisation, or a skip that says why there are none.

    The `None` placeholder is deliberate: an empty parameter list would make these tests
    vanish from the report, and a check that disappears when its input is missing is
    worse than one that says what it would have needed.
    """
    if request.param is None:
        pytest.skip(
            "no .uimf file here; run tools/fetch_testdata.py or set MAINSPRING_SMOKE_UIMF"
        )
    return request.param
