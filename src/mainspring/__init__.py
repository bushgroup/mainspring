"""mainspring: an interactive viewer and a Python reader for UIMF files.

UIMF is the SQLite-based format in which SLIM ion mobility mass spectrometers
record their data: one row per (frame, scan) holding a compressed sparse
time-of-flight spectrum, plus per-frame and per-file parameter tables. This
package has two layers with one seam:

    uimf      SQLite access, blob decoding, calibration, sparse frames and
              rasterisation -- numpy, optional numba, never Qt
    viewer    the PySide6 + pyqtgraph application on top of it

Anything that imports `mainspring.uimf` must keep working with no GUI stack
installed, which is what lets a pipeline reuse the reader and what
`tools/check_public.py` exercises.
"""

from __future__ import annotations

import os

__version__ = "0.1.0"

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
"""The repository root when running from a checkout (the parent of src/)."""
EXTERNAL_DIR = os.path.join(ROOT, "external")
"""Fetched, never-committed material: PNNL's test excerpts (tools/fetch_testdata.py)."""

# The lab repo's directories a session may need, by name. Nothing in src/ hard
# codes a lab-side path; everything goes through lab_dir().
_LAB_SUBDIRS = ("tasks", "notes", "explorations", "data")


def lab_dir(name: str | None = None) -> str | None:
    """Resolve the private lab repo, or one of its top-level directories.

    Order: `$MAINSPRING_LAB`, then this repo's own root (a lab checkout that
    contains the code), then the sibling `../mainspring-lab`. Returns None when
    nothing resolves -- a public clone has no lab material and every public
    code path must work without it.
    """
    candidates = [
        os.environ.get("MAINSPRING_LAB"),
        ROOT,
        os.path.abspath(os.path.join(ROOT, "..", "mainspring-lab")),
    ]
    for root in candidates:
        if not root or not os.path.isdir(root):
            continue
        # A lab checkout is recognised by its task system, not by its name.
        if not os.path.isfile(os.path.join(root, "tasks", "README.md")):
            continue
        if name is None:
            return root
        if name not in _LAB_SUBDIRS:
            raise ValueError(f"unknown lab directory {name!r}; one of {_LAB_SUBDIRS}")
        path = os.path.join(root, name)
        return path if os.path.isdir(path) else None
    return None


__all__ = ["EXTERNAL_DIR", "ROOT", "__version__", "lab_dir"]
