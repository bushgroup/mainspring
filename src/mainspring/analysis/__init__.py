"""Processing for what the reader returns: the arrays a pipeline computes from a UIMF
file, as pure functions.

**Never imports Qt**, directly or transitively, on the same terms as `mainspring.uimf`:
it is part of the base install, which a pipeline machine installs with no GUI stack.
`tests/test_package.py`, `tests/test_architecture.py` and `tools/check_public.py` hold
the line for both packages.

Every name here is published. A name is frozen in the self-check once a program
outside this repository imports it at a release tag; until then its signature may
change in a minor release, and the release notes say so.

Where things live:

    profile.py  bin_range, scan_profile, summed_profile, ScanProfile -- an m/z
                window's intensity per scan, one frame or summed over frames,
                each through its own calibration (imports mainspring.uimf)
    shape.py    argmax, centroid, median, iqr, fwhm, rise_10_90, local_maxima --
                what is read off a profile, in scans, no smoothing (numpy only)
    smooth.py   gaussian_smooth, gaussian_gain -- a Gaussian smoother for an
                arrival-time profile, its width in ms and required (numpy only)

What the package leaves to the caller, being one program's or one experiment's
convention: which frames make a run (a first repetition left out, a method frame,
a fold), times after a release mark, a count added per stored point, and which
smoother at what width. `profile.py` sums the frames it is given; `shape.py` reads
the profile it is given.

The dependency order is one way: `analysis` may import `mainspring.uimf`, nothing in
`uimf` names `analysis`, and neither names the viewer (lab record, task 41).
"""

from __future__ import annotations

from .profile import ScanProfile, bin_range, scan_profile, summed_profile
from .shape import argmax, centroid, fwhm, iqr, local_maxima, median, rise_10_90
from .smooth import gaussian_gain, gaussian_smooth

__all__ = [
    "ScanProfile",
    "argmax",
    "bin_range",
    "centroid",
    "fwhm",
    "gaussian_gain",
    "gaussian_smooth",
    "iqr",
    "local_maxima",
    "median",
    "rise_10_90",
    "scan_profile",
    "summed_profile",
]
