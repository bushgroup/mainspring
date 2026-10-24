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

    smooth.py   gaussian_smooth, gaussian_gain -- a Gaussian smoother for an
                arrival-time profile, its width in ms and required (numpy only)

The dependency order is one way: `analysis` may import `mainspring.uimf`, nothing in
`uimf` names `analysis`, and neither names the viewer (lab record, task 41).
"""

from __future__ import annotations

from .smooth import gaussian_gain, gaussian_smooth

__all__ = ["gaussian_gain", "gaussian_smooth"]
