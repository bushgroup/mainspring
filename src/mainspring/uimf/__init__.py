"""The UIMF data layer: SQLite access, blob decoding, calibration, sparse frames,
rasterisation, and the `uimf-info` command line.

**Never imports Qt**, directly or transitively -- a pipeline that installs mainspring
for the reader must not pull a GUI into its import path. `tests/test_package.py` and
`tools/check_public.py` both hold the line.

Where things live:

    reader.py   UimfFile, GlobalParams, FrameParams -- SQLite, short-lived ro connections
    decode.py   the LZF + run-length-zero blob, both directions
    calib.py    Calibration: bins to m/z, scans to arrival time, as axis tables
    frame.py    SparseFrame: a frame as points, CSR by scan, never dense
    raster.py   DisplayAxes, RasterResult: a frame reduced to viewport pixels
    cache.py    FrameCache: recently read frames under a byte budget
    cli.py      uimf-info

The dependency order is one way: `reader` uses `decode`, `calib` and `frame`; `raster`
uses `calib` and `frame`; nothing lower reaches back up. `notes/architecture.md` in the
lab record says why (lab record, task 02).
"""

from __future__ import annotations

from .cache import FrameCache
from .calib import Calibration
from .decode import decode_intensities, encode_intensities
from .frame import SparseFrame
from .raster import DisplayAxes, RasterResult, rasterise
from .reader import FrameParams, GlobalParams, UimfFile

__all__ = [
    "Calibration",
    "DisplayAxes",
    "FrameCache",
    "FrameParams",
    "GlobalParams",
    "RasterResult",
    "SparseFrame",
    "UimfFile",
    "decode_intensities",
    "encode_intensities",
    "rasterise",
]
