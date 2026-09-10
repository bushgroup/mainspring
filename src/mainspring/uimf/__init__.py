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
    writer.py   UimfWriter: the one place mainspring writes a UIMF file
    cli.py      uimf-info

The dependency order is one way: `writer` uses `decode`, `calib` and `frame`; `reader`
uses those three and `writer`, for the names of the parameters only mainspring writes;
`raster` uses `calib` and `frame`; nothing lower reaches back up. `notes/architecture.md`
in the lab record says why (lab record, task 02).
"""

from __future__ import annotations

from .cache import FrameCache
from .calib import Calibration, arrival_time_ms, scan_axis_ms
from .decode import decode_frame_blobs, decode_intensities, encode_intensities, numba_available
from .frame import SparseFrame, sum_frames
from .raster import DisplayAxes, RasterResult, profile, rasterise
from .reader import FrameParams, GlobalParams, UimfFile
from .writer import (
    DETECTOR_BITS,
    FRAME_COMPLETE,
    METHOD_FRAME,
    REPETITION,
    REPETITIONS,
    WRITER_STAMP,
    FrameSpec,
    GlobalSpec,
    UimfWriter,
)

__all__ = [
    "DETECTOR_BITS",
    "FRAME_COMPLETE",
    "METHOD_FRAME",
    "REPETITION",
    "REPETITIONS",
    "WRITER_STAMP",
    "Calibration",
    "DisplayAxes",
    "FrameCache",
    "FrameParams",
    "FrameSpec",
    "GlobalParams",
    "GlobalSpec",
    "RasterResult",
    "SparseFrame",
    "UimfFile",
    "UimfWriter",
    "arrival_time_ms",
    "decode_frame_blobs",
    "decode_intensities",
    "encode_intensities",
    "numba_available",
    "profile",
    "rasterise",
    "scan_axis_ms",
    "sum_frames",
]
