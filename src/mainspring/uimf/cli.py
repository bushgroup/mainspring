"""`uimf-info`: what is in this file, does our decode agree with it, and how fast is it.

The console entry point of the reader layer, and the only part of mainspring a pipeline
machine with no GUI stack is likely to run. Three modes, all writing to stdout:

    uimf-info FILE                 parameters, frame list, per-frame scan counts
    uimf-info FILE --verify        decode every scan and compare with the stored columns
    uimf-info FILE --bench         time the decode and the raster, numba and pure

`--verify` is the reader's acceptance test in the field, not merely a developer tool:
`TIC` and `BPI` are exact ground truth on every row, and a file whose blobs do not
reproduce them is a file we do not understand. `NonZeroCount` is compared as the upper
bound it is, and `BPI_MZ` as a plus-or-minus-three-bin sanity check, because the
writers compute it inconsistently (lab record, task 01).

`--json` puts any mode's output through the reporting stamp, so a number quoted from a
run carries the version and commit that produced it.
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv: "list[str] | None" = None) -> int:
    """Run `uimf-info`; returns a process exit status, 0 for success.

    Arrives with the lab record's task 03.
    """
    raise NotImplementedError("the uimf-info command arrives with the lab record's task 03")
