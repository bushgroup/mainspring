"""`uimf-info`: what is in this file, does our decode agree with it, and how fast is it.

The console entry point of the reader layer, and the only part of mainspring a pipeline
machine with no GUI stack is likely to run. Three modes, all writing to stdout:

    uimf-info FILE                 parameters, frame list, per-frame scan counts
    uimf-info FILE --verify        decode every scan and compare with the stored columns
    uimf-info FILE --bench         time the decode and the raster, numba and pure

`--verify` is the reader's acceptance test in the field, not merely a developer tool:
`TIC` and `BPI` are exact ground truth on every row, and a file whose blobs do not
reproduce them is a file we do not understand. `NonZeroCount` is compared as the upper
bound it is, and `BPI_MZ` as a sanity check on the *spread* of the writer's own bin
convention rather than on its absolute value, because the writers compute it
inconsistently -- one is a constant `TimeOffset` out, one never computed it at all
(lab record, task 01).

`--json` puts any mode's output through the reporting stamp, so a number quoted from a
run carries the version and commit that produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .calib import Calibration
from .decode import decode_frame_blobs, numba_available
from .raster import DisplayAxes, profile, rasterise
from .reader import UimfFile, connect

__all__ = ["main"]

MZ_TOLERANCE_BINS = 3.0
"""How far apart the implied `BPI_MZ` bins of one file may be, around their own median,
before `--verify` calls it a failure. Three is what the worst writer we have costs
(lab record, task 01); a real decode error moves this by thousands."""


def main(argv: "list[str] | None" = None) -> int:
    """Run `uimf-info`; returns a process exit status, 0 for success."""
    parser = argparse.ArgumentParser(
        prog="uimf-info",
        description="Inspect, verify and benchmark a UIMF file.",
    )
    parser.add_argument("path", help="the .uimf file to read")
    parser.add_argument("--verify", action="store_true",
                        help="decode every scan and compare with the stored columns")
    parser.add_argument("--bench", action="store_true",
                        help="time the decode and the rasteriser on one frame")
    parser.add_argument("--frame", type=int, default=None, metavar="N",
                        help="restrict --verify and --bench to one frame")
    parser.add_argument("--mz-tolerance-bins", type=float, default=MZ_TOLERANCE_BINS,
                        metavar="B", help=f"BPI_MZ spread allowed (default {MZ_TOLERANCE_BINS})")
    parser.add_argument("--json", nargs="?", const="-", default=None, metavar="PATH",
                        help="write the stamped results as JSON ('-' for stdout)")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    try:
        uimf = UimfFile(args.path)
    except FileNotFoundError:
        print(f"uimf-info: no such file: {args.path}", file=sys.stderr)
        return 2

    results: dict = {"file": os.path.basename(uimf.path), "path": uimf.path}
    status = 0
    try:
        results["summary"] = _summarise(uimf, args.frame)
        _print_summary(results["summary"])
        if args.verify:
            results["verify"] = _verify(uimf, args.frame, args.mz_tolerance_bins)
            status |= _print_verify(results["verify"])
        if args.bench:
            results["bench"] = _bench(uimf, args.frame)
            _print_bench(results["bench"])
    except (ValueError, KeyError) as failure:
        print(f"uimf-info: {failure}", file=sys.stderr)
        return 2

    if args.json is not None:
        from .. import report

        stamped = report.stamp(results, task="uimf-info")
        if args.json == "-":
            print(json.dumps(stamped, indent=2, default=str))
        else:
            with open(args.json, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(stamped, handle, indent=2, default=str)
            print(f"\nwrote {args.json}")
    return status


# --- what is in this file ------------------------------------------------------------


def _summarise(uimf: UimfFile, only: int | None) -> dict:
    globals_ = uimf.global_params()
    numbers = uimf.frame_numbers()
    if only is not None:
        numbers = [n for n in numbers if n == only]
        if not numbers:
            raise KeyError(f"frame {only} is not in this file")
    types = uimf.frame_types()

    frames = []
    for number in numbers:
        params = uimf.frame_params(number)
        scan, _, bpi, tic = uimf.scan_summary(number)
        frames.append({
            "frame": number,
            "frame_type": types.get(number, params.frame_type),
            "scans": params.scans,
            "scans_stored": int(scan.size),
            "scan_range": [int(scan[0]), int(scan[-1])] if scan.size else None,
            "accumulations": params.accumulations,
            "average_tof_length_ns": params.average_tof_length_ns,
            "duration_ms": params.duration_ms,
            "calibration_slope": params.calibration_slope,
            "calibration_intercept": params.calibration_intercept,
            "calibration_done": params.calibration_done,
            "tic": float(tic.sum()),
            "bpi": float(bpi.max()) if bpi.size else 0.0,
        })
    return {
        "size_bytes": os.path.getsize(uimf.path),
        "legacy_only": uimf.is_legacy_only,
        "tables": sorted(uimf.tables),
        "instrument_name": globals_.instrument_name,
        "date_started": globals_.date_started,
        "num_frames_param": globals_.num_frames,
        "frames_present": len(uimf.frame_numbers()),
        "bins": globals_.bins,
        "bin_width_ns": globals_.bin_width_ns,
        "tof_intensity_type": globals_.tof_intensity_type,
        "dtype": str(globals_.dtype),
        "time_offset_ns": globals_.time_offset_ns,
        "numba": numba_available(),
        "frames": frames,
        "global_extra": dict(globals_.extra),
    }


def _print_summary(summary: dict) -> None:
    print(f"{summary['instrument_name'] or '(no instrument name)'}"
          f"   {summary['date_started'] or '(no date)'}"
          f"   {summary['size_bytes'] / 1e6:.1f} MB")
    print(f"  tables      {'legacy only' if summary['legacy_only'] else 'modern + legacy'}"
          f" ({len(summary['tables'])} present)")
    print(f"  bins        {summary['bins']} x {summary['bin_width_ns']:g} ns"
          f"   intensity {summary['tof_intensity_type']} ({summary['dtype']})")
    print(f"  TimeOffset  {summary['time_offset_ns']:g} ns"
          f"   (read, and deliberately not applied to the calibration)")
    print(f"  frames      {summary['frames_present']} present,"
          f" {summary['num_frames_param']} claimed by NumFrames")
    print(f"  decode      {'numba' if summary['numba'] else 'pure Python (numba absent)'}")
    print()
    print("  frame  type  scans  stored  scan range     ms      accum         TIC      BPI")
    for row in summary["frames"]:
        span = f"{row['scan_range'][0]}-{row['scan_range'][1]}" if row["scan_range"] else "-"
        print(f"  {row['frame']:5d}  {row['frame_type']:4d}  {row['scans']:5d}"
              f"  {row['scans_stored']:6d}  {span:>11s}  {row['duration_ms']:7.1f}"
              f"  {row['accumulations']:5d}  {row['tic']:10.0f}  {row['bpi']:7.0f}")


# --- does our decode agree with it ---------------------------------------------------


def _verify(uimf: UimfFile, only: int | None, mz_tolerance_bins: float) -> dict:
    globals_ = uimf.global_params()
    numbers = uimf.frame_numbers() if only is None else [only]
    frames = []
    for number in numbers:
        params = uimf.frame_params(number)
        scan, non_zero, bpi, tic = uimf.scan_summary(number)
        frame = uimf.read_frame(number)
        with connect(uimf.path, uimf.busy_timeout_ms) as conn:
            stored_mz = np.asarray(
                [row[0] for row in conn.execute(
                    "SELECT BPI_MZ FROM Frame_Scans WHERE FrameNum = ? ORDER BY ScanNum",
                    (number,),
                )],
                dtype=np.float64,
            ) if "BPI_MZ" in _scan_columns(conn) else np.zeros(scan.size)

        ours_tic = frame.tic()[scan]
        ours_bpi = frame.bpi()[scan].astype(np.float64)
        ours_points = np.diff(frame.scan_start)[scan]
        row = {
            "frame": number,
            "rows": int(scan.size),
            "points": len(frame),
            "tic_mismatches": int(np.count_nonzero(ours_tic != tic)),
            "bpi_mismatches": int(np.count_nonzero(ours_bpi != bpi)),
            "non_zero_count_exceeded": int(np.count_nonzero(ours_points > non_zero)),
            "non_zero_count_exact": int(np.count_nonzero(ours_points == non_zero)),
        }
        row.update(_verify_bpi_mz(frame, params.calibration(globals_.bin_width_ns),
                                 scan, stored_mz, mz_tolerance_bins))
        frames.append(row)
    return {"mz_tolerance_bins": mz_tolerance_bins, "frames": frames}


def _scan_columns(conn) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(Frame_Scans)")}


def _verify_bpi_mz(
    frame, calibration: Calibration, scan: np.ndarray, stored_mz: np.ndarray,
    tolerance_bins: float,
) -> dict:
    """Compare the writer's `BPI_MZ` with the calibration of our own argmax bin.

    Two separate questions, and only the first is about our decode. Whether the writer's
    `BPI_MZ` inverts to a whole bin says the formula, its units and its constants are
    right. Whether that bin is *our* argmax bin is the writer's own convention, which
    drifts by up to three bins and, on one 2011 file, by a constant `TimeOffset`; so the
    check is on the spread of the offsets rather than on their value.
    """
    usable = stored_mz.size == scan.size and calibration.usable
    ours_bin = frame.bpi_bin()[scan] if usable else np.empty(0)
    keep = (stored_mz > 0) & (ours_bin >= 0) if usable else np.zeros(0, dtype=bool)
    skipped = {"bpi_mz_checked": 0, "bpi_mz_spread_bins": None,
               "bpi_mz_median_offset_bins": None, "bpi_mz_fractional_max": None,
               "bpi_mz_ok": True, "bpi_mz_note": ""}
    if not usable or not keep.any():
        skipped["bpi_mz_note"] = "no usable BPI_MZ column"
        return skipped
    implied = np.asarray(calibration.bin_of(stored_mz[keep]), dtype=np.float64)
    # One 2011 writer leaves a placeholder on every row -- 1.4481e-4, which inverts to
    # bin 0.1 -- while its base peak moves across thousands of bins. A column that sits
    # still while the data moves was never computed, and comparing against it says
    # nothing about our decode (lab record, task 01).
    if np.ptp(implied) < 1.0 and np.ptp(ours_bin[keep]) > tolerance_bins:
        skipped["bpi_mz_note"] = "the same value on every row: never computed"
        return skipped
    offset = implied - ours_bin[keep]
    fractional = np.abs(implied - np.round(implied))
    spread = float(offset.max() - offset.min())
    return {
        "bpi_mz_checked": int(keep.sum()),
        "bpi_mz_spread_bins": spread,
        "bpi_mz_median_offset_bins": float(np.median(offset)),
        "bpi_mz_fractional_max": float(fractional.max()),
        # A hair of slack: the sample's offsets are the whole numbers 0 to -3, and a
        # spread of exactly 3 arrives from float arithmetic as 3.0000000004.
        "bpi_mz_ok": spread <= tolerance_bins + 1e-6,
        "bpi_mz_note": "",
    }


def _print_verify(verify: dict) -> int:
    print()
    print("  --verify: decoded scans against the file's own columns")
    print("  frame     rows    points   TIC   BPI   NZC>   BPI_MZ n  spread  median  frac")
    failed = 0
    for row in verify["frames"]:
        bad = (row["tic_mismatches"] or row["bpi_mismatches"]
               or row["non_zero_count_exceeded"] or not row["bpi_mz_ok"])
        failed += bool(bad)
        spread = row["bpi_mz_spread_bins"]
        median = row["bpi_mz_median_offset_bins"]
        fractional = row["bpi_mz_fractional_max"]
        print(f"  {row['frame']:5d}  {row['rows']:7d}  {row['points']:8d}"
              f"  {row['tic_mismatches']:4d}  {row['bpi_mismatches']:4d}"
              f"  {row['non_zero_count_exceeded']:4d}"
              f"  {row['bpi_mz_checked']:9d}"
              f"  {'-' if spread is None else format(spread, '6.2f')}"
              f"  {'-' if median is None else format(median, '6.2f')}"
              f"  {'-' if fractional is None else format(fractional, '.0e')}"
              f"{'   FAIL' if bad else ''}"
              f"{'   ' + row['bpi_mz_note'] if row['bpi_mz_note'] else ''}")
    print("  TIC and BPI must be 0 mismatches; NZC> counts scans whose decoded points")
    print("  exceed NonZeroCount, which is an upper bound. BPI_MZ spread is the writer's")
    print("  own bin convention, tolerance"
          f" {verify['mz_tolerance_bins']:g} bins; frac is how far its implied bin is")
    print("  from a whole one, which is what settles the calibration's units.")
    if failed:
        print(f"  {failed} frame(s) FAILED")
    else:
        print("  all frames agree")
    return 1 if failed else 0


# --- and how fast is it --------------------------------------------------------------


def _bench(uimf: UimfFile, only: int | None) -> dict:
    numbers = uimf.frame_numbers()
    number = numbers[0] if only is None else only
    globals_ = uimf.global_params()
    params = uimf.frame_params(number)
    with connect(uimf.path, uimf.busy_timeout_ms) as conn:
        blobs = [row[0] for row in conn.execute(
            "SELECT Intensities FROM Frame_Scans WHERE FrameNum = ? ORDER BY ScanNum",
            (number,),
        )]

    timings: dict = {"frame": number, "blobs": len(blobs), "numba": numba_available()}
    if numba_available():
        decode_frame_blobs(blobs, globals_.dtype, globals_.bins, backend="numba")  # warm
        timings["decode_numba_ms"] = _time(
            lambda: decode_frame_blobs(blobs, globals_.dtype, globals_.bins, backend="numba"))
    timings["decode_pure_ms"] = _time(
        lambda: decode_frame_blobs(blobs, globals_.dtype, globals_.bins, backend="pure"))

    frame = uimf.read_frame(number)
    axes = DisplayAxes.build(frame, params.calibration(globals_.bin_width_ns),
                             params.average_tof_length_ns)
    x_range, y_range = axes.full_range
    timings["points"] = len(frame)
    timings["frame_nbytes"] = frame.nbytes
    timings["read_frame_ms"] = _time(lambda: uimf.read_frame(number))
    for width, height in ((800, 600), (1600, 1000)):
        timings[f"raster_{width}x{height}_ms"] = _time(
            lambda w=width, h=height: rasterise(frame, axes, x_range, y_range, w, h))
    timings["raster_max_ms"] = _time(
        lambda: rasterise(frame, axes, x_range, y_range, 1600, 1000, "max"))
    timings["profile_x_ms"] = _time(lambda: profile(frame, axes, x_range, y_range, "x"))
    timings["profile_y_ms"] = _time(lambda: profile(frame, axes, x_range, y_range, "y"))
    return timings


def _time(call, repeats: int = 3) -> float:
    """Best of `repeats`, in milliseconds. Best rather than mean: this is measuring the
    code, and the slow runs are measuring whatever else the workstation was doing."""
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - started)
    return round(best * 1000.0, 2)


def _print_bench(bench: dict) -> None:
    print()
    print(f"  --bench: frame {bench['frame']}, {bench['blobs']} blobs,"
          f" {bench['points']} points, {bench['frame_nbytes'] / 1e6:.1f} MB in memory")
    for key, value in bench.items():
        if key.endswith("_ms"):
            print(f"    {key[:-3]:24s} {value:8.2f} ms")
    if not bench["numba"]:
        print("    (numba is not installed; the viewer would run the pure path)")


if __name__ == "__main__":
    sys.exit(main())
