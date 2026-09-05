"""Self-check for a fresh clone: no data file, no lab repo, no instrument needed.

Every check here must pass in a bare public clone. Checks that need something a clone
does not ship -- PNNL's test excerpts in `external/`, a file named by
`MAINSPRING_SMOKE_UIMF`, the lab repository -- are reported as SKIPPED when it is
absent, never as FAIL. What is left is still a real test of the decode path, because a
synthetic UIMF file can be written from nothing: `tests/synthetic.py` puts a few hundred
known points through the real SQLite schema and the real intensity encoder.

What it covers: the package imports, the `uimf` layer stays free of Qt, the module
layout is complete, the reporting stamp, the lab-directory resolution, the intensity
codec against the format's own rules and against itself in both directions, and a
synthetic file -- written through the schema a 2026 acquisition carries -- read back
through the whole reader, rasterised, and put through `uimf-info --verify`. Where a
real file is present, `--verify` runs on that too, which is the acceptance test the
milestone is written in terms of (lab record, task 03).

Run:  uv run tools/check_public.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "tests"))  # the synthetic writer is a test fixture

FAIL: list[str] = []
SKIPPED: list[str] = []


def check_true(name: str, cond: object) -> None:
    print("{:4s} {}".format("OK" if cond else "FAIL", name))
    if not cond:
        FAIL.append(name)


def check_raises(name: str, exc: type[BaseException], call) -> None:
    try:
        call()
    except exc:
        check_true(name, True)
        return
    except BaseException as other:  # noqa: BLE001 -- reporting, not handling
        print(f"FAIL {name} (raised {other!r}, wanted {exc.__name__})")
        FAIL.append(name)
        return
    print(f"FAIL {name} (did not raise {exc.__name__})")
    FAIL.append(name)


def skip(name: str, why: str) -> None:
    print(f"SKIP {name} ({why})")
    SKIPPED.append(name)


def section(title: str) -> None:
    print()
    print("--- " + title + " " + "-" * max(3, 72 - len(title)))


UIMF_MODULES = ("cache", "calib", "cli", "decode", "frame", "raster", "reader")
VIEWER_MODULES = (
    "app", "heatmap", "info_panel", "main_window", "settings", "side_plots", "workers",
)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    # --------------------------------------------------------------------------------
    section("package and the no-Qt seam")
    # Order matters: the data layer is imported and inspected before anything can pull
    # Qt in, because that is the condition a pipeline machine actually runs under.
    import importlib

    import numpy as np

    import mainspring
    import mainspring.report as report
    import mainspring.uimf
    from mainspring.uimf import decode

    check_true("mainspring imports and carries a version", bool(mainspring.__version__))
    for name in UIMF_MODULES:
        check_true(
            f"mainspring.uimf.{name} imports",
            importlib.import_module(f"mainspring.uimf.{name}") is not None,
        )
    qt_loaded = sorted(m for m in sys.modules if m.startswith(("PySide6", "pyqtgraph", "shiboken6")))
    check_true(f"the uimf layer loaded no Qt module ({qt_loaded or 'none'})", not qt_loaded)
    check_true(
        "the uimf layer exports its public names",
        {"UimfFile", "SparseFrame", "Calibration", "rasterise"} <= set(mainspring.uimf.__all__),
    )

    # --------------------------------------------------------------------------------
    section("the intensity codec")
    # Only tests write UIMF files, so the encoder exists to make the synthetic fixture
    # possible; the decoder is what the viewer runs, and the two must be inverses.
    check_true("ADC is int32", decode.dtype_for("ADC") == np.dtype("<i4"))
    check_true("TDC is int16", decode.dtype_for("TDC") == np.dtype("<i2"))
    check_true("FOLDED is float32", decode.dtype_for("FOLDED") == np.dtype("<f4"))
    check_raises("an unknown TOFIntensityType is refused, not guessed",
                 ValueError, lambda: decode.dtype_for("SOMETHING_NEW"))

    stream = decode.rlz_encode(np.array([0, 1, 5]), np.array([7, 8, 9]))
    check_true("run-length-zero writes a skip as a negative value",
               stream.tolist() == [7, 8, -3, 9])
    stream = decode.rlz_encode(np.array([2, 3, 4]), np.array([5, 0, 6]))
    check_true("an explicit zero is a stream entry, not a wider skip",
               stream.tolist() == [-2, 5, 0, 6])
    wide = decode.rlz_encode(np.array([0, 100000]), np.array([5, 6]), "<i2")
    check_true("a gap too wide for int16 is split into several skips",
               -int(wide[1:-1].sum()) == 99999)

    bins = np.arange(0, 114688, 700)
    sparse = decode.rlz_encode(bins, np.full(bins.size, 42))
    blob = decode.lzf_compress(sparse.tobytes())
    check_true(f"LZF compresses a sparse spectrum ({sparse.nbytes} -> {len(blob)} bytes)",
               len(blob) < sparse.nbytes / 4)
    check_true("LZF of nothing is nothing", decode.lzf_compress(b"") == b"")
    check_true("every LZF control byte is in range and points behind the cursor",
               _lzf_stream_is_valid(decode.lzf_compress(
                   decode.rlz_encode(np.arange(0, 4096, 3), np.full(1366, 7)).tobytes())))

    check_true("LZF round trips every degenerate length",
               all(decode.lzf_decompress(decode.lzf_compress(bytes(range(n)))) == bytes(range(n))
                   for n in range(40)))
    overlapping = b"AB" * 5000 + bytes(range(256)) * 20
    check_true("LZF round trips the overlapping back-references a sparse spectrum makes",
               decode.lzf_decompress(decode.lzf_compress(overlapping)) == overlapping)
    check_true("run-length-zero decoding undoes the format by hand",
               [a.tolist() for a in decode.rlz_decode(np.array([7, 8, -3, 0, 9], dtype="<i4"))]
               == [[0, 1, 6], [7, 8, 9]])
    check_true("an explicit zero decodes to no point, which is why NonZeroCount is a bound",
               [a.tolist() for a in decode.decode_intensities(
                   decode.encode_intensities(np.array([2, 3, 4]), np.array([5, 0, 6])))]
               == [[2, 4], [5, 6]])
    check_raises("a truncated LZF stream raises rather than returning a short spectrum",
                 ValueError, lambda: decode.lzf_decompress(blob[:-1]))
    check_true(f"the compiled decode path is {'available' if decode.numba_available() else 'absent, so the pure path runs'}",
               decode.numba_available() in (True, False))

    # --------------------------------------------------------------------------------
    section("a synthetic UIMF file")
    from synthetic import write_synthetic_uimf

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "synthetic.uimf")
        spec = write_synthetic_uimf(path, frames=2, scans=16, bins=4096)
        conn = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
        try:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
            rows = list(conn.execute(
                "SELECT FrameNum, ScanNum, NonZeroCount, BPI, TIC, Intensities FROM Frame_Scans"))
            modern_type = conn.execute(
                "SELECT ParamValue FROM V_Frame_Params"
                " WHERE FrameNum = 1 AND ParamName = 'FrameType'").fetchone()
            legacy_type = conn.execute(
                "SELECT FrameType FROM Frame_Parameters WHERE FrameNum = 1").fetchone()
        finally:
            conn.close()

        check_true("it writes the modern parameter tables",
                   {"Global_Params", "Frame_Param_Keys", "Frame_Params", "V_Frame_Params"} <= names)
        check_true("it writes the legacy twins as our sample carries them",
                   {"Global_Parameters", "Frame_Parameters"} <= names)
        check_true("it writes Frame_Scans and its unique index",
                   {"Frame_Scans", "pk_index_FrameScans"} <= names)
        check_true("the two parameter tables disagree about frame type, as real files do",
                   modern_type[0] == "0" and legacy_type[0] == 1)
        check_true(f"it stores only the scans with signal ({len(spec.stored_scans(1))} of 16)",
                   0 < len(spec.stored_scans(1)) < 16)
        check_true(f"scan numbering does not start at zero ({min(spec.stored_scans(1))})",
                   min(spec.stored_scans(1)) > 0)
        check_true(f"it wrote points ({spec.points(1)} in frame 1)", spec.points(1) > 0)

        stored_tic = {(f, s): tic for f, s, _, _, tic, _ in rows}
        stored_bpi = {(f, s): bpi for f, s, _, bpi, _, _ in rows}
        check_true("every stored TIC is the sum of the points that produced it",
                   all(stored_tic[k] == int(v.intensity.sum()) for k, v in spec.scan_rows.items()))
        check_true("every stored BPI is the largest of them",
                   all(stored_bpi[k] == (int(v.intensity.max()) if v.intensity.size else 0)
                       for k, v in spec.scan_rows.items()))
        check_true("NonZeroCount is an upper bound, not a count",
                   all(nzc >= spec.scan(f, s).bin_index.size for f, s, nzc, _, _, _ in rows)
                   and sum(r.non_zero_count for r in spec.scan_rows.values())
                   > sum(r.bin_index.size for r in spec.scan_rows.values()))
        check_true("every scan with signal has a blob", all(blob for *_, blob in rows))

        # --- the round trip this file exists for: encode, store, read, decode, compare
        from mainspring.uimf.cli import main as uimf_info_main
        from mainspring.uimf.frame import sum_frames
        from mainspring.uimf.raster import DisplayAxes, profile, rasterise
        from mainspring.uimf.reader import UimfFile

        uimf = UimfFile(path)
        check_true("the reader prefers the modern parameter table (FrameType 0, not 1)",
                   uimf.frame_params(1).frame_type == 0 and not uimf.is_legacy_only)
        check_true("the reader finds the frames the parameters declare",
                   uimf.frame_numbers() == list(spec.frames))

        decoded = uimf.read_frame(1)
        check_true(f"a frame reads back as its points ({len(decoded)} of them)",
                   len(decoded) == spec.points(1))
        check_true("the axis extent is the parameters, not the data",
                   decoded.extent == (spec.scans, spec.bins))
        check_true("every scan decodes to exactly the points that were written",
                   all(decoded.scan(s)[0].tolist() == spec.scan(1, s).bin_index.tolist()
                       and decoded.scan(s)[1].tolist() == spec.scan(1, s).intensity.tolist()
                       for s in spec.stored_scans(1)))
        check_true("a scan the writer never stored is an empty slice, not a missing key",
                   decoded.scan(min(spec.stored_scans(1)) - 1)[0].size == 0)
        stored = spec.stored_scans(1)
        check_true("every stored TIC is reproduced exactly",
                   decoded.tic()[stored].tolist() == [spec.scan(1, s).tic for s in stored])
        check_true("every stored BPI is reproduced exactly",
                   decoded.bpi()[stored].tolist() == [spec.scan(1, s).bpi for s in stored])
        check_true("no scan decodes to more points than NonZeroCount allows",
                   all(int(np.diff(decoded.scan_start)[s]) <= spec.scan(1, s).non_zero_count
                       for s in stored))

        params = uimf.frame_params(1)
        axes = DisplayAxes.build(decoded, params.calibration(spec.bin_width_ns),
                                 params.average_tof_length_ns)
        check_true("the default axes are m/z against arrival time",
                   (axes.x_label, axes.y_label) == ("m/z", "Arrival time (ms)"))
        check_true("the m/z axis never decreases",
                   bool((np.diff(axes.x_edges) >= 0).all()))
        raster = rasterise(decoded, axes, *axes.full_range, 320, 240)
        check_true(f"a full-range image conserves the frame total ({raster.tic_in_view:.0f})",
                   raster.tic_in_view == spec.tic(1)
                   and abs(float(raster.image.sum(dtype=np.float64)) - spec.tic(1))
                   <= 1e-5 * spec.tic(1))
        peak = rasterise(decoded, axes, *axes.full_range, 320, 240, aggregate="max")
        check_true("a full-range max image finds the largest stored intensity",
                   float(peak.image.max()) == float(decoded.intensity.max()))
        check_true("a side-plot profile totals what the image over the same window does",
                   abs(float(profile(decoded, axes, *axes.full_range, along="x")[1].sum())
                       - raster.tic_in_view) <= 1e-6 * raster.tic_in_view)
        check_true("summing the frames totals their separate TICs",
                   abs(float(sum_frames([uimf.read_frame(n) for n in spec.frames]).tic().sum())
                       - sum(spec.tic(n) for n in spec.frames)) < 1e-6)
        check_true("uimf-info --verify passes on a file whose every column we computed",
                   _quiet(uimf_info_main, [path, "--verify"]) == 0)

    # --------------------------------------------------------------------------------
    section("reporting and the lab checkout")
    stamped = report.stamp({"x": 1}, task="check")
    check_true("report.stamp carries task, date, version, commit, results",
               set(stamped) >= {"task", "date", "mainspring_version", "mainspring_commit",
                                "results"})
    check_true("report.stamp wraps the results untouched", stamped["results"] == {"x": 1})

    lab = mainspring.lab_dir()
    if lab is None:
        skip("lab_dir resolves", "no lab checkout beside this clone; a public clone has none")
    else:
        check_true("lab_dir points at a task system",
                   os.path.isfile(os.path.join(lab, "tasks", "README.md")))
        data = mainspring.lab_dir("data")
        check_true("lab_dir('data') resolves or is honestly None",
                   data is None or os.path.isdir(data))

    # --------------------------------------------------------------------------------
    section("the viewer layer")
    # Last, because importing it loads Qt -- which is exactly what the seam check above
    # must not see.
    for name in VIEWER_MODULES:
        check_true(
            f"mainspring.viewer.{name} imports",
            importlib.import_module(f"mainspring.viewer.{name}") is not None,
        )
    from mainspring.viewer.app import main as viewer_main

    check_raises("the viewer entry point is declared and says task 04 wrote nothing yet",
                 NotImplementedError, lambda: viewer_main([]))

    # --------------------------------------------------------------------------------
    section("real files, if this clone has any")
    # `uimf-info --verify` decodes every scan of every frame and compares it with the
    # file's own TIC, BPI and NonZeroCount columns. That is the whole acceptance test,
    # so a clone with a real file runs it rather than a smaller imitation of it.
    from mainspring.uimf.cli import main as uimf_info_main

    testdata = os.path.join(mainspring.EXTERNAL_DIR, "pnnl-testdata")
    present = sorted(os.path.join(testdata, f) for f in os.listdir(testdata)
                     if f.endswith(".uimf")) if os.path.isdir(testdata) else []
    if not present:
        skip("PNNL excerpt decode", "not fetched; run tools/fetch_testdata.py")
    for excerpt in present:
        check_true(f"uimf-info --verify passes on {os.path.basename(excerpt)}",
                   _quiet(uimf_info_main, [excerpt, "--verify"]) == 0)

    smoke = os.environ.get("MAINSPRING_SMOKE_UIMF")
    if not smoke:
        skip("smoke-file decode", "MAINSPRING_SMOKE_UIMF not set")
    elif not os.path.isfile(smoke):
        check_true(f"MAINSPRING_SMOKE_UIMF names a file ({smoke})", False)
    else:
        check_true(f"uimf-info --verify passes on {os.path.basename(smoke)}",
                   _quiet(uimf_info_main, [smoke, "--verify"]) == 0)

    print()
    print(f"{len(FAIL)} failed, {len(SKIPPED)} skipped")
    for name in FAIL:
        print("  FAIL", name)
    return 1 if FAIL else 0


def _quiet(call, argv) -> int:
    """Run a `main(argv)` with its output swallowed, returning its exit status.

    `uimf-info --verify` prints a table per frame, and a self-check that buried its own
    result under twenty-five of them would not be read.
    """
    import contextlib
    import io

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return call(argv)


def _lzf_stream_is_valid(blob: bytes) -> bool:
    """Walk an LZF stream by the format's own rules: does it end cleanly and never
    reference behind the start of its output?"""
    i, produced = 0, 0
    while i < len(blob):
        control = blob[i]
        i += 1
        if control < 32:
            i += control + 1
            produced += control + 1
        else:
            length = control >> 5
            if length == 7:
                if i >= len(blob):
                    return False
                length += blob[i]
                i += 1
            if i >= len(blob):
                return False
            offset = ((control & 31) << 8 | blob[i]) + 1
            i += 1
            if offset > produced:
                return False
            produced += length + 2
    return i == len(blob)


if __name__ == "__main__":
    sys.exit(main())
