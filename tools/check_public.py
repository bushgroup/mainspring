"""Self-check for a fresh clone: no data file, no lab repo, no instrument needed.

Every check here must pass in a bare public clone. Checks that need something a clone
does not ship -- PNNL's test excerpts in `external/`, a file named by
`MAINSPRING_SMOKE_UIMF`, the lab repository -- are reported as SKIPPED when it is
absent, never as FAIL. What is left is still a real test of the decode path, because a
synthetic UIMF file can be written from nothing: `tests/synthetic.py` puts a few hundred
known points through the real SQLite schema and the real intensity encoder.

What it covers: the package imports, the `uimf` layer stays free of Qt, the three
hand-carried version declarations agree, the module layout is complete, the reporting stamp, the lab-directory resolution, the intensity
codec against the format's own rules and against itself in both directions, and a
synthetic file -- written through the schema a 2026 acquisition carries -- read back
through the whole reader, rasterised, and put through `uimf-info --verify`; and, since
task 04, the same synthetic file opened and painted by the viewer's own window,
offscreen, alongside the window icon that ships with it. Where a real file is present, `--verify` runs on that too, which is the
acceptance test the milestone is written in terms of (lab record, task 03).

Run:  uv run tools/check_public.py
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import tempfile
import time
import tomllib

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


def declared_versions() -> dict[str, str]:
    """The version as each of the three files that hand-carry it states it.

    Nothing derives one of these from another: the package literal is what an
    import reports, `pyproject.toml` is what a wheel is built as, and the Inno
    Setup define is what the installer calls itself and names its own file. They
    only agree because someone keeps them agreeing, which is why this is checked
    rather than trusted.
    """
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
        pyproject = tomllib.load(handle)["project"]["version"]
    iss = open(os.path.join(ROOT, "packaging", "mainspring.iss"), encoding="utf-8").read()
    found = re.search(r'^#define\s+MyAppVersion\s+"([^"]+)"', iss, re.MULTILINE)
    return {"pyproject.toml": pyproject,
            "packaging/mainspring.iss": found.group(1) if found else "(not found)"}


UIMF_MODULES = ("cache", "calib", "cli", "decode", "frame", "raster", "reader")
VIEWER_MODULES = (
    "app", "controls", "export", "heatmap", "info_panel", "main_window", "settings",
    "side_plots", "theme", "workers",
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
    declared = declared_versions() | {"mainspring.__version__": mainspring.__version__}
    check_true(
        "the package, the wheel and the installer declare one version ("
        + ", ".join(f"{where} {what}" for where, what in declared.items()) + ")",
        len(set(declared.values())) == 1,
    )
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
    # must not see. QT_QPA_PLATFORM is set before that import, offscreen, so this runs
    # with no display -- this workstation has one, but CI and a bare clone may not.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    for name in VIEWER_MODULES:
        check_true(
            f"mainspring.viewer.{name} imports",
            importlib.import_module(f"mainspring.viewer.{name}") is not None,
        )

    from PySide6.QtWidgets import QApplication

    from mainspring.viewer import theme
    from mainspring.viewer.controls import unexplained
    from mainspring.viewer.main_window import MainWindow
    from mainspring.viewer.settings import THEMES

    qt_app = QApplication.instance() or QApplication([])

    # The icon is package data, so a clone that has it can check it: that it is there at
    # all (the .exe, the installer and every window take it from this one file), that Qt
    # can read every frame out of the container this repo writes by hand, and that it
    # still matches the drawings it came from -- an edit to packaging/icon/*.svg without
    # a `uv run tools/make_icon.py` after it would otherwise ship the old picture.
    import make_icon
    import pyqtgraph as pg
    from PySide6.QtGui import QIcon, QImage

    from mainspring.viewer.app import _icon_path

    check_true("the window icon ships with the package", os.path.isfile(_icon_path()))
    icon_sizes = {s.width() for s in QIcon(_icon_path()).availableSizes()}
    check_true(
        f"the icon carries every frame Windows asks for (has {sorted(icon_sizes)})",
        icon_sizes == {size for size, _ in make_icon.FRAMES},
    )
    check_true(
        "the icon is up to date with packaging/icon/",
        open(_icon_path(), "rb").read() == make_icon.build(),
    )

    # The heatmap's logo mark is the same kind of shipped-copy-of-a-drawing as the icon,
    # just SVG rasterised at runtime instead of pre-baked into an .ico: `heatmap._resource`
    # resolves it beside the module the same way `_icon_path` resolves the .ico, and an
    # edit to packaging/icon/mainspring.svg without copying it into
    # src/mainspring/viewer/resources/ would otherwise ship the old mark.
    from mainspring.viewer.heatmap import _resource

    shipped_logo = _resource("mainspring.svg")
    source_logo = os.path.join(ROOT, "packaging", "icon", "mainspring.svg")
    check_true(
        "mainspring.svg ships with the package and matches packaging/icon/",
        os.path.isfile(shipped_logo)
        and open(shipped_logo, "rb").read() == open(source_logo, "rb").read(),
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "viewer.uimf")
        viewer_spec = write_synthetic_uimf(path, frames=1, scans=16, bins=4096)
        window = MainWindow()
        painted: list = []
        window.frame_shown.connect(painted.append)
        window.open_file(path)

        deadline = time.time() + 5.0
        while not painted and time.time() < deadline:
            qt_app.processEvents()
            time.sleep(0.01)
        check_true("the viewer window loads a synthetic file and paints a frame",
                   bool(painted))

        # Every control the user can touch explains itself. Data-free, so this FAILs in
        # a bare clone rather than skipping: a rule the pytest suite alone enforced is
        # not the rule CLAUDE.md claims. Run with a file open, because opening one
        # repopulates the frame-type filter and the parameter tree.
        mute = unexplained(window)
        check_true(
            "every control the user can touch explains itself"
            + (f" (mute: {', '.join(mute)})" if mute else ""),
            not mute,
        )

        # And nothing on the plot canvas is painted a colour its palette does not hold.
        # Under **light** above all: under the dark palette this passes for anything
        # hardcoded to the values the viewer has always drawn, which is most of what the
        # walk exists to catch (`viewer/theme.py`). Data-free, so it FAILs in a bare
        # clone rather than skipping, and the theme is put back afterwards because
        # `theme.apply` sets pyqtgraph's process-wide config options and the export
        # checks below read the canvas background back.
        for name in THEMES:
            theme.apply(window, name)
            stray = theme.themed(window)
            check_true(
                f"nothing on the {name} canvas is painted outside its palette"
                + (f" (stray: {', '.join(stray)})" if stray else ""),
                not stray,
            )
        theme.apply(window, window.settings.theme)
        if painted:
            result = painted[0]
            check_true("the painted image's extent matches the calibrated full range",
                       (result.x_range, result.y_range) == result.axes.full_range)
            check_true("the painted image conserves the frame's total ion current",
                       abs(result.tic_in_view - viewer_spec.tic(1)) <= 1e-6 * max(1.0, viewer_spec.tic(1)))
            # The side plots must describe the image, not a window it has moved on from.
            profiles = window.last_render
            check_true(
                "the side plots project the same points the image does",
                all(
                    abs(float(values.sum()) - result.tic_in_view)
                    <= 1e-6 * max(1.0, result.tic_in_view)
                    for _, values in (profiles.x_profile, profiles.y_profile)
                ),
            )

            # File > Export: the figure a user actually takes away. Data-free, so it
            # runs in a bare clone -- and it checks the two things a screenshot would
            # not, that the colour bar's column is outside the rectangle rendered and
            # that the heatmap was resampled for the export rather than upscaled.
            from mainspring.viewer.export import (
                BASE_DPI, content_rect, export_display, export_pixels,
            )

            rect = content_rect(window.heatmap, window.side_plots)
            bar = window.heatmap.scene().items()
            colour_bar = [i for i in bar if isinstance(i, pg.ColorBarItem)]
            check_true(
                "the exported rectangle leaves the colour bar out",
                bool(colour_bar)
                and not rect.intersects(colour_bar[0].mapRectToScene(colour_bar[0].boundingRect())),
            )
            screen_columns = window.heatmap.image_item.image.shape[1]
            figure = os.path.join(tmp, "figure.png")
            dpi = 3 * int(BASE_DPI)
            size = export_display(
                window.heatmap, window.side_plots, window._current_frame,
                window.last_render.result, window.settings.colour_scale,
                figure, "png", dpi,
            )
            written = QImage(figure)
            check_true(
                f"File > Export writes a PNG at the size it promised ({size[0]}x{size[1]})",
                size == export_pixels(rect, dpi) == (written.width(), written.height()),
            )
            check_true(
                "the export resamples the heatmap rather than upscaling the screen's",
                window.heatmap.image_item.image.shape[1] == screen_columns,
            )

            # And a gesture must reach the render worker and come back with a narrower
            # window: the whole interactive path, in one check, through the same signal
            # a wheel tick travels (lab record, task 05).
            (x0, x1), (y0, y1) = result.axes.full_range
            zoom = (x0 + 0.25 * (x1 - x0), x0 + 0.5 * (x1 - x0))
            painted.clear()
            window.heatmap.view_box.setRange(xRange=zoom, yRange=(y0, y1), padding=0.0)
            deadline = time.time() + 5.0
            while not painted and time.time() < deadline:
                qt_app.processEvents()
                time.sleep(0.01)
            check_true("a view change repaints through the render worker", bool(painted))
            if painted:
                check_true(
                    "the repainted image is the window that was asked for",
                    abs(painted[0].x_range[0] - zoom[0]) <= 1e-9 * max(1.0, abs(zoom[0]))
                    and painted[0].tic_in_view < result.tic_in_view,
                )
        window.close()

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

    # --------------------------------------------------------------------------------
    section("the wheel (uv build, clean-venv install)")
    # Unlike the in-process import check above, this builds the wheel and installs it
    # in a subprocess venv, so import order here cannot leak Qt into this process. What
    # it catches instead is a packaging mistake the in-process check cannot see: a file
    # `hatchling` left out of the wheel, or a dependency the wheel declares wrong.
    # Building and installing needs `uv` and a venv it can create; only that second
    # part is reported SKIPPED rather than FAILED, since a sandboxed or offline machine
    # can otherwise run every check above (lab record, task 07).
    import shutil
    import subprocess

    uv = shutil.which("uv")
    if uv is None:
        skip("wheel build and clean-venv install", "uv is not on PATH")
    else:
        with tempfile.TemporaryDirectory() as wheel_tmp:
            wheel_dir = os.path.join(wheel_tmp, "dist")
            build = subprocess.run(
                [uv, "build", "--wheel", "--out-dir", wheel_dir, ROOT],
                capture_output=True, text=True,
            )
            if build.returncode != 0:
                check_true("uv build produces a wheel", False)
                print(build.stdout[-2000:])
                print(build.stderr[-2000:])
            else:
                wheels = sorted(f for f in os.listdir(wheel_dir) if f.endswith(".whl"))
                check_true(f"uv build produces exactly one wheel ({wheels})", len(wheels) == 1)

                venv_dir = os.path.join(wheel_tmp, "venv")
                venv_result = subprocess.run(
                    [uv, "venv", venv_dir, "--no-project"], capture_output=True, text=True,
                )
                if venv_result.returncode != 0:
                    skip("wheel clean-venv install",
                         f"uv could not create a venv ({venv_result.stderr.strip()[:200]})")
                elif wheels:
                    venv_python = os.path.join(
                        venv_dir,
                        "Scripts" if os.name == "nt" else "bin",
                        "python.exe" if os.name == "nt" else "python",
                    )
                    wheel_path = os.path.join(wheel_dir, wheels[0])
                    install = subprocess.run(
                        [uv, "pip", "install", "--python", venv_python, wheel_path],
                        capture_output=True, text=True,
                    )
                    check_true("the wheel installs into a clean venv", install.returncode == 0)
                    if install.returncode != 0:
                        print(install.stdout[-2000:])
                        print(install.stderr[-2000:])
                    else:
                        probe = subprocess.run(
                            [venv_python, "-c",
                             "import sys, mainspring.uimf\n"
                             "qt = [m for m in sys.modules if m.startswith(('PySide6', 'pyqtgraph', 'shiboken6'))]\n"
                             "print('QT:' + ','.join(qt) if qt else 'NOQT')"],
                            capture_output=True, text=True,
                        )
                        check_true(
                            "the installed wheel imports mainspring.uimf with no Qt loaded "
                            f"({probe.stdout.strip() or probe.stderr.strip()[-200:]})",
                            probe.returncode == 0 and probe.stdout.strip() == "NOQT",
                        )

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
