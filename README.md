# mainspring

mainspring is an interactive viewer for UIMF files, the SQLite-based format in which SLIM
ion mobility mass spectrometers record their data, and a Python reader for the same files
that analysis pipelines can import on their own. The viewer shows a frame as a heat map of
m/z against arrival time with the mass spectrum and the arrival-time distribution of the
visible region beside it, zooms and pans in one gesture, keeps the current ranges when the
next file is opened, and reports the peak intensity per time-of-flight push so that detector
saturation can be judged while the instrument is still running. It is written for the Bush
lab's SLIMPHONY instrument and its CLOCK experiments, and for any other laboratory whose
instrument writes UIMF.

## Status

Under construction. The reader layer, the viewer, and the Windows installer are being built
in that order; the development record lives in a private companion repository. There is no
release yet.

## Requirements

- Windows 10 or 11. The viewer will also be distributed as a single executable that needs no
  Python installation.
- [uv](https://docs.astral.sh/uv/) for working from source. The interpreter (CPython 3.12)
  and every library version are pinned by `pyproject.toml`, `.python-version`, and the
  committed `uv.lock`.

## Fresh clone

```
git clone https://github.com/bushgroup/mainspring
cd mainspring
git config core.hooksPath .githooks
uv sync                          # creates .venv/ from uv.lock
uv run tools/check_public.py     # data-free self-check
```

The self-check passes with no data file present: the checks that need one report themselves
skipped rather than failed. To add them, fetch the small test excerpts that PNNL distributes
with its UIMF library:

```
uv run tools/fetch_testdata.py
```

`uv run <script>` needs no flags and no venv activation. Use `uv run` or activate `.venv/`
rather than a bare `python` on `PATH`.

`git config core.hooksPath .githooks` is per-clone and not tracked by git. Without it,
commits do not receive the `Assisted-by:` trailer rewrite, and nothing refuses an oversized
file or a `.uimf` file entering history.

## Layout

```
src/mainspring/uimf/     the reader: SQLite access, blob decoding, calibration, sparse frames,
                         rasterisation, and the uimf-info command line
src/mainspring/viewer/   the PySide6 + pyqtgraph application
tools/                   check_public.py (the self-check), fetch_testdata.py (PNNL excerpts),
                         build_exe.ps1 (the Windows executable)
tests/                   pytest suite; tests/fixtures/README.md says what is synthetic
packaging/               PyInstaller specification and installer files
docs/                    user documentation
external/                fetched test data, read in place and never committed (not in this repository)
```

## Development record

The task history, working notes, and exploration scripts live in a private companion
repository. When it is present as a sibling checkout, or is named by the `MAINSPRING_LAB`
environment variable, `mainspring.lab_dir()` resolves it; without it the package is fully
functional.

## License

BSD 3-Clause, in [`LICENSE`](LICENSE). UIMF is a format published by Pacific Northwest
National Laboratory; its reference implementation, UIMF-Library, is separately licensed by
PNNL and is not redistributed here.
