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

Under construction. The reader layer is written and verified; the viewer and the Windows
installer follow, in that order, and the development record lives in a private companion
repository. There is no release yet.

The reader reproduces every scan's stored total ion current and base peak intensity exactly
on the four files it has been tested against, which were written by four different versions
of PNNL's acquisition software between 2011 and 2026. `uimf-info` reports that comparison on
any file:

```
uv run uimf-info FILE            # parameters, frames, per-frame scan counts
uv run uimf-info FILE --verify   # decode every scan, compare with the stored columns
uv run uimf-info FILE --bench    # time the decode and the rasteriser
```

A file whose blobs do not reproduce its own columns is a file mainspring does not
understand, so `--verify` exits nonzero and names the frames that disagree. Note that
`BPI_MZ` is compared as a tolerance rather than an equality, because the writers compute it
inconsistently.

## Reading a file from Python

```python
from mainspring.uimf import UimfFile

uimf = UimfFile("run.uimf")
frame = uimf.read_frame(uimf.frame_numbers()[0])
bins, intensity = frame.scan(2482)      # one time-of-flight spectrum
arrival = frame.tic()                   # total ion current per scan
```

A frame is held as the points that exist rather than as a dense array of scans by
time-of-flight bins, which on the Bush lab's instrument would be 2.3 GB for a single frame.
`mainspring.uimf` imports without Qt, so a pipeline that installs mainspring for the reader
does not pull a graphical stack onto a machine that has no use for one. Installing the
optional `fast` extra adds numba, which compiles the decoder and is worth a factor of 25.

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
