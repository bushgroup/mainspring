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

## Using the viewer

The [user guide](docs/user-guide.md) covers the window, every gesture and shortcut, every
toolbar control, and how to read the per-push intensity the info panel reports. From a source
checkout the viewer starts with `uv run mainspring`, and takes an optional file to open:

```
uv run mainspring
uv run mainspring FILE.uimf
```

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

- Windows 10 or 11. The viewer will also be distributed as an installer that needs no Python
  installation.
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

## Building the executable

Building the Windows executable needs the fresh-clone steps above and nothing more, since
PyInstaller is in the `dev` dependency group that `uv sync` installs. From a PowerShell prompt
at the root of the checkout:

```
tools\build_exe.ps1
```

The script compiles the numba kernels into a cache that the build carries, runs PyInstaller
against `packaging/mainspring.spec`, and then launches the result twice and reports how long
each launch took to put a window on screen. The build lands in `dist\mainspring\`, a folder of
about 350 MB holding `mainspring.exe` beside the Qt and numpy libraries it loads. A folder is
the deliverable rather than one self-extracting file because a self-extracting file unpacks its
payload to a temporary directory on every launch, which ran past 180 s with no window against
the folder's 3 to 6 s on a fast workstation. Pass `-SkipBuild` to time a build that already
exists.

The script also cuts `PATH` down to the Windows directories and uv's own for the duration of the
build, and the spec refuses to build if any bundled file resolves outside the virtual
environment, the interpreter it was created from, or this repository. Both guard the same
failure: PyInstaller resolves a library's dependencies through `PATH` as a last resort, so a
second Python distribution there is bundled in place of the intended copy. One build made that
way shipped another distribution's ICU library and died importing `QtCore` on every machine it
was installed on.

## Creating the installer

Creating the installer needs Inno Setup 6, which is not part of the Python environment:

```
winget install JRSoftware.InnoSetup --scope user
```

The installer script packages `dist\mainspring\`, so build the executable first. Then, from the
root of the checkout:

```
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" packaging\mainspring.iss
```

Compilation takes about two minutes and writes `dist\installer\mainspring-0.1.0-setup.exe`,
93 MB. That installer is per-user and asks for no administrator rights, because the viewer keeps
its settings in the current user's registry hive and an instrument PC's operator account may
have no administrator rights to give. It offers a Start menu entry, an optional desktop icon,
and an uninstaller. The version in the file name comes from `MyAppVersion` in
`packaging/mainspring.iss`, which is kept in step with the version in `pyproject.toml` by hand.

The wizard also offers, checked by default, to open `.uimf` files with mainspring. On a machine
where nothing else has claimed the extension, which is the ordinary case for an instrument PC,
accepting it means a double-click opens the file directly, with no prompt and no reboot.
Windows keeps a choice you have already made for an extension, though, so if something else was
previously set as the `.uimf` handler, mainspring registers itself as a choice rather than
taking the slot back; Settings > Default apps is where to point `.uimf` at it. Uninstalling
removes the registration.

## Layout

```
src/mainspring/uimf/     the reader: SQLite access, blob decoding, calibration, sparse frames,
                         rasterisation, and the uimf-info command line
src/mainspring/viewer/   the PySide6 + pyqtgraph application
tools/                   check_public.py (the self-check), fetch_testdata.py (PNNL excerpts),
                         build_exe.ps1 (the Windows executable), make_icon.py (its icon)
tests/                   pytest suite; tests/fixtures/README.md says what is synthetic
packaging/               PyInstaller specification, Inno Setup script, icon artwork
docs/                    user documentation; docs/user-guide.md is the viewer's
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
