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

Version 1.16.0. The reader layer, the viewer, and the Windows installer are written and
verified, and the development record lives in a private companion repository. This release
adds `Data settings > Sum all on open`, which sums every frame of a file as soon as its first
frame is on screen, so a file opened by hand no longer needs `Sum all` pressed after every open.
It is off until you turn it on, and a file opened to follow with `Live` is left to the `Show`
box. The reader and `mainspring.analysis` are unchanged.

Since 1.15.0, `mainspring.analysis`, a package for processing what the reader returns, offers an
ion's arrival-time distribution read from an m/z window, the statistics the Bush lab reads off
such a distribution, and a Gaussian smoother for it (see
[Reading a file from Python](#reading-a-file-from-python)).

Since 1.14.0, the viewer shows a run being acquired one finished frame at a time. A frame's
scans reach the file in arrival-time order, so a frame half written holds about 0.1 per cent of
its signal, and following it as it filled put an empty heat map on screen for most of every
second. `Newest frame` and the chromatogram now wait for each frame to finish, and `View > Show
frames still being written` brings the filling view back.

Since 1.13.0, peaks fall where they should in files written by PNNL's acquisition software.
After a run of 32,768 empty bins, that software writes an extra zero that its own library skips
on reading, and earlier versions of mainspring read the zero as a bin, so every peak after it in the scan
sat one bin too high, up to three bins on the files we have, i.e., about 0.1 at m/z 530 on
SLIMPHONY. Peaks now fall where each scan's stored base peak m/z puts them, whereas
intensities, total ion current, sums and clipping counts are unchanged, as is everything read
from a file mainspring wrote. The reader can also open a finished file without creating
anything beside it, through `UimfFile(path, immutable=True)`, and installing it as a Python
package now asks only for numpy 2.0 and scipy 1.13 or newer rather than the versions
mainspring is developed against. The [user guide](docs/user-guide.md) covers the viewer.

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

## Installing on Windows

The installer is attached to the [latest
release](https://github.com/bushgroup/mainspring/releases/latest) as
`mainspring-<version>-setup.exe`, about 94 MB, and needs no Python installation. It installs
into the current user's profile and asks for no administrator rights, offers a Start menu entry
and an optional desktop icon, and offers to open `.uimf` files with mainspring. The same release
carries `mainspring-<version>-py3-none-any.whl`, which is the reader as a Python package for a
machine that already has Python 3.12. To install the viewer from it as well, ask for the
`viewer` extra, i.e., `pip install "mainspring-<version>-py3-none-any.whl[viewer]"`. Both file
names carry the version of the release they are attached to, so follow the link rather than
typing one.

The installer is not code signed, so Microsoft Defender SmartScreen shows "Windows protected
your PC" the first time it is run from a download. Choose More info, then Run anyway. Signing
would need a certificate the lab does not hold.

## Using the viewer

The [user guide](docs/user-guide.md) covers the window, every gesture and shortcut, every
toolbar control, and how to read the per-push intensity the info panel reports. From a source
checkout the viewer starts with `uv run mainspring`, and takes an optional file to open:

```
uv run mainspring
uv run mainspring FILE.uimf
uv run mainspring FILE.uimf --follow --show newest
```

To open a file already following it, which is how another program starts the viewer on a
run it is writing, pass `--follow` and `--show`. `--show` chooses what following does with
each new frame and takes `fixed`, `newest`, `method-sum` or `rolling-sum`, the four entries
of the viewer's own `Show` box. A file that cannot be followed, one on a network drive,
opens anyway and the status bar says why it is not being followed. An option the viewer
does not offer is an error rather than a file by that name, and the process exits 2 without
opening a window.

The `Live` control in the window needs no file name. It finds the run being acquired, from a
pointer the acquisition software publishes or from the newest growing acquisition in the folder
the open file came from, opens it and follows it, and moves to the next run when one starts.

## Reading a file from Python

```python
from mainspring.uimf import UimfFile

uimf = UimfFile("run.uimf")
number = uimf.frame_numbers()[0]
frame = uimf.read_frame(number)
bins, intensity = frame.scan(2482)      # one time-of-flight spectrum
arrival = frame.tic()                   # total ion current per scan
```

A frame is held as the points that exist rather than as a dense array of scans by
time-of-flight bins, which on the Bush lab's instrument would be 2.3 GB for a single frame.
To use the reader from another project, install mainspring from a release tag:

```
pip install "mainspring[fast] @ git+https://github.com/bushgroup/mainspring@v1.16.0"
```

The same requirement string works as a dependency in another project's `pyproject.toml`,
under pip and uv alike. Neither `mainspring.uimf` nor the installation it comes from carries
Qt, so a pipeline that installs mainspring for the reader does not pull a graphical stack onto
a machine that has no use for one. The `fast` extra adds numba, which compiles the decoder and
the encoder: reading a frame becomes 25 times faster and encoding one 80 times faster, with
the same bytes either way. The `viewer` extra adds the viewer and includes `fast`. Without it
the `mainspring` command prints one line naming `mainspring[viewer]` and exits with status 3.

### Reading an ion's arrival-time distribution

To follow one ion through a run, `mainspring.analysis` sums the intensity of every stored point
inside an m/z window, scan by scan, over the frames it is given:

```python
from mainspring.analysis import summed_profile

frames = uimf.frame_numbers()[1:]                       # which frames count is the caller's call
atd = summed_profile(uimf, frames, (1418.5, 1438.5))    # m/z window, both ends included
atd.values                                              # intensity per scan
atd.axis_ms                                             # arrival time of each scan, in ms
```

Each frame is placed through its own calibration and decoded only over the window's bins, so a
file whose calibration changes from frame to frame is summed correctly and no frame is ever held
whole. The window holds every bin whose m/z lies between its two ends, ends included, and
`bin_range` returns those bins as a half-open range for a caller that reads frames itself, whereas
`scan_profile` gives the distribution of one frame already in memory. Which frames make up a run
is left to the caller, because leaving out a first repetition, choosing a method frame through
`FrameGrouping`, or counting time from a release mark are each the convention of one experiment,
and the package sums exactly the frames it is handed. Note that `mainspring.uimf.profile` is a
different function, i.e., the viewer's side plot, summed over whatever is on screen.

### Smoothing an arrival-time distribution

To draw an arrival-time distribution without the ringing that a low-pass filter puts around a
sharp leading edge, `mainspring.analysis` smooths it with a Gaussian kernel:

```python
from mainspring.analysis import gaussian_gain, gaussian_smooth

tick_ms = uimf.frame_params(number).average_tof_length_ns * 1e-6   # one scan, in ms
smoothed = gaussian_smooth(arrival, tick_ms=tick_ms, sigma_ms=0.6)
gaussian_gain(954.0, sigma_ms=0.6)                                 # 0.0016
```

The kernel is a normal distribution of width σ, sampled once per scan and truncated at ±5σ.
Because it is never negative, a step comes out as the normal cumulative distribution, rising from
10 to 90 per cent in 2.56σ with no undershoot ahead of the edge and no overshoot after it, and a
distribution with no negative intensity gains none. The zero-phase Butterworth filter at 200 Hz
that it replaces in the Bush lab's figures put negative intensity ahead of every leading edge.
Both σ and the scan interval are given in milliseconds, so a file acquired at 62 µs per scan and
one acquired at 129 µs receive the same kernel in time. Neither has a default, because the right
width depends on the figure and the data, and a default would be a width that nobody chose.
Beyond either end the distribution is taken to be zero, i.e., zero padding, since a frame begins
before the ions arrive and reflecting the end back would invent counts.

`gaussian_gain` returns the factor by which the smoother multiplies a sinusoid of frequency f,
exp(−2π²σ²f²) with σ in seconds. It is the continuous kernel's response, and it matches the
sampled kernel's to within 10⁻⁶ at every frequency up to Nyquist whenever σ spans at least two
scans.

### Reading positions and widths off a distribution

To report where a distribution peaks and how wide it is, the package reads seven statistics off
any profile, each in scans:

```python
from mainspring.analysis import centroid, fwhm, local_maxima

smoothed = gaussian_smooth(atd.values, tick_ms=atd.tick_ms, sigma_ms=0.6)
atd.ms(centroid(smoothed))                  # arrival time of the peak, in ms
atd.ms(fwhm(smoothed))                      # its full width at half maximum, in ms
local_maxima(smoothed, fraction=0.15)       # every maximum above 15 per cent of the tallest
```

`argmax` is the scan of the largest value, `centroid` the intensity-weighted mean scan over the
contiguous run around it that stays at or above half its height, and `median` the scan at which
the cumulative intensity reaches half the total. `iqr` is the span holding the central half of
the intensity and `fwhm` the full width at half the maximum, 1.349σ and 2.355σ respectively for a
Gaussian, and `rise_10_90` is the leading edge from 10 to 90 per cent of the maximum, 2.563σ for a
step smoothed by a Gaussian. None of the seven smooths, because one repetition of a CLOCK
experiment holds a few hundred stored points and the smoother and its width belong to the
caller, so the order is smooth first and then read. `local_maxima` takes its floor as a required
fraction for the reason the smoother takes its width without a default. A profile with nothing in
it gives `None` from every statistic, and `ScanProfile.ms` passes `None` through unchanged.

Every name in `mainspring.analysis` is published. A name is frozen in the self-check once a
program outside this repository imports it at a release tag; until then its signature may change
in a minor release, and the release notes say so. `gaussian_smooth` and `gaussian_gain` are frozen
from 1.15.0, whereas the profile read and the seven statistics join the frozen list when a program
outside this repository first imports them.

## Requirements

- Windows 11, on which the viewer and its installer are tested.
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
PyInstaller and the viewer's own libraries are in the `dev` dependency group that `uv sync`
installs. From a PowerShell prompt
at the root of the checkout:

```
tools\build_exe.ps1
```

The script runs PyInstaller against `packaging/mainspring.spec`, has the built program compile
its numba kernels into a cache that the build carries, and then launches the result twice and
reports how long each launch took to put a window on screen. The cache has to come from the
built program itself, because numba recognises a packaged program's compiled kernels by the
executable's size and modification time, and the installer keeps both. The build lands in `dist\mainspring\`, a folder of
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

Compilation takes about two minutes and writes `dist\installer\mainspring-1.1.0-setup.exe`,
93 MB. That installer is per-user and asks for no administrator rights, because the viewer keeps
its settings in the current user's registry hive and an instrument PC's operator account may
have no administrator rights to give. It offers a Start menu entry, an optional desktop icon,
and an uninstaller. The version in the file name comes from `MyAppVersion` in
`packaging/mainspring.iss`. Three files declare the version and none derives it from another, so
`tools/check_public.py` fails unless `MyAppVersion`, the version in `pyproject.toml`, and
`mainspring.__version__` all agree.

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
src/mainspring/analysis/ processing what the reader returns: an ion's arrival-time distribution,
                         the statistics read off it, and the arrival-time smoother
src/mainspring/viewer/   the PySide6 + pyqtgraph application
tools/                   check_public.py (the self-check), fetch_testdata.py (PNNL excerpts),
                         build_exe.ps1 (the Windows executable), make_icon.py (its icon),
                         write_commit.py (the commit a build records in itself)
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
