"""The `mainspring` console entry point: build a `QApplication` and a `MainWindow`.

Thin on purpose. Everything this module does -- parse a command line that is at most a
file path, point numba's compiled-kernel cache somewhere that survives a frozen `.exe`
rebuild (seeding it from a build-time pre-warmed copy on first launch), set
pyqtgraph's config, set the application and organisation names that `QSettings` keys
off, construct the window, and hand control to the Qt event loop -- is startup, and
none of it is behaviour worth testing through. What is worth testing goes in
`main_window.py` and below, where a test can reach it with `pytest-qt` and no event
loop.

It is also the PyInstaller entry point (via `packaging/entrypoint.py`), so it stays the
one place a frozen-build workaround goes (lab record, task 07).
"""

from __future__ import annotations

import os
import shutil
import sys

__all__ = ["main"]


def _numba_cache_dir() -> str:
    """A writable, persistent directory for numba's compiled-kernel cache.

    Must be set before numba is imported, which happens lazily and off the GUI thread
    the first time a frame is decoded (`mainspring.uimf.decode`). A frozen `.exe`'s own
    directory is not a candidate -- PyInstaller rebuilds it on every packaging run, and
    an installed copy may not be writable -- so this is a per-user directory outside it
    (lab record, task 07).
    """
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    return os.path.join(base, "mainspring", "numba-cache")


def _seed_numba_cache(cache_dir: str) -> None:
    """Copy a build-time pre-warmed numba cache into `cache_dir`, on a frozen build's
    first launch only.

    Compiling the four decode kernels from nothing costs 2.5-4.7 s, against 0.9-1.4 s
    once an on-disk cache exists (`notes/reader-layer.md`, task 04); that first-launch
    cost cannot be moved into the decode path, only paid somewhere else -- here, once
    per install, instead of by the first researcher who opens a file. Not a candidate
    for `_numba_cache_dir` itself: numba's cache is keyed by target CPU features, and
    a directory that is only ever read (never written back to after a mismatch) would
    silently stop helping the moment this build runs on a different CPU.
    `packaging/mainspring.spec` bundles `packaging/numba_cache_seed/`
    (`tools/warm_numba_cache.py`) as `numba_cache_seed/` beside the frozen app; `sys.frozen`
    is set only by PyInstaller, so a source checkout never looks for it.
    """
    if not getattr(sys, "frozen", False):
        return
    if os.path.isdir(cache_dir) and os.listdir(cache_dir):
        return  # already seeded, or numba has already compiled into it
    bundle_root = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    seed = os.path.join(bundle_root, "numba_cache_seed")
    if os.path.isdir(seed):
        shutil.copytree(seed, cache_dir, dirs_exist_ok=True)


def main(argv: "list[str] | None" = None) -> int:
    """Run the viewer; returns a process exit status.

    An optional path argument opens that file, which is what a file association and a
    drag onto the `.exe` both amount to. `argv` is the arguments after the program name,
    same as `sys.argv[1:]`.
    """
    os.environ.setdefault("NUMBA_CACHE_DIR", _numba_cache_dir())
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    _seed_numba_cache(os.environ["NUMBA_CACHE_DIR"])

    import pyqtgraph as pg
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow
    from .settings import APPLICATION, ORGANISATION

    # Row-major so a `RasterResult.image` (height x width, numpy's own order) needs no
    # transpose; no OpenGL and no antialiasing, since a heatmap redrawn on every wheel
    # tick wants raw speed over smoothing (lab record, task 02).
    pg.setConfigOptions(imageAxisOrder="row-major", useOpenGL=False, antialias=False)

    args = list(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication([sys.argv[0], *args])
    app.setOrganizationName(ORGANISATION)
    app.setApplicationName(APPLICATION)

    window = MainWindow()
    window.show()
    if args:
        window.open_file(args[0])
    return app.exec()
