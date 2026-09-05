"""The `mainspring` console entry point: build a `QApplication` and a `MainWindow`.

Thin on purpose. Everything this module does -- parse a command line that is at most a
file path, point numba's compiled-kernel cache somewhere that survives a frozen `.exe`
rebuild, set pyqtgraph's config, set the application and organisation names that
`QSettings` keys off, construct the window, and hand control to the Qt event loop -- is
startup, and none of it is behaviour worth testing through. What is worth testing goes
in `main_window.py` and below, where a test can reach it with `pytest-qt` and no event
loop.

It is also the PyInstaller entry point, so it stays the one place a frozen-build
workaround would go if the packaged `.exe` ever needs one (lab record, task 07).
"""

from __future__ import annotations

import os
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


def main(argv: "list[str] | None" = None) -> int:
    """Run the viewer; returns a process exit status.

    An optional path argument opens that file, which is what a file association and a
    drag onto the `.exe` both amount to. `argv` is the arguments after the program name,
    same as `sys.argv[1:]`.
    """
    os.environ.setdefault("NUMBA_CACHE_DIR", _numba_cache_dir())
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

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
