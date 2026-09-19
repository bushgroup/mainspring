"""The `mainspring` console entry point: build a `QApplication` and a `MainWindow`.

Thin on purpose. Everything this module does -- parse a command line that is a file path
and how to look at it, point numba's compiled-kernel cache somewhere that survives a
frozen `.exe` rebuild (seeding it from a build-time pre-warmed copy on first launch), set
pyqtgraph's config, set the application and organisation names that `QSettings` keys
off, set the window icon and the Windows taskbar identity that groups under it,
construct the window, and hand control to the Qt event loop -- is startup, and
none of it is behaviour worth testing through. What is worth testing goes in
`main_window.py` and below, where a test can reach it with `pytest-qt` and no event
loop.

It is also the PyInstaller entry point (via `packaging/entrypoint.py`), so it stays the
one place a frozen-build workaround goes (lab record, task 07).

**The options are an interface to another program, not to a person.** The clockwork
acquisition window starts the viewer on the run it is writing, and what it passes has to
open that file already following it -- a trainee should not have to find `Follow` and
`Show` on a window that has just appeared (lab record, task 26). So the words are long
options with stable names and values a person could have typed, an unrecognised one is
an error rather than a path, and the whole parse happens **before** `QApplication` is
constructed, since Qt eats the options it recognises out of any list it is handed.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

__all__ = ["Launch", "main", "parse_arguments"]

USAGE = """usage: mainspring [FILE] [--follow] [--show MODE]

Open a UIMF file in the mainspring viewer.

  FILE           the .uimf file to open
  --follow       watch the file for what the instrument is still writing to it
  --show MODE    what following does with each new frame: fixed, newest,
                 method-sum or rolling-sum. Implies --follow
  -h, --help     print this and exit
"""
"""What `--help` prints, and what an unrecognised option is answered with.

Hand-written, and the parse below with it, rather than `argparse`: the frozen build is
windowed, so `sys.stdout` and `sys.stderr` may be nothing at all, and `argparse`
answers a bad option by writing to a stream it assumes is there and raising `SystemExit`
through a caller that is a `main()` returning a status. A dozen lines of parsing keep
both of those under this module's control."""


@dataclass(frozen=True)
class Launch:
    """What one command line asked for: a file, and how to look at it."""

    path: "str | None" = None
    follow: bool = False
    show: "str | None" = None
    """The `--show` word as it was typed (`mainspring.interface.SHOW_WORDS`), or None to
    leave the box where it is. The word and not the `Show` label it names: translating
    the two is the window's job and needs the window (`main_window.FOLLOW_MODE_WORDS`),
    and this dataclass is what a parse produces before there is one."""


def parse_arguments(args: "list[str]") -> Launch:
    """`args` as a `Launch`, raising `ValueError` on anything this does not offer.

    One optional path and long options only. A refusal rather than a guess is the whole
    point of parsing at all: an unrecognised `--folow` taken as a positional argument
    would be a file by that name, and the viewer would put up a dialog about a file
    nobody asked for instead of failing where the mistake is.

    `--show` is given a word rather than a label -- `rolling-sum`, not
    `Sum newest frames` -- because the caller is another program and the labels are what
    this window happens to say today. The words come from `mainspring.interface`, which
    imports no Qt, so this whole parse runs without one: the program on the other end of
    them cannot import a GUI either, and a command line that only one side can read is
    not an interface (lab record, task 27).
    """
    from ..interface import OPTION_FOLLOW, OPTION_SHOW, SHOW_WORDS

    path: "str | None" = None
    follow = False
    show: "str | None" = None
    rest = list(args)
    while rest:
        arg = rest.pop(0)
        if arg == OPTION_FOLLOW:
            follow = True
        elif arg == OPTION_SHOW or arg.startswith(OPTION_SHOW + "="):
            word = arg[len(OPTION_SHOW) + 1:] if "=" in arg else (rest.pop(0) if rest else "")
            if word not in SHOW_WORDS:
                offered = ", ".join(SHOW_WORDS)
                raise ValueError(f"{OPTION_SHOW} takes one of {offered}, not {word!r}")
            show = word
        elif arg.startswith("-") and arg != "-":
            raise ValueError(f"unrecognised option {arg}")
        elif path is None:
            path = arg
        else:
            raise ValueError("only one file can be opened at a time")
    return Launch(path=path, follow=follow, show=show)


def _write(stream: object, text: str) -> None:
    """Print to `stream` if there is one.

    The frozen build is windowed (`packaging/mainspring.spec`, `console=False`), where
    the standard streams may be `None` or a handle nothing is attached to. A usage
    message that cannot be shown is not a reason for the exit status to be wrong, and it
    is certainly not a reason to raise.
    """
    try:
        stream.write(text)
        stream.flush()
    except (AttributeError, OSError, ValueError):
        pass


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


def _icon_path() -> str:
    """The multi-resolution `.ico`, which travels with the package.

    `os.path.dirname(__file__)` resolves in a frozen build as well as a source tree:
    PyInstaller sets `__file__` to the bundled module's path, and
    `packaging/mainspring.spec` collects the package's data files beside it.
    `tools/make_icon.py` generates the file from the drawings in `packaging/icon/`.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "mainspring.ico")


def _set_windows_app_id() -> None:
    """Tell the Windows shell this process is mainspring rather than its host executable.

    Without an explicit AppUserModelID the shell groups a window under whatever launched
    it and shows that program's icon in the taskbar -- for a source run, the interpreter's.
    Setting one costs nothing when the icon is right anyway (the frozen `.exe` carries it
    in its resources) and fixes the case where it is not. Everything here is Windows-only
    and best-effort: nothing about the viewer depends on it working.
    """
    if sys.platform != "win32":
        return
    import ctypes

    from .settings import APPLICATION, ORGANISATION

    app_id = f"{ORGANISATION}.{APPLICATION}".replace(" ", "")
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except (AttributeError, OSError):  # older shell32, or a Windows without it
        pass


def main(argv: "list[str] | None" = None) -> int:
    """Run the viewer; returns a process exit status.

    An optional path argument opens that file, which is what a file association and a
    drag onto the `.exe` both amount to; `--follow` and `--show` say how to look at it,
    which is what another program starting the viewer on a run in progress asks for.
    `argv` is the arguments after the program name, same as `sys.argv[1:]`.

    Three statuses: 0 when the viewer ran (and when `--help` was all that was wanted),
    2 for a command line this cannot act on, and whatever Qt's event loop returns
    otherwise. A file that cannot be followed is **not** one of those -- it opens, the
    status bar says why it is not being followed, and the process exits 0, because the
    file is what the person was trying to see.
    """
    os.environ.setdefault("NUMBA_CACHE_DIR", _numba_cache_dir())
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    _seed_numba_cache(os.environ["NUMBA_CACHE_DIR"])

    _set_windows_app_id()

    import pyqtgraph as pg
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from .main_window import FOLLOW_MODE_WORDS, MainWindow
    from .settings import APPLICATION, ORGANISATION

    # Row-major so a `RasterResult.image` (height x width, numpy's own order) needs no
    # transpose; no OpenGL and no antialiasing, since a heatmap redrawn on every wheel
    # tick wants raw speed over smoothing (lab record, task 02).
    pg.setConfigOptions(imageAxisOrder="row-major", useOpenGL=False, antialias=False)

    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args:
        _write(sys.stdout, USAGE)
        return 0
    try:
        launch = parse_arguments(args)
    except ValueError as exc:
        _write(sys.stderr, f"mainspring: {exc}\n\n{USAGE}")
        return 2

    # Qt strips the options it recognises out of whatever list it is given, so it is
    # handed the program name alone: everything after it has been read above, and a
    # `-style` or a `-platform` meant for Qt is not something this viewer's callers pass.
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setOrganizationName(ORGANISATION)
    app.setApplicationName(APPLICATION)
    app.setWindowIcon(QIcon(_icon_path()))  # inherited by every window the viewer opens

    window = MainWindow()
    window.show()
    if launch.path is not None:
        # Asked for before the open rather than after it, because the open is
        # asynchronous: the window applies this when the file is on screen and the frame
        # list is known (`MainWindow.follow_when_opened`). A `--show` on its own asks
        # for following too, since the mode it names is what following does. The word
        # becomes a `Show` label here, where the window exists: the parse above deals in
        # words alone so that it needs no Qt.
        if launch.follow or launch.show is not None:
            show = None if launch.show is None else FOLLOW_MODE_WORDS[launch.show]
            window.follow_when_opened(show=show)
        # A file association or a drag onto the .exe both arrive here the same way: a
        # path Windows chose on the user's behalf, not a dialog they were sitting in
        # front of. `open_file` uses the distinction to decide how loudly a bad open
        # complains (lab record, task 15).
        window.open_file(launch.path, from_command_line=True)
    return app.exec()
