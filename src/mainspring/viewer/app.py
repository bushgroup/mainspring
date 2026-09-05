"""The `mainspring` console entry point: build a `QApplication` and a `MainWindow`.

Thin on purpose. Everything this module does -- parse a command line that is at most a
file path, set the application and organisation names that `QSettings` keys off, apply
the high-DPI and colour-scheme policy, construct the window, and hand control to the Qt
event loop -- is startup, and none of it is behaviour worth testing through. What is
worth testing goes in `main_window.py` and below, where a test can reach it with
`pytest-qt` and no event loop.

It is also the PyInstaller entry point, so it stays the one place a frozen-build
workaround would go if the packaged `.exe` ever needs one (lab record, task 07).
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv: "list[str] | None" = None) -> int:
    """Run the viewer; returns a process exit status.

    An optional path argument opens that file, which is what a file association and a
    drag onto the `.exe` both amount to. Arrives with the lab record's task 04.
    """
    raise NotImplementedError("the viewer application arrives with the lab record's task 04")
