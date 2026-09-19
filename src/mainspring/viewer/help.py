"""The Help menu: the user guide as it shipped, the user guide as it stands, and About.

Three entries answering the two questions a window on an instrument PC raises first --
how do I use this, and which version is it -- without either answer depending on the
network. The guide is `docs/user-guide.md`, the one source, carried into the build as a
data file; `guide_path()` finds it whether the program is a checkout or the frozen
onedir, and `read_guide()` is what the reader window shows.

**Rendered by Qt, not converted.** `QTextBrowser.setMarkdown` reads the file as it is
written, so there is no second form of the guide to keep in step and no dependency to
add. What it does not render as richly as a browser would is worth less than what a
second copy would cost.
"""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
)

from .. import ROOT, __version__
from ..report import mainspring_commit
from .controls import describe

__all__ = [
    "GUIDE_URL",
    "GuideWindow",
    "REPOSITORY_URL",
    "about_text",
    "guide_path",
    "open_guide_online",
    "read_guide",
]

REPOSITORY_URL = "https://github.com/bushgroup/mainspring"
"""Where the code is. Named once, and both URLs below are built from it."""

GUIDE_URL = f"{REPOSITORY_URL}/blob/main/docs/user-guide.md"
"""The guide as it stands on the default branch, which is not necessarily the guide this
program shipped with -- which is why it is the second entry and not the first."""

_GUIDE_RELATIVE = os.path.join("docs", "user-guide.md")


def guide_path() -> "str | None":
    """Where the user guide is on this machine, or None if it was not carried.

    Two places, in this order. A frozen build unpacks its data files beside the
    executable, which PyInstaller reports as `sys._MEIPASS`; a checkout has `docs/` at
    the repository root, which `mainspring.ROOT` already resolves. Returning None rather
    than raising, because a missing guide is a packaging fault to report in the window,
    not a reason for the Help menu to fail to open.
    """
    roots = []
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        roots.append(str(frozen))
        roots.append(os.path.dirname(os.path.abspath(sys.executable)))
    roots.append(ROOT)
    for root in roots:
        candidate = os.path.join(root, _GUIDE_RELATIVE)
        if os.path.isfile(candidate):
            return candidate
    return None


def read_guide() -> "tuple[str, str | None]":
    """`(markdown, directory)` for the guide, or a sentence saying it is not here.

    The directory travels with the text because the guide's images are relative to it
    and `QTextBrowser` needs a search path to resolve them.
    """
    path = guide_path()
    if path is None:
        return (
            "# User guide\n\nThe user guide was not installed with this copy of"
            f" mainspring. It is online at {GUIDE_URL}.\n",
            None,
        )
    with open(path, encoding="utf-8") as handle:
        return handle.read(), os.path.dirname(path)


def open_guide_online() -> bool:
    """Open the guide on GitHub in whatever the machine uses for a browser."""
    return bool(QDesktopServices.openUrl(QUrl(GUIDE_URL)))


def about_text() -> str:
    """What About says, as rich text.

    The version, the commit the build came from when there is one, and the license --
    which is the whole of what `README.md` states and the whole of what anyone asks a
    viewer about itself. The commit is the same one `mainspring.report` stamps a result
    with, so a figure and the window that made it name the same build.
    """
    commit = mainspring_commit()
    built = f"<p>Built from commit <code>{commit}</code>.</p>" if commit else ""
    return (
        f"<h3>mainspring {__version__}</h3>"
        "<p>An interactive viewer and a Python reader for UIMF ion mobility mass"
        " spectrometry files.</p>"
        f"{built}"
        "<p>BSD 3-Clause. Copyright University of Washington.</p>"
        "<p>UIMF is a format published by Pacific Northwest National Laboratory. Its"
        " reference implementation, UIMF-Library, is separately licensed by PNNL and is"
        " not redistributed here.</p>"
        f'<p><a href="{REPOSITORY_URL}">{REPOSITORY_URL}</a></p>'
    )


class GuideWindow(QDialog):
    """The user guide, scrolling, in a window of its own.

    Not modal: the guide is read *while* using the viewer, and a modal would make
    following it impossible. Its own window rather than a dock for the same reason a
    dock would be wrong -- it is read for a minute and closed, and it should not cost the
    heat map any width while it is open.
    """

    def __init__(self, parent: "object | None" = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("mainspring user guide")
        self.setModal(False)
        self.resize(760, 720)

        text, directory = read_guide()
        self._browser = QTextBrowser(self)
        self._browser.setOpenExternalLinks(True)
        if directory:
            # Relative image paths in the guide are relative to the file, and a text
            # browser resolves them against its search paths and nothing else.
            self._browser.setSearchPaths([directory])
        self._browser.setMarkdown(text)
        describe(self._browser, "The user guide for this version of mainspring.")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        describe(buttons.button(QDialogButtonBox.StandardButton.Close), "Close the guide.")

        layout = QVBoxLayout(self)
        layout.addWidget(self._browser)
        layout.addWidget(buttons)

    def text(self) -> str:
        """What the window is showing, for a caller that wants to check it has content."""
        return self._browser.toPlainText()
