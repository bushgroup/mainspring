"""The Help menu: the guide that shipped, the guide online, and About.

The guide is `docs/user-guide.md` and nothing converts it, so what these check is that
the file is found wherever the program is running from, that the window renders it, and
that About names the version the three declarations agree on.
"""

from __future__ import annotations

import os

import pytest

from mainspring import __version__
from mainspring.viewer import help as help_module
from mainspring.viewer.help import (
    GUIDE_URL,
    GuideWindow,
    about_text,
    guide_path,
    read_guide,
)
from mainspring.viewer.main_window import MainWindow


def test_the_guide_is_found_in_a_checkout():
    """A bare clone has `docs/user-guide.md` at the repository root, and that is what a
    developer running `uv run mainspring` gets."""
    path = guide_path()
    assert path is not None
    assert os.path.isfile(path)
    assert os.path.basename(path) == "user-guide.md"


def test_the_guide_is_found_beside_a_frozen_executable(tmp_path, monkeypatch):
    """PyInstaller unpacks data files to `sys._MEIPASS`, and the guide is carried as
    one. Checked by building that layout rather than by trusting the spec: the failure
    it prevents is a Help menu that works in every test and is empty in the installer."""
    bundled = tmp_path / "docs"
    bundled.mkdir()
    (bundled / "user-guide.md").write_text("# frozen\n", encoding="utf-8")
    monkeypatch.setattr(help_module.sys, "_MEIPASS", str(tmp_path), raising=False)

    assert guide_path() == str(bundled / "user-guide.md")
    text, directory = read_guide()
    assert text == "# frozen\n"
    assert directory == str(bundled)


def test_a_guide_that_was_not_carried_says_so_and_names_the_web_copy(monkeypatch):
    """A packaging fault is reported in the window rather than raised: the Help menu
    opening to an explanation is better than a traceback, and the sentence says where
    the guide actually is."""
    monkeypatch.setattr(help_module, "guide_path", lambda: None)
    text, directory = read_guide()
    assert directory is None
    assert GUIDE_URL in text


def test_the_guide_window_shows_the_guide(qtbot):
    window = GuideWindow()
    qtbot.addWidget(window)
    shown = window.text()
    assert "mainspring user guide" in shown
    assert len(shown) > 5000  # the whole document, not a heading


def test_about_names_the_version_and_the_license():
    text = about_text()
    assert __version__ in text
    assert "BSD 3-Clause" in text
    assert "University of Washington" in text


def test_the_help_menu_carries_three_entries(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    menus = [menu.title() for menu in window.menuBar().findChildren(type(window.menuBar().addMenu("x")))]
    assert any("Help" in title for title in menus)
    labels = [
        window._guide_action.text(),
        window._guide_online_action.text(),
        window._about_action.text(),
    ]
    assert [label.replace("&", "") for label in labels] == [
        "User guide", "User guide online", "About mainspring",
    ]


def test_the_guide_window_is_the_same_window_the_second_time(qtbot):
    """A 31 KB document rebuilt on every press would lose the reader's place."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._show_guide()
    first = window._guide_window
    window._show_guide()
    assert window._guide_window is first


def test_a_browser_that_will_not_open_is_reported(qtbot, monkeypatch):
    """Nothing visible happens on a machine with no browser association, so the entry
    would look broken rather than blocked."""
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr("mainspring.viewer.main_window.open_guide_online", lambda: False)
    window._open_guide_online()
    assert "browser" in window.status_text()


def test_a_relative_link_in_the_guide_points_at_this_release_online():
    """Links are written relative to `docs/`, as GitHub renders them."""
    base = f"{help_module.REPOSITORY_URL}/blob/v{__version__}"
    assert help_module.online_url("uimf-format.md") == f"{base}/docs/uimf-format.md"
    assert (help_module.online_url("uimf-format.md#when-a-frame-is-finished")
            == f"{base}/docs/uimf-format.md#when-a-frame-is-finished")
    assert help_module.online_url("../README.md") == f"{base}/README.md"
    assert help_module.online_url("../../elsewhere.md") == help_module.REPOSITORY_URL


def test_the_guide_stays_on_screen_when_a_link_leaves_it(qtbot, monkeypatch):
    """Qt follows a relative link by loading the target in place of the guide: an empty
    page in the installer, which does not carry the README or the format page, and a
    page with no way back in a checkout. Such a link opens online instead."""
    opened = []
    monkeypatch.setattr(help_module.QDesktopServices, "openUrl",
                        lambda url: opened.append(url.toString()) or True)
    window = GuideWindow()
    qtbot.addWidget(window)
    browser = window._browser
    before = window.text()

    browser.setSource(help_module.QUrl("uimf-format.md"))
    browser.setSource(help_module.QUrl("../README.md#writing-a-file"))
    assert window.text() == before
    assert opened == [
        help_module.online_url("uimf-format.md"),
        help_module.online_url("../README.md#writing-a-file"),
    ]


def test_a_link_within_the_guide_scrolls_to_its_heading(qtbot, monkeypatch):
    """Qt's Markdown import names no heading, so without the slug lookup every
    cross-reference in the guide landed at the top of it."""
    opened = []
    monkeypatch.setattr(help_module.QDesktopServices, "openUrl",
                        lambda url: opened.append(url.toString()) or True)
    window = GuideWindow()
    window.resize(760, 720)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    browser = window._browser
    before = window.text()
    browser.setSource(help_module.QUrl("#frames-that-are-not-finished-yet"))
    assert window.text() == before
    assert opened == []
    assert browser.verticalScrollBar().value() > 0


def test_every_link_within_the_guide_names_a_heading(qtbot):
    import re

    window = GuideWindow()
    qtbot.addWidget(window)
    text, _ = read_guide()
    fragments = set(re.findall(r"\]\(#([^)\s]+)\)", text))
    assert len(fragments) > 5
    missing = [f for f in sorted(fragments) if window._browser.heading(f) is None]
    assert missing == []


def test_every_relative_link_in_the_guide_names_a_file_that_exists():
    """The link opens at the release tag, so the file has to be in the release."""
    import re

    text, directory = read_guide()
    links = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
    relative = [link for link in links
                if "://" not in link and not link.startswith("mailto:")
                and not link.startswith("images/")]
    assert relative, "the guide links to the README and the format page"
    for link in relative:
        assert os.path.isfile(os.path.join(directory, link)), link
