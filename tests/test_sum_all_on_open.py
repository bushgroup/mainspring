"""`Data settings > Sum all on open`: a file opened by hand paints, then sums every frame.

What is checked is what the user sees: frame 1 first, then the sum of every frame with
the file's own `TIC` columns behind it; Cancel leaving frame 1; the type filter back on
`All frames`; a growing file opened by hand getting what the button would give it; and a
followed open -- a launch or a `Live` switch -- left to `Show` (lab record, task 43).
"""

from __future__ import annotations

import time

import pytest

from mainspring import interface
from mainspring.uimf import UimfFile
from mainspring.viewer.main_window import FOLLOW_NEWEST, MainWindow
from mainspring.viewer.settings import ViewerSettings, load_settings, save_settings

from synthetic import write_synthetic_uimf
from test_live_viewing import Acquisition, acquisition, pointer, quick_poll  # noqa: F401


def _window(qtbot, **settings):
    """A window whose every paint is recorded as the frame number it showed."""
    window = MainWindow(ViewerSettings(**settings))
    qtbot.addWidget(window)
    shown: list[int] = []
    window.frame_shown.connect(lambda _result: shown.append(window._current_frame_number))
    return window, shown


def _spy_sums(window) -> list:
    """Every `sum_frames` call the window makes from here on, still made."""
    calls: list = []
    original = window.sum_frames

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    window.sum_frames = spy
    return calls


def _file_tic(path: str, frames) -> float:
    file = UimfFile(path)
    return sum(float(file.scan_summary(n)[3].sum()) for n in frames)


def test_the_toggle_is_in_data_settings_unticked_and_remembered(qtbot):
    window, _ = _window(qtbot)
    action = window._sum_all_on_open_action
    assert not window.settings.sum_all_on_open
    assert not action.isChecked()
    data_menu = next(a.menu() for a in window.menuBar().actions()
                     if a.text() == "&Data settings")
    assert action in data_menu.actions()
    assert action.toolTip()

    action.setChecked(True)
    assert window.settings.sum_all_on_open
    save_settings(window.settings)
    assert load_settings().sum_all_on_open


def test_unticked_an_open_shows_frame_one_and_sums_nothing(qtbot, tmp_path):
    spec = write_synthetic_uimf(tmp_path / "five.uimf", frames=5)
    window, shown = _window(qtbot)
    calls = _spy_sums(window)
    window.open_file(spec.path)
    qtbot.waitUntil(lambda: bool(shown), timeout=5000)
    qtbot.wait(300)
    assert shown == [1]
    assert calls == [] and window._sum_dialog is None


def test_ticked_an_open_paints_frame_one_then_the_sum_of_every_frame(qtbot, tmp_path):
    spec = write_synthetic_uimf(tmp_path / "five.uimf", frames=5)
    window, shown = _window(qtbot, sum_all_on_open=True)
    calls = _spy_sums(window)
    window.open_file(spec.path)
    qtbot.waitUntil(lambda: 0 in shown, timeout=5000)

    assert shown[0] == 1, "first paint is still what an open is measured by"
    assert calls == [((), {})], "the toolbar's own Sum all, once"
    assert window._sum_dialog is None
    assert "Sum of 5 frames" in window.status_text()
    expected = sum(spec.tic(n) for n in spec.frames)
    assert float(window._current_frame.intensity.sum()) == pytest.approx(expected, rel=1e-9)
    assert _file_tic(spec.path, spec.frames) == pytest.approx(expected, rel=1e-9)

    # Nothing else of the same file starts another one: not a frame, not the filter.
    window.show_frame(2)
    qtbot.waitUntil(lambda: shown[-1] == 2, timeout=5000)
    window._type_actions["MS1"].trigger()
    window._type_actions["All frames"].trigger()
    qtbot.wait(200)
    assert len(calls) == 1


def test_each_open_sums_once(qtbot, tmp_path):
    spec = write_synthetic_uimf(tmp_path / "two.uimf", frames=2)
    window, shown = _window(qtbot, sum_all_on_open=True)
    calls = _spy_sums(window)
    for opens in (1, 2):
        window.open_file(spec.path)
        qtbot.waitUntil(lambda: shown.count(0) == opens, timeout=5000)
    assert len(calls) == 2


def test_cancel_leaves_frame_one_on_screen(qtbot, tmp_path, monkeypatch):
    import mainspring.uimf.reader as reader_module

    original_read_frame = reader_module.UimfFile.read_frame

    def _slow_read_frame(self, *args, **kwargs):
        time.sleep(0.02)
        return original_read_frame(self, *args, **kwargs)

    spec = write_synthetic_uimf(tmp_path / "many.uimf", frames=40, scans=16)
    monkeypatch.setattr(reader_module.UimfFile, "read_frame", _slow_read_frame)
    window, shown = _window(qtbot, sum_all_on_open=True)
    window.open_file(spec.path)
    qtbot.waitUntil(lambda: bool(shown), timeout=5000)
    qtbot.waitUntil(lambda: window._sum_dialog is not None, timeout=5000)
    window._sum_dialog.canceled.emit()  # what the Cancel button sends
    qtbot.waitUntil(lambda: window._sum_dialog is None, timeout=5000)
    qtbot.wait(200)

    assert shown == [1]
    assert window._current_frame_number == 1


def test_the_sum_covers_every_frame_type(qtbot, acquisition):
    """Calibration frames included: the open puts the filter back on `All frames`, so the
    sum is what pressing the button straight after the open would give."""
    acquisition.frame(frame_type=1)
    acquisition.frame(frame_type=3)
    acquisition.frame(frame_type=1)
    acquisition.close()
    window, shown = _window(qtbot)
    window.open_file(acquisition.path)
    qtbot.waitUntil(lambda: bool(shown), timeout=5000)
    window._type_actions["MS1"].trigger()
    assert window._active_frame_numbers == [1, 3]

    window._sum_all_on_open_action.setChecked(True)
    window.open_file(acquisition.path)
    qtbot.waitUntil(lambda: 0 in shown, timeout=5000)
    assert window._type_actions["All frames"].isChecked()
    assert "Sum of 3 frames" in window.status_text()
    assert float(window._current_frame.intensity.sum()) == pytest.approx(
        _file_tic(acquisition.path, (1, 2, 3)))


def test_a_growing_file_opened_by_hand_sums_what_the_button_would(qtbot, acquisition):
    """Not followed, so the option applies: the frames listed at open, the last one as
    far as it has been written, and the total said to be still being written -- the
    same as a pressed `Sum all` (`mainspring.uimf.frame.sum_frames`)."""
    acquisition.frame()
    acquisition.frame()
    acquisition.start(scans=5)
    window, shown = _window(qtbot, sum_all_on_open=True)
    window.open_file(acquisition.path)
    qtbot.waitUntil(lambda: 0 in shown, timeout=5000)

    assert shown[0] == 1
    assert "Sum of 3 frames" in window.status_text()
    assert "still being written" in window.status_text()
    assert window._current_frame.provisional
    assert float(window._current_frame.intensity.sum()) == pytest.approx(
        _file_tic(acquisition.path, (1, 2, 3)))
    assert not window.following


def test_a_launched_follow_is_left_to_show(qtbot, acquisition, quick_poll):
    acquisition.frame()
    acquisition.frame()
    window, shown = _window(qtbot, sum_all_on_open=True)
    calls = _spy_sums(window)
    window.follow_when_opened(show=FOLLOW_NEWEST)
    window.open_file(acquisition.path, from_command_line=True)
    qtbot.waitUntil(lambda: window.following, timeout=5000)
    qtbot.wait(300)
    assert calls == [] and window._sum_dialog is None
    assert 0 not in shown


def test_a_live_switch_is_left_to_show(qtbot, acquisition, quick_poll, pointer, tmp_path):
    finished = Acquisition(tmp_path / "yesterday.uimf")
    finished.frame()
    finished.close()
    acquisition.frame()
    acquisition.frame()
    interface.write_live_pointer(acquisition.path, writer="clockwork")

    window, shown = _window(qtbot)
    window.open_file(finished.path)
    qtbot.waitUntil(lambda: bool(shown), timeout=5000)
    window._sum_all_on_open_action.setChecked(True)
    calls = _spy_sums(window)

    window._follow_action.setChecked(True)
    qtbot.waitUntil(lambda: window.following and window._path == acquisition.path,
                    timeout=5000)
    qtbot.wait(300)
    assert calls == [] and window._sum_dialog is None
