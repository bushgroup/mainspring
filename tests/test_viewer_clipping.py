"""The clipping rows of the info panel, in a real window over `write_clipping_uimf`.

What is checked is what an operator reads: the full scale and where it came from, the
raw maximum, the frame's count, and the file's -- off until asked for, counted in the
background when it is, recounted when the bit depth moves, and "so far" while a run is
being followed (lab record, task 36).
"""

from __future__ import annotations

import numpy as np
import pytest

from mainspring.uimf import FrameSpec, GlobalSpec, UimfFile, UimfWriter
from mainspring.viewer.main_window import MainWindow
from mainspring.viewer.settings import ViewerSettings
from synthetic import write_clipping_uimf

from console_stub import ConsoleStub


def _open(qtbot, path, **settings):
    window = MainWindow(ViewerSettings(**settings))
    qtbot.addWidget(window)
    painted: list = []
    window.frame_shown.connect(painted.append)
    window.open_file(path)
    qtbot.waitUntil(lambda: bool(painted), timeout=5000)
    return window, painted


def _row(window, name):
    return window.info_panel.readouts()[name]


def test_a_clockwork_file_is_judged_at_its_stored_sixteen_bits(qtbot, tmp_path):
    """The full scale, said once, with the depth and where it came from; and the per-push
    line divided by the same 65535 rather than by a 14-bit ceiling four times smaller."""
    spec = write_clipping_uimf(tmp_path / "clip.uimf")
    window, _ = _open(qtbot, spec.path, detector_bits=8)
    assert _row(window, "Full scale") == "65,535 per push (14-bit from file, stored as 16-bit)"
    assert "14-bit from file, stored as 16-bit" in _row(window, "Per push")
    assert "100.00% full scale" in _row(window, "Per push"), "frame 1 peaks at 65535 a push"


def test_the_raw_maximum_and_the_frames_count_are_always_shown(qtbot, tmp_path):
    spec = write_clipping_uimf(tmp_path / "clip.uimf")
    window, _ = _open(qtbot, spec.path)
    qtbot.waitUntil(lambda: _row(window, "Raw maximum") == "262,140", timeout=5000)
    assert _row(window, "Clipped in frame") == "2 points"
    window.show_frame(2)
    qtbot.waitUntil(lambda: window._current_frame_number == 2, timeout=5000)
    assert _row(window, "Clipped in frame") == "1 point", "65535 is not clipped at four pushes"
    window.show_frame(3)
    qtbot.waitUntil(lambda: window._current_frame_number == 3, timeout=5000)
    assert _row(window, "Clipped in frame") == "0 points"


def test_the_file_count_is_off_until_asked_for_and_says_where_to_ask(qtbot, tmp_path):
    spec = write_clipping_uimf(tmp_path / "clip.uimf")
    window, _ = _open(qtbot, spec.path)
    assert not window.settings.count_clipping
    assert "off" in _row(window, "Clipped in file")
    assert "Data settings > Count clipping in file" in _row(window, "Clipped in file")

    window._count_clipping_action.setChecked(True)
    qtbot.waitUntil(lambda: _row(window, "Clipped in file") == "6 points in 3 frames",
                    timeout=5000)
    assert window.settings.count_clipping

    window._count_clipping_action.setChecked(False)
    assert "off" in _row(window, "Clipped in file")


def test_with_the_count_on_a_file_is_counted_as_it_opens(qtbot, tmp_path):
    spec = write_clipping_uimf(tmp_path / "clip.uimf")
    window, _ = _open(qtbot, spec.path, count_clipping=True)
    qtbot.waitUntil(lambda: _row(window, "Clipped in file") == "6 points in 3 frames",
                    timeout=5000)


def test_a_sum_says_the_count_is_of_single_frames(qtbot, tmp_path):
    spec = write_clipping_uimf(tmp_path / "clip.uimf")
    window, painted = _open(qtbot, spec.path)
    window.sum_frames([1, 2])
    qtbot.waitUntil(lambda: window._current_frame_number == 0, timeout=5000)
    assert _row(window, "Clipped in frame") == "single frames only, and a sum is shown"


def test_a_bit_depth_set_by_hand_recounts_both_and_says_when_it_is_too_low(qtbot, tmp_path):
    """On a file that declares nothing the count is only as right as the setting, and a
    point above full scale is the one sign the setting is wrong."""
    spec = write_clipping_uimf(tmp_path / "pnnl.uimf", detector_bits=None)
    window, _ = _open(qtbot, spec.path, detector_bits=16, count_clipping=True)
    assert _row(window, "Full scale") == "65,535 per push (16-bit from setting)"
    qtbot.waitUntil(lambda: _row(window, "Clipped in file") == "6 points in 3 frames",
                    timeout=5000)

    window._bits_box.setValue(8)
    assert _row(window, "Full scale") == "255 per push (8-bit from setting)"
    clipped, over = spec.expected(255)
    frame_over = over.get(1, 0)
    assert _row(window, "Clipped in frame").startswith(f"{clipped[1]:,} points")
    assert f"{frame_over:,} above full scale" in _row(window, "Clipped in frame")
    total, frames = sum(clipped.values()), len(clipped)
    qtbot.waitUntil(
        lambda: _row(window, "Clipped in file").startswith(
            f"{total:,} points in {frames:,} frames"),
        timeout=5000,
    )
    assert "bit depth is too low" in _row(window, "Clipped in file")


# --- a run being followed ----------------------------------------------------------------

@pytest.fixture
def quick_poll(monkeypatch):
    from mainspring.viewer import workers

    monkeypatch.setattr(workers, "POLL_INTERVAL_S", 0.05)


def test_a_followed_run_is_counted_as_it_is_written_and_says_so_far(qtbot, tmp_path,
                                                                    quick_poll):
    """The two-party stand-in: `UimfWriter` declares and finalises frames, the console
    stub appends their scans, and the window, following, recounts only the tail."""
    path = str(tmp_path / "live.uimf")
    writer = UimfWriter(path, GlobalSpec(bins=1024, prescan_tof_pulses=8, detector_bits=14))
    console = ConsoleStub(path)

    def frame(number, values):
        writer.add_frame(FrameSpec(scans=8, accumulations=1, calibration_slope=0.738123,
                                   calibration_intercept=0.0769,
                                   average_tof_length_ns=129003.6))
        rows = [(3, np.arange(len(values), dtype=np.int64) * 7 + 5,
                 np.asarray(values, dtype=np.int32))]
        console.acquire_frame(number, iter(rows))

    try:
        frame(1, [65535, 10])
        writer.finalise_frame(1, duration_s=0.1)
        window, _ = _open(qtbot, path, count_clipping=True)
        qtbot.waitUntil(lambda: _row(window, "Clipped in file") == "1 point in 1 frame",
                        timeout=5000)
        window._follow_action.setChecked(True)
        assert window.following

        frame(2, [65535, 65535, 3])  # declared and written, not yet finished
        qtbot.waitUntil(
            lambda: _row(window, "Clipped in file") == "3 points in 2 frames so far,"
                                                         " still being written",
            timeout=5000,
        )
        assert _row(window, "Raw maximum") == "65,535 so far"
        console.acquire_frame(2, iter([(4, np.array([1], dtype=np.int64),
                                        np.array([65535], dtype=np.int32))]))
        writer.finalise_frame(2, duration_s=0.1)
        qtbot.waitUntil(lambda: _row(window, "Clipped in file").startswith("4 points"),
                        timeout=5000)

        window._follow_action.setChecked(False)
        assert _row(window, "Clipped in file") == "4 points in 2 frames"
        assert _row(window, "Raw maximum") == "65,535"
        assert UimfFile(path).clipping().total == 4
    finally:
        writer.close()
