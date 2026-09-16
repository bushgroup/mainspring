"""Task 08: following a file while something else is writing it.

Two parties, no instrument. `mainspring.uimf.writer` creates the file and owns its
parameters, `tests/console_stub.py` appends `Frame_Scans` the way PNNL's console does,
and the reader and the viewer under test see exactly what they would see on the rig:
a WAL database growing under them, frames that are not finished yet, and a completion
marker arriving after the scans rather than before (lab record, tasks 08 and 16).

The reader half needs no Qt. The viewer half drives a real `MainWindow` offscreen and
waits on the signals the poll travels through, rather than calling the poll itself --
the thing worth testing is that a frame written by another process reaches the screen,
not that a method returns.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mainspring.uimf import (
    FrameSpec,
    GlobalSpec,
    LiveState,
    UimfFile,
    UimfWriter,
    is_local_path,
)
from mainspring.viewer.main_window import (
    FOLLOW_FIXED,
    FOLLOW_METHOD_SUM,
    FOLLOW_NEWEST,
    MainWindow,
)

from console_stub import ConsoleStub

BINS = 4096
SCANS = 12


def _scans(seed: int, first: int = 0, last: int = SCANS):
    """`(scan, bin_index, intensity)` triples for one batch of a frame's scans.

    Every scan gets more than one point, so none is dropped by the console's
    "one stream element or none" rule and the row count is predictable. `first` and
    `last` are what let a frame arrive in two batches, which is what the console does
    every `NotifyOnScansCount` scans.
    """
    rng = np.random.default_rng(seed)
    for scan in range(first, last):
        bins = np.sort(rng.choice(BINS, size=4, replace=False)).astype(np.int64)
        yield scan, bins, np.full(4, 10 + seed, dtype=np.int32)


class Acquisition:
    """A file being acquired: the writer, the console stand-in, and a frame counter.

    Mirrors the real division of labour rather than writing both halves one way --
    `add_frame` before the scans, the console appending them, `finalise_frame` after.
    A test that wants an unfinished frame simply does not call `finish`.
    """

    def __init__(self, path, **global_spec):
        self.path = os.fspath(path)
        self.writer = UimfWriter(
            self.path,
            GlobalSpec(bins=BINS, prescan_tof_pulses=SCANS, detector_bits=14,
                       **global_spec),
        )
        self.console = ConsoleStub(self.path)
        self.frames = 0

    def start(self, scans: int = SCANS, **frame_spec) -> int:
        """Declare the next frame and let the console fill part of it. Not finished yet.

        `scans` short of the frame's own `Scans` leaves the frame where a real one
        spends most of its life: declared, partly written, and not yet finalised.
        """
        self.frames += 1
        self.writer.add_frame(FrameSpec(
            scans=SCANS, calibration_slope=0.738123, calibration_intercept=0.0769,
            average_tof_length_ns=129003.6, **frame_spec,
        ))
        self.console.acquire_frame(self.frames, _scans(self.frames, 0, scans))
        return self.frames

    def append(self, frame: int, first: int, last: int = SCANS) -> None:
        """One more batch of scans into a frame the console is still filling."""
        self.console.acquire_frame(frame, _scans(frame, first, last))

    def finish(self, frame: "int | None" = None) -> None:
        self.writer.finalise_frame(frame or self.frames, duration_s=0.5)

    def frame(self, **frame_spec) -> int:
        """A whole frame, declared, filled and finished."""
        number = self.start(**frame_spec)
        self.finish(number)
        return number

    def close(self) -> None:
        self.writer.close()


@pytest.fixture
def acquisition(tmp_path):
    running = Acquisition(tmp_path / "live.uimf")
    yield running
    running.close()


# --- the reader's own answer ------------------------------------------------------------

def test_refresh_reports_frames_as_they_are_written_and_finished(acquisition):
    """The whole live contract in one sequence: a frame appears, then becomes final.

    One `UimfFile` across all of it, because that is what the viewer holds -- a reader
    that answered from what it cached the first time would report the acquisition's
    first second for the rest of the run.
    """
    file = UimfFile(acquisition.path)
    assert file.refresh() == LiveState(frames=(), provisional=frozenset())

    acquisition.start()
    growing = file.refresh()
    assert growing.frames == (1,)
    assert growing.provisional == frozenset({1})
    assert growing.final == ()

    acquisition.finish()
    finished = file.refresh()
    assert finished.frames == (1,)
    assert finished.provisional == frozenset()
    assert finished.final == (1,)

    acquisition.start()
    both = file.refresh()
    assert both.frames == (1, 2)
    assert both.provisional == frozenset({2})
    assert both.final == (1,)


def test_a_frame_still_being_written_is_never_kept_in_the_cache(acquisition):
    """The rule the cache exists to hold, exercised through the read path.

    A provisional frame is decoded and handed over like any other; what must not happen
    is `FrameCache` keeping it, because the copy would stay wrong for as long as it was
    held and the frame is about to grow.
    """
    from mainspring.uimf import FrameCache

    acquisition.start()
    file = UimfFile(acquisition.path)
    cache = FrameCache()

    growing = file.read_frame(1)
    assert growing.provisional
    assert not cache.put(file.path, growing)
    assert len(cache) == 0

    acquisition.finish()
    file.refresh()
    settled = file.read_frame(1)
    assert not settled.provisional
    assert cache.put(file.path, settled)
    assert len(cache) == 1


def test_refresh_sees_scans_appended_to_a_frame_it_already_knows_about(acquisition):
    """A frame's *contents* grow without its number changing, which is why a poll that
    only watched the frame list would show an experiment in one jump at the end."""
    number = acquisition.start(scans=SCANS // 2)
    file = UimfFile(acquisition.path)
    half = file.read_frame(number)

    acquisition.append(number, SCANS // 2)
    state = file.refresh()
    assert state.provisional == frozenset({number})
    assert len(file.read_frame(number)) > len(half)


def test_refresh_costs_two_queries_however_many_frames_there_are(acquisition, monkeypatch):
    """The rule the whole poll rests on: nothing here is asked once per frame.

    Counted rather than timed, because what matters is the shape and not this
    workstation's speed -- one `frame_params` call per frame is 6.9 s over 5,000 frames
    and would be invisible on the handful here (lab record, task 17).
    """
    import contextlib

    from mainspring.uimf import reader as reader_module

    for _ in range(6):
        acquisition.frame()
    file = UimfFile(acquisition.path)
    file.refresh()  # the first look learns which frames are already finished
    acquisition.start()

    opened: list[str] = []
    real_connect = reader_module.connect

    @contextlib.contextmanager
    def counting_connect(*args, **kwargs):
        opened.append("connect")
        with real_connect(*args, **kwargs) as conn:
            yield conn

    monkeypatch.setattr(reader_module, "connect", counting_connect)
    state = file.refresh()

    assert state.provisional == frozenset({7})
    assert len(opened) == 2  # the frame list, and the markers it does not already know


def test_a_file_with_no_completion_marker_falls_back_to_its_mtime(tmp_path, monkeypatch):
    """Every file PNNL's writers produce, where nothing says when a frame is finished.

    The heuristic is deliberately conservative: the last frame of a file written to
    recently. Over-reporting costs one uncached frame; under-reporting would hand the
    viewer half a frame and let it keep it.
    """
    import sqlite3
    import time

    from mainspring.uimf import WRITER_STAMP
    from mainspring.uimf import reader as reader_module
    from synthetic import write_synthetic_uimf

    path = tmp_path / "unstamped.uimf"
    write_synthetic_uimf(path, frames=3, scans=SCANS, bins=BINS)
    # The one thing the fixture cannot produce, since it writes through our own writer:
    # a file nobody stamped. Dropping the stamp is what makes this file PNNL-shaped for
    # the purpose of this question, and it is the exact condition the branch reads.
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM Global_Params WHERE ParamName = ?", (WRITER_STAMP,))

    file = UimfFile(path)
    assert not file.has_completion_markers

    fresh = file.refresh()
    assert fresh.provisional == frozenset({3})  # just written, so the last frame may grow
    assert fresh.final == (1, 2)
    assert file.is_provisional(3) and not file.is_provisional(2)

    monkeypatch.setattr(
        reader_module.os.path, "getmtime",
        lambda _p: time.time() - reader_module.PROVISIONAL_WINDOW_S - 1.0,
    )
    settled = file.refresh()
    assert settled.provisional == frozenset()
    assert settled.final == (1, 2, 3)


def test_a_network_path_is_not_a_path_a_file_can_be_followed_on(tmp_path):
    """WAL coordinates its readers and its writer through shared memory that only
    exists when both are on the same machine, so a share is refused for *following*.
    Opening and reading one is untouched."""
    assert is_local_path(tmp_path / "here.uimf")
    assert not is_local_path("\\\\masstro\\runs\\today.uimf")
    assert not is_local_path("//masstro/runs/today.uimf")


# --- the viewer following one --------------------------------------------------------

@pytest.fixture
def quick_poll(monkeypatch):
    """Poll twenty times a second instead of once, so the suite is not paced by it.

    The interval is a property of how often an instrument has anything new to say, not
    of anything under test here; every test below waits on the poll's own signal rather
    than on the clock, so the number only decides how long they take.
    """
    from mainspring.viewer import workers

    monkeypatch.setattr(workers, "POLL_INTERVAL_S", 0.05)


@pytest.fixture
def following_window(qtbot, acquisition, quick_poll):
    """A window with the acquisition open and `Follow` on, and its first frame painted."""
    acquisition.frame(method_frame=1, repetition=1, repetitions=3)
    window = MainWindow()
    qtbot.addWidget(window)
    painted: list = []
    window.frame_shown.connect(painted.append)
    window.open_file(acquisition.path)
    qtbot.waitUntil(lambda: bool(painted), timeout=5000)
    window._follow_action.setChecked(True)
    assert window.following
    return window, acquisition, painted


def test_follow_is_refused_on_a_path_that_is_not_a_local_drive(qtbot, monkeypatch,
                                                              acquisition):
    """The tick goes back and the status bar says why. A dialog would be a modal in
    front of someone watching a run."""
    acquisition.frame()
    window = MainWindow()
    qtbot.addWidget(window)
    painted: list = []
    window.frame_shown.connect(painted.append)
    window.open_file(acquisition.path)
    qtbot.waitUntil(lambda: bool(painted), timeout=5000)

    monkeypatch.setattr("mainspring.viewer.main_window.is_local_path", lambda _p: False)
    window._follow_action.setChecked(True)

    assert not window.following
    assert not window._follow_action.isChecked()
    assert "local drive" in window.status_text()


def test_follow_is_refused_before_a_file_is_open(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._follow_action.setChecked(True)
    assert not window.following
    assert "Open a file" in window.status_text()


def test_following_grows_the_frame_axis_without_moving_the_view(following_window, qtbot):
    """`Fixed frame`: the default, and the one a person studying a frame while the run
    continues wants. The spinner's reach grows; the frame under them does not change."""
    window, acquisition, _ = following_window
    assert window._follow_mode.currentText() == FOLLOW_FIXED
    before = window.heatmap.view_range()

    acquisition.frame(method_frame=1, repetition=2, repetitions=3)
    qtbot.waitUntil(lambda: window._frame_spin.maximum() == 2, timeout=5000)

    assert window._current_frame_number == 1
    assert window.heatmap.view_range() == before
    assert window._frame_numbers == [1, 2]


def test_following_the_newest_frame_lands_on_each_frame_as_it_arrives(following_window,
                                                                     qtbot):
    window, acquisition, painted = following_window
    # Switching the mode acts on the last poll's findings straight away, so let that
    # land first -- otherwise the paint this test waits for could be that one.
    window._follow_mode.setCurrentText(FOLLOW_NEWEST)
    qtbot.waitUntil(lambda: window._current_frame_number == 1, timeout=5000)

    painted.clear()
    acquisition.frame(method_frame=1, repetition=2, repetitions=3)
    qtbot.waitUntil(lambda: window._current_frame_number == 2, timeout=5000)
    qtbot.waitUntil(lambda: bool(painted), timeout=5000)


def test_a_frame_still_being_written_is_drawn_and_labelled_as_unfinished(following_window,
                                                                        qtbot):
    """Drawn, because an experiment filling in front of the operator is the live view
    worth having; labelled, because a partial frame quoted as a finished one is a number
    that will be wrong by the time it is written down."""
    window, acquisition, _ = following_window
    window._follow_mode.setCurrentText(FOLLOW_NEWEST)

    acquisition.start(method_frame=1, repetition=2, repetitions=3)
    qtbot.waitUntil(lambda: window._current_frame_number == 2, timeout=5000)
    assert window._current_frame is not None and window._current_frame.provisional
    assert "still being written" in window.status_text()
    assert window.info_panel._state_label.text() == "still being written"

    acquisition.finish()
    qtbot.waitUntil(
        lambda: window._current_frame is not None and not window._current_frame.provisional,
        timeout=5000,
    )
    assert "still being written" not in window.status_text()
    assert window.info_panel._state_label.text() == "complete"


def test_the_running_sum_totals_the_finished_repetitions_only(following_window, qtbot):
    """`Method frame sum`: the summed heatmap today's files hold, arriving as it is
    earned. A frame still being written is deliberately left out -- a total that
    included one would change every time it was recomputed."""
    window, acquisition, _ = following_window
    window._follow_mode.setCurrentText(FOLLOW_METHOD_SUM)
    qtbot.waitUntil(lambda: window._live_sum_frames == (1,), timeout=5000)

    acquisition.frame(method_frame=1, repetition=2, repetitions=3)
    qtbot.waitUntil(
        lambda: "2 repetitions" in window.status_text(), timeout=5000
    )
    assert "Running sum of method frame 1" in window.status_text()
    assert window._live_sum_frames == (1, 2)

    # An unfinished third repetition is not added until it is finished.
    acquisition.start(method_frame=1, repetition=3, repetitions=3)
    qtbot.waitUntil(lambda: window._frame_spin.maximum() == 3, timeout=5000)
    assert window._live_sum_frames == (1, 2)
    acquisition.finish()
    qtbot.waitUntil(lambda: window._live_sum_frames == (1, 2, 3), timeout=5000)


def test_the_running_sum_totals_what_the_frames_own_tic_columns_say(following_window,
                                                                   qtbot):
    """Against the file's own summary columns, which are ground truth for a decode
    (lab record, task 01) -- so this checks the live total and not just that one
    arrived."""
    window, acquisition, _ = following_window
    acquisition.frame(method_frame=1, repetition=2, repetitions=3)
    window._follow_mode.setCurrentText(FOLLOW_METHOD_SUM)
    # On the message rather than on `_live_sum_frames`, which is set when the sum is
    # *asked for*: the total itself arrives from the load worker a moment later.
    qtbot.waitUntil(
        lambda: "2 repetitions" in window.status_text(), timeout=5000
    )
    assert window._current_frame_number == 0

    file = UimfFile(acquisition.path)
    expected = sum(float(file.scan_summary(n)[3].sum()) for n in (1, 2))
    assert float(window._current_frame.intensity.sum()) == pytest.approx(expected)


def test_the_method_frame_sum_is_offered_only_on_a_file_that_groups_its_frames(
    qtbot, tmp_path
):
    """Removed rather than greyed, for the reason the method-frame spinners are hidden:
    an entry that can never be chosen on this file is not a choice."""
    ungrouped = Acquisition(tmp_path / "ungrouped.uimf")
    ungrouped.frame()
    ungrouped.close()

    window = MainWindow()
    qtbot.addWidget(window)
    painted: list = []
    window.frame_shown.connect(painted.append)
    window.open_file(ungrouped.path)
    qtbot.waitUntil(lambda: bool(painted), timeout=5000)

    offered = [window._follow_mode.itemText(i) for i in range(window._follow_mode.count())]
    assert offered == [FOLLOW_FIXED, FOLLOW_NEWEST]


def test_opening_another_file_stops_following_the_first(following_window, qtbot, tmp_path):
    """Following is about one acquisition. The second file a person opens is nearly
    always a finished one, and a poll left running on it is lock traffic against
    nothing."""
    window, _, painted = following_window
    other = Acquisition(tmp_path / "other.uimf")
    other.frame()
    other.close()

    painted.clear()
    window.open_file(other.path)
    qtbot.waitUntil(lambda: bool(painted), timeout=5000)
    assert not window.following
    assert not window._follow_mode.isEnabled()


def test_a_poll_that_keeps_failing_stops_following_instead_of_repeating(following_window,
                                                                       qtbot, monkeypatch):
    """A file moved or unmounted mid-run fails every poll. Reporting that once a second
    until someone looks is worse than stopping and saying so once."""
    window, _, _ = following_window
    from mainspring.uimf import UimfFile

    def _gone(self):
        raise OSError("the file has gone away")

    monkeypatch.setattr(UimfFile, "refresh", _gone)
    qtbot.waitUntil(lambda: not window.following, timeout=5000)

    assert not window._follow_action.isChecked()
    assert not window._follow_mode.isEnabled()
    assert "gone away" in window.status_text()


def test_following_does_not_reset_a_frame_type_filter_the_user_set(following_window, qtbot):
    """A rebuilt menu ticks its first entry, so a poll that repopulated it every time
    would put an operator who had filtered to one type back on `All frames`."""
    window, acquisition, _ = following_window
    window._type_actions["MS1"].trigger()
    assert window._type_group.checkedAction().text() == "MS1"

    acquisition.frame(method_frame=1, repetition=2, repetitions=3)
    qtbot.waitUntil(lambda: window._frame_spin.maximum() == 2, timeout=5000)

    assert window._type_group.checkedAction().text() == "MS1"
    assert window._active_frame_numbers == [1, 2]
