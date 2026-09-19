"""Tasks 08 and 26: following a file while something else is writing it.

Two parties, no instrument. `mainspring.uimf.writer` creates the file and owns its
parameters, `tests/console_stub.py` appends `Frame_Scans` the way PNNL's console does,
and the reader and the viewer under test see exactly what they would see on the rig:
a WAL database growing under them, frames that are not finished yet, and a completion
marker arriving after the scans rather than before (lab record, tasks 08 and 16).

The reader half needs no Qt. The viewer half drives a real `MainWindow` offscreen and
waits on the signals the poll travels through, rather than calling the poll itself --
the thing worth testing is that a frame written by another process reaches the screen,
not that a method returns.

The last two sections are task 26's: the rolling sum over the newest finished frames,
and the launch route another program starts the viewer by. Both are the same acquisition
stand-in and the same window, since both are ways of asking for what task 08 built.

Task 28 adds the end of a run that does not keep its raw file: the followed file is
deleted, and the window has to say so in words and offer the companion that survived.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mainspring import interface
from mainspring.uimf import (
    FileGone,
    FrameSpec,
    GlobalSpec,
    LiveState,
    UimfFile,
    UimfWriter,
    is_local_path,
    summed_companion,
)
from mainspring.viewer import app as viewer_app
from mainspring.viewer.app import Launch, parse_arguments
from mainspring.viewer.app import main as viewer_main
from mainspring.viewer.main_window import (
    FOLLOW_FIXED,
    FOLLOW_METHOD_SUM,
    FOLLOW_MODE_WORDS,
    FOLLOW_MODES,
    FOLLOW_NEWEST,
    FOLLOW_ROLLING_SUM,
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


def test_a_deleted_file_reads_as_gone_rather_than_as_sqlite_being_unhappy(acquisition):
    """A run that does not keep its raw file deletes it at the close, which is an
    ordinary end to an acquisition. SQLite has one message for that, for a corrupt file
    and for an unreadable one, so the reader is what tells them apart."""
    acquisition.frame()
    live = UimfFile(acquisition.path)
    assert live.refresh().frames == (1,)

    acquisition.close()  # the acquisition software closes both files, then removes one
    os.remove(acquisition.path)

    with pytest.raises(FileGone):
        live.refresh()
    with pytest.raises(FileNotFoundError):
        live.refresh()  # catchable as the ordinary thing it is, as well as as itself


def test_nothing_holds_the_followed_file_open_between_calls(acquisition):
    """The delete landing at all is the never-do rule paying off in a way nobody
    designed for. Windows refuses to delete a file any process has open, so a reader
    that kept a connection across polls would turn the acquisition software's
    `os.remove` into a `PermissionError` and `keep_raw = false` into a silent no-op.

    Asserted rather than assumed: nothing else in the suite would notice if a connection
    started outliving a call.
    """
    acquisition.frame()
    live = UimfFile(acquisition.path)
    live.refresh()
    live.read_frame(1)
    live.frame_grouping()
    acquisition.close()

    os.remove(acquisition.path)  # raises PermissionError on Windows if anything is open
    assert not os.path.exists(acquisition.path)


def test_the_summed_companion_is_the_one_beside_the_raw_file(tmp_path):
    """`<stem>.summed.uimf` beside `<stem>.uimf`, and only when it is really there."""
    raw = tmp_path / "run.uimf"
    raw.write_bytes(b"")
    assert summed_companion(raw) is None  # the fold has not written one yet

    companion = tmp_path / "run.summed.uimf"
    companion.write_bytes(b"")
    assert summed_companion(raw) == os.path.abspath(companion)
    assert summed_companion(companion) is None  # a companion is not its own
    assert summed_companion(tmp_path / "notes.txt") is None
    assert summed_companion(tmp_path / "never-written.uimf") is None


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
    an entry that can never be chosen on this file is not a choice.

    `Sum newest frames` is the other half of the same assertion: it is offered here,
    because a window of the newest finished frames needs nothing of the file but its
    frame list, which is why it exists (lab record, task 26)."""
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
    assert offered == [FOLLOW_FIXED, FOLLOW_NEWEST, FOLLOW_ROLLING_SUM]


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
    # A poll that failed for any other reason still reads the way it always did: the
    # message is the exception's, and there is nothing to offer instead of the file.
    assert window.companion_offer is None


def _remove_between_polls(path: str, qtbot) -> int:
    """Delete `path`, retrying until no poll is in flight. Returns the tries it took.

    Nothing outlives a call, which is what makes the delete possible at all. But a poll
    *in flight* holds the file for as long as its query takes, and Windows will not
    delete a file any process has open -- so at this suite's interval, a twentieth of
    the real one, the delete takes a try or two. The acquisition software's own removal
    is a single attempt inside a suppressed `OSError`, which is the other half of this
    and is that repository's to fix (lab record, task 28).
    """
    for tries in range(1, 2001):
        try:
            os.remove(path)
            return tries
        except OSError:
            qtbot.wait(1)
    raise AssertionError(f"{path} was still held open after {tries} tries")


def _fold(raw_path: str) -> str:
    """Write the summed companion a run's fold would have left beside its raw file."""
    companion = raw_path[: -len(".uimf")] + ".summed.uimf"
    folded = Acquisition(companion)
    folded.frame()
    folded.close()
    return companion


def _run_ends_discarding_its_raw_file(acquisition, qtbot, *, fold: bool = True) -> str:
    """The close of a run with `keep_raw = false`: both files closed, the raw one gone.

    In that order, because it is the real one -- the acquisition software closes what it
    has open before it removes anything, and a raw file its own writer still held could
    not be deleted either.
    """
    companion = _fold(acquisition.path) if fold else ""
    acquisition.close()
    _remove_between_polls(acquisition.path, qtbot)
    return companion


def test_a_run_that_discards_its_raw_file_says_so_and_offers_the_companion(
        following_window, qtbot):
    """The whole point of task 28. The operator watching a run should read that the file
    went, not sqlite's wording for a file it cannot open, and should be given the file
    their data actually ended up in."""
    window, acquisition, _ = following_window
    companion = _run_ends_discarding_its_raw_file(acquisition, qtbot)

    qtbot.waitUntil(lambda: not window.following, timeout=5000)
    said = window.status_text()
    assert os.path.basename(acquisition.path) in said
    assert "is no longer there" in said
    assert "database" not in said.lower() and "sqlite" not in said.lower()
    assert os.path.normcase(str(window.companion_offer)) == os.path.normcase(companion)
    assert window._offer.text() == f"Open {os.path.basename(companion)}"


def test_the_companion_is_offered_and_not_opened(following_window, qtbot):
    """Offered rather than opened, because opening it would change which frames are on
    screen while the operator was looking somewhere else (Matt, 2026-09-18)."""
    window, acquisition, painted = following_window
    raw = acquisition.path
    _run_ends_discarding_its_raw_file(acquisition, qtbot)
    qtbot.waitUntil(lambda: not window.following, timeout=5000)

    assert window._path == raw
    drawn = len(painted)
    qtbot.wait(150)  # several poll intervals: nothing is watching and nothing repaints
    assert len(painted) == drawn


def test_clicking_the_offer_opens_the_companion_and_takes_it_down(following_window,
                                                                 qtbot):
    """A shortcut past the file dialog, not a second way of opening a file: it goes
    through `open_file`, so everything else about the open is unchanged."""
    window, acquisition, painted = following_window
    companion = _run_ends_discarding_its_raw_file(acquisition, qtbot)
    qtbot.waitUntil(lambda: not window.following, timeout=5000)

    drawn = len(painted)
    window._offer.click()
    qtbot.waitUntil(lambda: len(painted) > drawn, timeout=5000)

    assert os.path.normcase(window._path) == os.path.normcase(companion)
    assert window.companion_offer is None
    assert window._offer.isHidden()


def test_a_discarded_raw_file_with_no_companion_still_says_what_happened(
        following_window, qtbot):
    """A run cut short before its first fold leaves no companion at all. There is still
    a sentence to read; there is simply nothing to offer."""
    window, acquisition, _ = following_window
    _run_ends_discarding_its_raw_file(acquisition, qtbot, fold=False)

    qtbot.waitUntil(lambda: not window.following, timeout=5000)
    assert "is no longer there" in window.status_text()
    assert window.companion_offer is None
    assert window._offer.isHidden()


def test_opening_another_file_takes_down_an_offer_nobody_answered(following_window,
                                                                 qtbot, tmp_path):
    """The offer belongs to one run's ending. Whatever the operator opens next answers
    it, including a file they chose themselves."""
    window, acquisition, painted = following_window
    _run_ends_discarding_its_raw_file(acquisition, qtbot)
    qtbot.waitUntil(lambda: window.companion_offer is not None, timeout=5000)

    other = Acquisition(tmp_path / "other.uimf")
    other.frame()
    drawn = len(painted)
    window.open_file(other.path)
    qtbot.waitUntil(lambda: len(painted) > drawn, timeout=5000)
    other.close()

    assert window.companion_offer is None
    assert window._offer.isHidden()


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


# --- the rolling sum ------------------------------------------------------------------

@pytest.fixture
def rolling_window(qtbot, tmp_path, quick_poll):
    """A window following a file that says nothing about how its frames group, in
    `Sum newest frames` over a window of two.

    Ungrouped on purpose: the rolling sum is the mode that works on a file the
    method-frame sum cannot be offered on. Two frames rather than the default five, so
    that a handful of frames is enough to watch the window slide.
    """
    running = Acquisition(tmp_path / "rolling.uimf")
    running.frame()
    window = MainWindow()
    qtbot.addWidget(window)
    painted: list = []
    window.frame_shown.connect(painted.append)
    window.open_file(running.path)
    qtbot.waitUntil(lambda: bool(painted), timeout=5000)
    window._rolling_sum_spin.setValue(2)
    window._follow_action.setChecked(True)
    window._follow_mode.setCurrentText(FOLLOW_ROLLING_SUM)
    assert window.following
    yield window, running
    running.close()


def test_the_rolling_sum_is_a_window_that_slides_over_the_finished_frames(rolling_window,
                                                                         qtbot):
    """The whole mode in one sequence: it fills to the length asked for, then moves.

    A fixed integration time that keeps moving through a run is what distinguishes it
    from the method-frame sum, which restarts at every method-frame boundary and only
    ever grows.
    """
    window, running = rolling_window
    qtbot.waitUntil(lambda: window._live_sum_frames == (1,), timeout=5000)

    running.frame()
    qtbot.waitUntil(lambda: window._live_sum_frames == (1, 2), timeout=5000)
    running.frame()
    qtbot.waitUntil(lambda: window._live_sum_frames == (2, 3), timeout=5000)


def test_the_rolling_sum_leaves_out_a_frame_that_is_still_being_written(rolling_window,
                                                                       qtbot):
    """The rule it inherits from the method-frame sum. A total that included a frame
    still being written would change every time it was recomputed and would settle on
    wherever the operator happened to stop."""
    window, running = rolling_window
    qtbot.waitUntil(lambda: window._live_sum_frames == (1,), timeout=5000)

    running.start()  # frame 2, declared and partly written, not finalised
    qtbot.waitUntil(lambda: window._frame_spin.maximum() == 2, timeout=5000)
    assert window._live_sum_frames == (1,)

    running.finish()
    qtbot.waitUntil(lambda: window._live_sum_frames == (1, 2), timeout=5000)


def test_the_rolling_sum_totals_what_the_frames_own_tic_columns_say(rolling_window, qtbot):
    """Against the file's own summary columns, which are ground truth for a decode (lab
    record, task 01) -- so this checks the total and not just that one arrived. The
    status line names the window by the frames it added rather than by the frames it was
    asked for."""
    window, running = rolling_window
    running.frame()
    qtbot.waitUntil(
        lambda: "Rolling sum of the 2 newest finished frames" in window.status_text(),
        timeout=5000,
    )
    assert window._current_frame_number == 0

    file = UimfFile(running.path)
    expected = sum(float(file.scan_summary(n)[3].sum()) for n in (1, 2))
    assert float(window._current_frame.intensity.sum()) == pytest.approx(expected)


def test_the_rolling_sum_says_how_many_frames_it_found_when_it_is_short(qtbot, tmp_path,
                                                                       quick_poll):
    """A run that has just started has fewer finished frames than the spinner asks for.
    It sums what there is and says so, rather than waiting for a window it may never
    get."""
    running = Acquisition(tmp_path / "short.uimf")
    running.frame()
    window = MainWindow()
    qtbot.addWidget(window)
    painted: list = []
    window.frame_shown.connect(painted.append)
    window.open_file(running.path)
    qtbot.waitUntil(lambda: bool(painted), timeout=5000)
    try:
        window._rolling_sum_spin.setValue(10)
        window._follow_action.setChecked(True)
        window._follow_mode.setCurrentText(FOLLOW_ROLLING_SUM)
        qtbot.waitUntil(
            lambda: "Rolling sum of the newest finished frame:" in window.status_text(),
            timeout=5000,
        )
        assert window._live_sum_frames == (1,)
    finally:
        running.close()


def test_a_poll_that_finds_nothing_new_does_not_re_total_the_same_frames(rolling_window,
                                                                        qtbot, monkeypatch):
    """The other rule the method-frame sum set: recomputed only when the set of frames
    has actually changed.

    Counted on the sum itself rather than on the frames it would read, because the load
    worker caches a finished frame -- so a re-added window would be cheap enough to hide
    in a timing, and would still be the load worker doing work once a second on a thread
    that is waiting for a frame (lab record, task 17).
    """
    window, running = rolling_window
    running.frame()
    qtbot.waitUntil(lambda: window._live_sum_frames == (1, 2), timeout=5000)
    qtbot.waitUntil(
        lambda: "Rolling sum of the 2 newest finished frames" in window.status_text(),
        timeout=5000,
    )

    polls: list = []
    sums: list = []
    window._worker.summed.connect(lambda *_: sums.append(1))
    real_refresh = UimfFile.refresh

    def counting_refresh(self):
        polls.append(1)
        return real_refresh(self)

    monkeypatch.setattr(UimfFile, "refresh", counting_refresh)
    qtbot.waitUntil(lambda: len(polls) >= 3, timeout=5000)
    qtbot.wait(50)  # a sum any of those polls had asked for would have landed by now
    assert sums == []
    assert window._live_sum_frames == (1, 2)


def test_changing_how_many_frames_to_sum_re_totals_at_once(rolling_window, qtbot):
    """The spinner is how an operator trades signal against latency while a run is going
    on, so it acts on the poll that has already happened rather than on the next one."""
    window, running = rolling_window
    running.frame()
    running.frame()
    qtbot.waitUntil(lambda: window._live_sum_frames == (2, 3), timeout=5000)

    window._rolling_sum_spin.setValue(3)
    assert window._live_sum_frames == (1, 2, 3)  # no wait: the change is what recomputes
    assert window.settings.rolling_sum_frames == 3


# --- starting the viewer on a run in progress -----------------------------------------

def test_a_launch_lands_following_in_the_mode_it_asked_for(qtbot, acquisition, quick_poll):
    """What the clockwork window's button asks for: a file open, `Follow` on and `Show`
    where it was told, on a window the trainee has not had to touch."""
    acquisition.frame()
    window = MainWindow()
    qtbot.addWidget(window)
    painted: list = []
    window.frame_shown.connect(painted.append)

    window.follow_when_opened(show=FOLLOW_NEWEST)
    window.open_file(acquisition.path, from_command_line=True)
    qtbot.waitUntil(lambda: window.following, timeout=5000)

    assert window._follow_mode.currentText() == FOLLOW_NEWEST
    assert window._follow_mode.isEnabled() and window._rolling_sum_spin.isEnabled()
    acquisition.frame()
    qtbot.waitUntil(lambda: window._current_frame_number == 2, timeout=5000)


def test_a_launch_can_follow_a_file_that_has_no_frames_in_it_yet(qtbot, tmp_path,
                                                                quick_poll):
    """The case the launch route exists for: the acquisition software creates the file
    and starts the viewer on it before the first experiment has finished.

    The first frame then arrives from the poll rather than from the open, and it is
    still the first sight of this file -- so the view is that frame's own full range
    rather than the empty window it would otherwise have been kept at.
    """
    running = Acquisition(tmp_path / "notyet.uimf")
    window = MainWindow()
    qtbot.addWidget(window)
    painted: list = []
    window.frame_shown.connect(painted.append)
    try:
        window.follow_when_opened(show=FOLLOW_NEWEST)
        window.open_file(running.path, from_command_line=True)
        qtbot.waitUntil(lambda: window.following, timeout=5000)

        running.frame()
        qtbot.waitUntil(lambda: bool(painted), timeout=5000)
        assert window._current_frame_number == 1
        result = painted[-1]
        assert (result.x_range, result.y_range) == result.axes.full_range
    finally:
        running.close()


def test_a_launch_on_a_file_that_cannot_be_followed_still_opens_it(qtbot, monkeypatch,
                                                                  acquisition):
    """The file is what the person was trying to look at. The refusal is the toolbar's
    own, in the status bar rather than in a modal in front of a running instrument."""
    acquisition.frame()
    window = MainWindow()
    qtbot.addWidget(window)
    painted: list = []
    window.frame_shown.connect(painted.append)

    monkeypatch.setattr("mainspring.viewer.main_window.is_local_path", lambda _p: False)
    window.follow_when_opened(show=FOLLOW_NEWEST)
    window.open_file(acquisition.path, from_command_line=True)
    qtbot.waitUntil(lambda: "local drive" in window.status_text(), timeout=5000)

    assert not window.following
    assert window._current_frame_number == 1
    assert bool(painted)


def test_a_launch_asking_for_a_mode_this_file_cannot_offer_says_so_and_follows(
    qtbot, tmp_path, quick_poll
):
    """`--show method-sum` on a file that does not record how its frames group. The mode
    is not there to be chosen, which is worth a line in the status bar; the run is still
    worth watching, so following goes on regardless."""
    running = Acquisition(tmp_path / "ungrouped-launch.uimf")
    running.frame()
    window = MainWindow()
    qtbot.addWidget(window)
    painted: list = []
    window.frame_shown.connect(painted.append)
    try:
        window.follow_when_opened(show=FOLLOW_METHOD_SUM)
        window.open_file(running.path, from_command_line=True)
        qtbot.waitUntil(lambda: window.following, timeout=5000)

        assert "not offered" in window.status_text()
        assert window._follow_mode.currentText() == FOLLOW_FIXED
    finally:
        running.close()


def test_an_open_that_fails_does_not_leave_a_launch_waiting(qtbot, tmp_path):
    """A window whose first file did not open is a window with File > Open in front of
    it, and the file chosen there is the user's own rather than the one a program asked
    to follow."""
    bad = tmp_path / "not-a-uimf.uimf"
    bad.write_bytes(b"this is not a database")
    window = MainWindow()
    qtbot.addWidget(window)

    window.follow_when_opened(show=FOLLOW_NEWEST)
    window.open_file(os.fspath(bad), from_command_line=False)
    qtbot.waitUntil(lambda: window.status_text().startswith("Error:"), timeout=5000)
    assert not window._launch_follow


def test_every_show_mode_has_a_command_line_word_and_every_word_a_mode():
    """The words are the interface another program holds; the labels are what the window
    happens to say today. A mode added without a word would be a mode the clockwork
    window cannot ask for."""
    assert sorted(FOLLOW_MODE_WORDS.values()) == sorted(FOLLOW_MODES)
    assert all(word == word.lower() and " " not in word for word in FOLLOW_MODE_WORDS)
    # And the words themselves are `mainspring.interface`'s, not this window's: that is
    # the copy the acquisition software imports (lab record, task 27).
    assert sorted(FOLLOW_MODE_WORDS) == sorted(interface.SHOW_WORDS)
    # `--help` offers the same words, since a word nobody is told about is not offered.
    assert all(word in viewer_app.USAGE for word in interface.SHOW_WORDS)


def test_the_command_line_reads_a_path_and_how_to_look_at_it():
    parsed = parse_arguments(["run.uimf", "--follow", "--show", "rolling-sum"])
    assert parsed.path == "run.uimf"
    # The word as typed, not the label it names. The parse deals in words alone so that
    # it needs no Qt; the window translates one into the other when it applies them.
    assert parsed.follow and parsed.show == "rolling-sum"
    assert FOLLOW_MODE_WORDS[parsed.show] == FOLLOW_ROLLING_SUM
    assert parse_arguments([]) == Launch()
    assert parse_arguments(["--show=method-sum"]).show == "method-sum"
    assert FOLLOW_MODE_WORDS["method-sum"] == FOLLOW_METHOD_SUM


def test_an_option_the_viewer_does_not_offer_is_an_error_and_never_a_file(monkeypatch,
                                                                         tmp_path, capsys):
    """A mistyped `--folow` taken as a positional argument would be a file by that name,
    and the viewer would complain about a file nobody asked for instead of failing where
    the mistake is. Exit 2, the message on stderr, and no window."""
    monkeypatch.setenv("NUMBA_CACHE_DIR", os.fspath(tmp_path))

    assert viewer_main(["--folow"]) == 2
    assert "--folow" in capsys.readouterr().err
    assert viewer_main(["run.uimf", "--show", "sideways"]) == 2
    assert "rolling-sum" in capsys.readouterr().err  # what it does take
    assert viewer_main(["--help"]) == 0
    assert "--show" in capsys.readouterr().out
