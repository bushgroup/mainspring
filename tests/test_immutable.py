"""Reading a finished file without writing beside it: `UimfFile(path, immutable=True)`.

An ordinary read-only open of a WAL database creates the `-shm` index and an empty
`-wal` beside it and, being read-only, leaves them -- every file mainspring's writer
made gets the pair from its first read, in whatever folder it is in. `immutable=1` is
SQLite's way to read without them, and it is only exact on a file that nothing is still
writing and whose commits are all in the database file itself. These tests hold both
halves: that nothing appears beside the file, on either decode backend, and that every
file the option would misread is refused rather than answered from (lab record, task 38).
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from mainspring.uimf import decode as uimf_decode
from mainspring.uimf import frame as uimf_frame
from mainspring.uimf import NotImmutable, UimfFile, numba_available, sum_frames
from mainspring.uimf.writer import (
    OUTCOME_COMPLETED,
    FrameSpec,
    GlobalSpec,
    UimfWriter,
)
from synthetic import write_synthetic_uimf


@pytest.fixture(params=["pure", pytest.param("numba", marks=pytest.mark.skipif(
    not numba_available(), reason="numba is not installed"))])
def backend(request, monkeypatch):
    """Run the test on the compiled path and again with it hidden."""
    if request.param == "pure":
        monkeypatch.setattr(uimf_decode, "_kernels", lambda: None)
        monkeypatch.setattr(uimf_frame, "_sum_kernel", lambda: None)
    return request.param


def _alone(tmp_path, **kwargs):
    """A closed, checkpointed writer-made file, alone in a folder of its own."""
    folder = tmp_path / "data"
    folder.mkdir()
    path = folder / "x.uimf"
    write_synthetic_uimf(path, frames=3, detector_bits=14, **kwargs)
    assert os.listdir(folder) == ["x.uimf"]
    return folder, path


def test_an_ordinary_read_leaves_the_pair_this_exists_to_avoid(tmp_path):
    folder, path = _alone(tmp_path)
    UimfFile(path).global_params()
    assert sorted(os.listdir(folder)) == ["x.uimf", "x.uimf-shm", "x.uimf-wal"]


def test_an_immutable_read_writes_nothing_beside_the_file(tmp_path, backend):
    folder, path = _alone(tmp_path)
    before = path.read_bytes()
    uimf = UimfFile(path, immutable=True)
    params = uimf.global_params()
    frames = [uimf.read_frame(n) for n in uimf.frame_numbers()]
    total = sum_frames(frames)
    clipping = uimf.clipping()
    uimf.frame_totals(), uimf.frame_grouping(), uimf.scan_summary(1)
    assert params.written_by and len(frames) == 3
    assert total is not None and len(total) > 0
    assert clipping.total == 0
    assert os.listdir(folder) == ["x.uimf"]
    assert path.read_bytes() == before


def test_it_answers_what_the_ordinary_open_answers(tmp_path):
    folder, path = _alone(tmp_path)
    still = UimfFile(path, immutable=True)
    frozen = {n: still.read_frame(n) for n in still.frame_numbers()}
    ordinary = UimfFile(path)
    assert still.global_params() == ordinary.global_params()
    for n, frame in frozen.items():
        other = ordinary.read_frame(n)
        assert np.array_equal(frame.intensity, other.intensity)
        assert np.array_equal(frame.bin_index, other.bin_index)


def test_a_file_left_with_a_reader_left_pair_still_opens(tmp_path):
    """The empty `-wal` an earlier ordinary read left is not a log with commits in it."""
    folder, path = _alone(tmp_path)
    UimfFile(path).frame_numbers()
    assert os.path.getsize(f"{path}-wal") == 0
    assert UimfFile(path, immutable=True).frame_numbers() == [1, 2, 3]


def test_a_delete_mode_file_opens(tmp_path):
    folder, path = _alone(tmp_path, journal_mode="delete")
    assert UimfFile(path, immutable=True).frame_numbers() == [1, 2, 3]
    assert os.listdir(folder) == ["x.uimf"]


def test_a_stamped_file_with_no_outcome_and_every_frame_finished_opens(tmp_path):
    """No outcome is every file from a writer caller that never opted into recording one,
    which says nothing about whether its run finished; the completion marker does."""
    folder, path = _alone(tmp_path)
    uimf = UimfFile(path, immutable=True)
    assert uimf.has_completion_markers
    assert "MainspringRunOutcome" not in uimf.global_params().extra
    assert uimf.frame_numbers() == [1, 2, 3]


def test_a_completed_outcome_opens(tmp_path):
    path = tmp_path / "done.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096, records_outcome=True)) as writer:
        writer.add_frame(FrameSpec(scans=8))
        writer.write_scans(1, [(1, np.array([50]), np.array([3]))])
        writer.finalise_frame(1, duration_s=0.5)
        writer.set_outcome(OUTCOME_COMPLETED)
    assert UimfFile(path, immutable=True).global_params().run_outcome == OUTCOME_COMPLETED


# --- the refusals ---------------------------------------------------------------------

ABANDONED_WRITER = (
    "import os, sys\n"
    "import numpy as np\n"
    "from mainspring.uimf import writer as w\n"
    "h = w.UimfWriter(sys.argv[1], w.GlobalSpec(bins=4096, detector_bits=14))\n"
    "for n in (1, 2):\n"
    "    h.add_frame(w.FrameSpec(scans=8), frame=n)\n"
    "    h.write_scans(n, [(1, np.array([50 * n]), np.array([3]))])\n"
    "    h.finalise_frame(n, duration_s=0.5)\n"
    "os._exit(0)\n"  # no close, so no checkpoint: the log is left hot
)


def test_a_hot_write_ahead_log_is_refused(tmp_path):
    """Read immutable, this file has no tables at all, and would say so as corruption --
    or, on a longer run, answer with the frames checkpointed so far and nothing else."""
    path = tmp_path / "cut.uimf"
    subprocess.run([sys.executable, "-c", ABANDONED_WRITER, str(path)],
                   check=True, capture_output=True)
    os.remove(f"{path}-shm")  # as a reboot leaves it
    with pytest.raises(NotImmutable) as refused:
        UimfFile(path, immutable=True)
    assert refused.value.reason == "log"
    assert "default open" in str(refused.value)
    assert not os.path.exists(f"{path}-shm")
    assert UimfFile(path).frame_numbers() == [1, 2]  # and the default open reads it


def test_a_non_empty_rollback_journal_is_refused(tmp_path):
    folder, path = _alone(tmp_path, journal_mode="delete")
    with open(f"{path}-journal", "wb") as journal:
        journal.write(b"\xd9\xd5\x05\xf9\x20\xa1\x63\xd7" + bytes(504))
    with pytest.raises(NotImmutable) as refused:
        UimfFile(path, immutable=True)
    assert refused.value.reason == "log"


def test_an_empty_rollback_journal_is_not_refused(tmp_path):
    folder, path = _alone(tmp_path, journal_mode="delete")
    open(f"{path}-journal", "wb").close()
    assert UimfFile(path, immutable=True).frame_numbers() == [1, 2, 3]


def test_a_last_frame_without_its_completion_marker_is_refused(tmp_path):
    folder, path = _alone(tmp_path, finalise=False)
    with pytest.raises(NotImmutable) as refused:
        UimfFile(path, immutable=True)
    assert refused.value.reason == "unfinished"
    assert os.listdir(folder) == ["x.uimf"]  # the guard's own read wrote nothing either


def test_an_incomplete_outcome_is_refused(tmp_path):
    """A run that was still being acquired, or whose writer never got to say how it ended,
    with every frame it did write finished."""
    path = tmp_path / "running.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096, records_outcome=True)) as writer:
        writer.add_frame(FrameSpec(scans=8))
        writer.write_scans(1, [(1, np.array([50]), np.array([3]))])
        writer.finalise_frame(1, duration_s=0.5)
    # Closed without `set_outcome`: the file still says what it said at create.
    assert UimfFile(path).global_params().run_outcome == "incomplete"
    with pytest.raises(NotImmutable) as refused:
        UimfFile(path, immutable=True)
    assert refused.value.reason == "incomplete"


def test_a_file_that_gains_a_log_after_opening_is_refused_at_the_next_read(tmp_path):
    """`immutable` turns SQLite's own change detection off, so the stat before each
    connection is the only thing that would notice."""
    folder, path = _alone(tmp_path)
    uimf = UimfFile(path, immutable=True)
    uimf.global_params()
    with open(f"{path}-wal", "wb") as log:
        log.write(b"\x37\x7f\x06\x82" + bytes(28))
    with pytest.raises(NotImmutable) as refused:
        uimf.frame_numbers()
    assert refused.value.reason == "log"


def test_a_file_rewritten_after_opening_is_refused_at_the_next_read(tmp_path):
    folder, path = _alone(tmp_path)
    uimf = UimfFile(path, immutable=True)
    uimf.global_params()
    with UimfWriter(tmp_path / "other.uimf", GlobalSpec(bins=4096)) as writer:
        writer.add_frame(FrameSpec(scans=8))
    os.replace(tmp_path / "other.uimf", path)
    with pytest.raises(NotImmutable) as refused:
        uimf.frame_numbers()
    assert refused.value.reason == "changed"


def test_refusal_is_a_value_error(tmp_path):
    assert issubclass(NotImmutable, ValueError)


# --- the live-viewing entry points under it -------------------------------------------

def test_refresh_refuses_on_an_immutable_open(tmp_path):
    folder, path = _alone(tmp_path)
    with pytest.raises(RuntimeError):
        UimfFile(path, immutable=True).refresh()


def test_is_provisional_is_unchanged(tmp_path):
    folder, path = _alone(tmp_path)
    uimf = UimfFile(path, immutable=True)
    assert [uimf.is_provisional(n) for n in (1, 2, 3)] == [False, False, False]


def test_the_default_open_is_unchanged(tmp_path):
    """Following still works: no guard, no immutable, the run may be growing."""
    path = tmp_path / "live.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(FrameSpec(scans=8))
        writer.write_scans(1, [(1, np.array([50]), np.array([3]))])
        live = UimfFile(path)
        assert live.refresh().provisional == frozenset({1})
        with pytest.raises(NotImmutable):
            UimfFile(path, immutable=True)


def test_immutable_is_keyword_only(tmp_path):
    folder, path = _alone(tmp_path)
    with pytest.raises(TypeError):
        UimfFile(path, 250, True)  # noqa: FBT003 -- the point of the test
