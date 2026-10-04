"""Clipping in `mainspring.uimf`: the full scale, the file-wide count, and the frame's own.

Qt-free, on `synthetic.write_clipping_uimf`: four frames at three `Accumulations`, whose
every point is known, so each count here is compared against brute force over the points
rather than against a number someone wrote down. The two things most worth catching are
a threshold taken from the wrong frame and a scan one count below full scale being
counted, or decoded at all (lab record, task 36).
"""

from __future__ import annotations

import pytest

from mainspring.uimf import (
    STORED_SAMPLE_BITS,
    Clipping,
    UimfFile,
    full_scale,
)
from mainspring.uimf import reader as uimf_reader
from synthetic import write_clipping_uimf

STORED = 2 ** STORED_SAMPLE_BITS - 1


@pytest.fixture
def clipping_file(tmp_path):
    return write_clipping_uimf(tmp_path / "clipping.uimf")


# --- the full scale -----------------------------------------------------------------------

def test_a_depth_the_file_declares_is_stored_as_sixteen_bits():
    """The console stores a 14-bit ADC's samples left-aligned in 16 bits, so a push of a
    file that says 14 runs to 65535, and a wider depth is taken at its word."""
    assert full_scale(14, from_file=True) == 65535 == STORED
    assert full_scale(16, from_file=True) == 65535
    assert full_scale(20, from_file=True) == 2 ** 20 - 1


def test_a_depth_set_by_hand_is_the_stored_width():
    assert full_scale(8, from_file=False) == 255
    assert full_scale(14, from_file=False) == 16383
    assert full_scale(0, from_file=False) == 1, "never a zero divisor"


def test_a_file_declares_its_full_scale_only_if_it_declares_its_depth(tmp_path):
    declared = write_clipping_uimf(tmp_path / "declared.uimf")
    silent = write_clipping_uimf(tmp_path / "silent.uimf", detector_bits=None)
    assert UimfFile(declared.path).global_params().full_scale == STORED
    assert UimfFile(silent.path).global_params().full_scale is None


# --- the file-wide count ------------------------------------------------------------------

def test_the_count_is_every_point_at_its_own_frames_full_scale(clipping_file):
    found = UimfFile(clipping_file.path).clipping()
    clipped, over = clipping_file.expected(STORED)
    assert found.full_scale == STORED
    assert dict(found.clipped) == clipped == {1: 2, 2: 1, 4: 3}
    assert dict(found.over) == over == {}
    assert (found.total, found.frames, found.over_total) == (6, 3, 0)


def test_a_full_scale_too_small_counts_points_above_it_as_well(tmp_path):
    """The case PNNL's 2011 and 2016 excerpts are at the default 8 bits: points *above*
    full scale, which no push can read, and which the count reports beside the clipped
    ones so that a caller can say the depth is wrong."""
    silent = write_clipping_uimf(tmp_path / "silent.uimf", detector_bits=None)
    file = UimfFile(silent.path)
    with pytest.raises(ValueError, match="does not declare"):
        file.clipping()
    found = file.clipping(full_scale(8, from_file=False))
    clipped, over = silent.expected(255)
    assert dict(found.clipped) == clipped and dict(found.over) == over
    assert found.over_total > 0


def test_only_the_scans_whose_bpi_reaches_full_scale_are_decoded(clipping_file, monkeypatch):
    """The whole cost model: `BPI` is each scan's largest value, so a scan below its
    frame's threshold -- including the two one count below it -- is never decompressed."""
    decoded: list[int] = []
    real = uimf_reader.decode_frame_blobs

    def counting(blobs, *args, **kwargs):
        decoded.append(len(blobs))
        return real(blobs, *args, **kwargs)

    monkeypatch.setattr(uimf_reader, "decode_frame_blobs", counting)
    UimfFile(clipping_file.path).clipping()
    assert decoded == [clipping_file.candidate_scans(STORED)] == [3]


def test_a_clean_file_decodes_nothing(tmp_path, monkeypatch):
    clean = write_clipping_uimf(tmp_path / "clean.uimf", frames=1, detector_bits=16)
    calls: list = []
    monkeypatch.setattr(uimf_reader, "decode_frame_blobs",
                        lambda *a, **k: calls.append(a))
    found = UimfFile(clean.path).clipping(full_scale=10 ** 6)
    assert (found.total, calls) == (0, [])


def test_a_file_mixing_accumulations_in_many_runs_still_counts_exactly(clipping_file,
                                                                       monkeypatch):
    """Past `_MAX_THRESHOLD_RUNS` the filter falls back to the lowest threshold and the
    exact test happens afterwards; the answer must not change, only what is fetched."""
    monkeypatch.setattr(uimf_reader, "_MAX_THRESHOLD_RUNS", 1)
    found = UimfFile(clipping_file.path).clipping()
    assert dict(found.clipped) == clipping_file.expected(STORED)[0]


def test_a_legacy_only_file_is_counted_from_its_fixed_columns(tmp_path):
    legacy = write_clipping_uimf(tmp_path / "legacy.uimf", tables="legacy", detector_bits=None)
    file = UimfFile(legacy.path)
    assert file.is_legacy_only
    assert file.frame_accumulations() == legacy.accumulations
    assert dict(file.clipping(STORED).clipped) == legacy.expected(STORED)[0]


def test_since_and_until_bound_the_frames_counted(clipping_file):
    file = UimfFile(clipping_file.path)
    assert dict(file.clipping(since=2).clipped) == {2: 1, 4: 3}
    assert dict(file.clipping(since=2, until=4).clipped) == {2: 1}
    assert file.frame_accumulations(since=2, until=4) == {2: 4, 3: 1}


def test_a_newer_answer_replaces_its_own_range_and_keeps_the_rest():
    whole = Clipping(STORED, {1: 2, 2: 1, 4: 3}, {4: 1})
    tail = Clipping(STORED, {4: 5}, {}, since=3)
    merged = whole.merged(tail)
    assert dict(merged.clipped) == {1: 2, 2: 1, 4: 5}
    assert dict(merged.over) == {}
    assert merged.since is None
    with pytest.raises(ValueError):
        whole.merged(Clipping(255, {}, {}))


def test_the_raw_maximum_is_the_largest_stored_value_and_narrows_by_frame(clipping_file):
    file = UimfFile(clipping_file.path)
    assert file.max_intensity() == clipping_file.max_intensity == 262140
    assert file.max_intensity(since=3) == 131070
    assert file.max_intensity(since=99) == 0.0


# --- the frame's own count ----------------------------------------------------------------

def test_a_frame_in_memory_counts_what_the_file_count_says_it_holds(clipping_file):
    """The panel's per-frame row and the file-wide row are two routes to one answer."""
    file = UimfFile(clipping_file.path)
    found = file.clipping()
    for frame in clipping_file.frames:
        threshold = file.frame_params(frame).accumulations * STORED
        assert file.read_frame(frame).count_clipped(threshold) == (
            found.clipped.get(frame, 0), found.over.get(frame, 0)
        )


def test_an_empty_frame_has_nothing_clipped(clipping_file):
    frame = UimfFile(clipping_file.path).read_frame(1, scan_range=(0, 1))
    assert frame.count_clipped(1) == (0, 0)
