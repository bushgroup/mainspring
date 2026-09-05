"""`SparseFrame` and `sum_frames`: the CSR layout, its accessors, and adding frames up.

The layout is worth testing directly rather than only through the reader, because two
of its properties are what everything downstream assumes: a scan nobody stored is an
empty slice rather than a missing key, and the axis extent is the frame parameters'
rather than the data's. Both are differences between writers that CSR is there to make
invisible (lab record, task 01).
"""

from __future__ import annotations

import numpy as np
import pytest

from mainspring.uimf.frame import SparseFrame, sum_frames
from mainspring.uimf.reader import UimfFile


def build(points, scans=8, bins=1000, frame=1, provisional=False):
    return SparseFrame.from_scans(frame, scans, bins, points, provisional)


@pytest.fixture
def small():
    return build({
        2: (np.array([3, 9, 40]), np.array([5, 7, 1])),
        5: (np.array([9]), np.array([2])),
    })


def test_from_scans_builds_the_row_pointer(small):
    assert small.scan_start.tolist() == [0, 0, 0, 3, 3, 3, 4, 4, 4]
    assert len(small) == 3 + 1
    assert small.extent == (8, 1000)


def test_a_scan_nobody_stored_is_an_empty_slice(small):
    bins, values = small.scan(0)
    assert bins.size == 0 and values.size == 0


def test_a_scan_outside_the_axis_is_an_index_error(small):
    with pytest.raises(IndexError):
        small.scan(8)


def test_from_scans_refuses_a_scan_outside_the_axis():
    with pytest.raises(ValueError, match="outside"):
        build({9: (np.array([1]), np.array([1]))}, scans=8)


def test_scan_returns_views_not_copies(small):
    bins, _ = small.scan(2)
    assert bins.base is small.bin_index


def test_scan_of_expands_the_row_pointer(small):
    assert small.scan_of().tolist() == [2, 2, 2, 5]


def test_tic_and_bpi_cover_every_scan_including_the_empty_ones(small):
    assert small.tic().tolist() == [0, 0, 13, 0, 0, 2, 0, 0]
    assert small.bpi().tolist() == [0, 0, 7, 0, 0, 2, 0, 0]
    assert small.bpi_bin().tolist() == [-1, -1, 9, -1, -1, 9, -1, -1]


def test_tic_is_exact_on_large_integer_intensities():
    """A float accumulation over a real frame's points drifts; the cumulative sum is in
    int64 for integer intensities on purpose."""
    values = np.full(200000, 2_000_003, dtype=np.int32)
    frame = build({0: (np.arange(200000), values)}, scans=1, bins=200000)
    assert frame.tic()[0] == int(values.sum(dtype=np.int64))


def test_nbytes_counts_the_index_too(small):
    assert small.nbytes == (small.scan_start.nbytes + small.bin_index.nbytes
                            + small.intensity.nbytes)


def test_slice_keeps_the_axis_and_empties_what_it_excludes(small):
    part = small.slice(scan_range=(2, 3))
    assert part.extent == small.extent
    assert len(part) == 3
    assert part.scan(5)[0].size == 0
    assert part.scan(2)[0].tolist() == [3, 9, 40]


def test_slice_on_bins_is_half_open(small):
    part = small.slice(bin_range=(3, 40))
    assert part.scan(2)[0].tolist() == [3, 9]
    assert part.scan(5)[0].tolist() == [9]


def test_slicing_everything_away_leaves_a_frame_with_its_axis(small):
    part = small.slice(scan_range=(7, 7))
    assert part.extent == small.extent
    assert len(part) == 0
    assert part.scan_start.tolist() == [0] * 9


def test_an_empty_frame_is_a_frame(small):
    empty = build({}, scans=4, bins=100)
    assert len(empty) == 0
    assert empty.tic().tolist() == [0, 0, 0, 0]
    assert empty.bpi().tolist() == [0, 0, 0, 0]


# --- sum_frames ------------------------------------------------------------------------


def test_sum_frames_adds_point_wise(small):
    total = sum_frames([small, small])
    assert total.tic().tolist() == [0, 0, 26, 0, 0, 4, 0, 0]
    assert total.scan(2)[0].tolist() == [3, 9, 40]
    assert total.scan(2)[1].tolist() == [10, 14, 2]


def test_sum_frames_merges_frames_that_do_not_overlap():
    a = build({1: (np.array([5]), np.array([3]))})
    b = build({1: (np.array([7]), np.array([4]))})
    total = sum_frames([a, b])
    assert total.scan(1)[0].tolist() == [5, 7]
    assert total.scan(1)[1].tolist() == [3, 4]


def test_a_sum_that_included_a_provisional_frame_is_provisional():
    """Propagated rather than refused: "sum all" must work on a live file, and the flag
    is what keeps the result out of the cache."""
    settled = build({1: (np.array([5]), np.array([3]))})
    growing = build({1: (np.array([7]), np.array([4]))}, provisional=True)
    assert not sum_frames([settled, settled]).provisional
    assert sum_frames([settled, growing]).provisional


def test_sum_frames_can_be_cancelled(small):
    assert sum_frames([small, small], should_cancel=lambda: True) is None


def test_sum_frames_of_nothing_is_an_empty_frame():
    total = sum_frames([])
    assert len(total) == 0 and total.extent == (0, 0)


def test_summing_a_real_file_totals_its_per_frame_tic(real_uimf):
    """The check the whole thing is for: every frame added up must carry the same total
    ion current as the frames did separately."""
    uimf = UimfFile(real_uimf)
    numbers = uimf.frame_numbers()[:6]
    frames = [uimf.read_frame(n) for n in numbers]
    want = sum(float(uimf.scan_summary(n)[3].sum()) for n in numbers)
    total = sum_frames(frames)
    assert float(total.tic().sum()) == pytest.approx(want)
