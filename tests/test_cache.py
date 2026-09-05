"""`FrameCache`: least recently used, bounded by bytes, and closed to provisional frames.

Two things distinguish this from any other LRU and both are tested here: the budget is
in bytes rather than in entries, because frames differ in size by orders of magnitude,
and a frame read from a file the instrument may still be writing is refused outright
rather than cached and regretted (lab record, task 08).
"""

from __future__ import annotations

import numpy as np

from mainspring.uimf.cache import DEFAULT_BUDGET_BYTES, FrameCache
from mainspring.uimf.frame import SparseFrame


def frame(number: int, points: int = 100, provisional: bool = False) -> SparseFrame:
    return SparseFrame.from_scans(
        number, 4, 1000,
        {1: (np.arange(points, dtype=np.int32), np.ones(points, dtype=np.int32))},
        provisional,
    )


def test_a_miss_is_none_and_a_hit_is_the_frame():
    cache = FrameCache()
    assert cache.get("a.uimf", 1) is None
    stored = frame(1)
    assert cache.put("a.uimf", stored)
    assert cache.get("a.uimf", 1) is stored


def test_the_key_is_the_path_as_well_as_the_frame():
    """One cache serves a session that opens several files, and reopening a different
    file at the same frame number must not find the old one."""
    cache = FrameCache()
    cache.put("a.uimf", frame(1))
    assert cache.get("b.uimf", 1) is None


def test_nbytes_is_what_it_is_holding():
    cache = FrameCache()
    first, second = frame(1, 100), frame(2, 300)
    cache.put("a.uimf", first)
    cache.put("a.uimf", second)
    assert cache.nbytes == first.nbytes + second.nbytes
    assert len(cache) == 2


def test_it_evicts_the_least_recently_used_until_it_fits():
    one, two, three = frame(1), frame(2), frame(3)
    cache = FrameCache(budget_bytes=one.nbytes + two.nbytes)
    cache.put("a.uimf", one)
    cache.put("a.uimf", two)
    cache.get("a.uimf", 1)  # frame 1 is now the most recently used
    cache.put("a.uimf", three)
    assert cache.get("a.uimf", 2) is None, "the least recently used went"
    assert cache.get("a.uimf", 1) is not None
    assert cache.get("a.uimf", 3) is not None


def test_re_putting_a_frame_does_not_count_it_twice():
    cache = FrameCache()
    stored = frame(1)
    cache.put("a.uimf", stored)
    cache.put("a.uimf", stored)
    assert len(cache) == 1 and cache.nbytes == stored.nbytes


def test_a_frame_larger_than_the_whole_budget_is_refused_not_ruinous():
    """Evicting everything else to fail anyway would lose a working cache to a frame
    that was never going to fit."""
    cache = FrameCache(budget_bytes=100)
    keeper = frame(1, points=2)
    assert cache.put("a.uimf", keeper)
    assert not cache.put("a.uimf", frame(2, points=10_000))
    assert cache.get("a.uimf", 1) is keeper


def test_a_provisional_frame_is_never_cached():
    cache = FrameCache()
    assert not cache.put("a.uimf", frame(1, provisional=True))
    assert cache.get("a.uimf", 1) is None


def test_clear_drops_everything():
    cache = FrameCache()
    cache.put("a.uimf", frame(1))
    cache.clear()
    assert len(cache) == 0 and cache.nbytes == 0


def test_the_default_budget_is_a_number_a_setting_can_show():
    assert FrameCache().budget_bytes == DEFAULT_BUDGET_BYTES
