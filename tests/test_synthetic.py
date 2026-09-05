"""The synthetic writer: a real UIMF file with no acquisition in it.

The fixture is what lets a bare clone exercise the decode path at all, so it is worth
testing that what it writes is the schema real files carry and that the awkward things
about real files -- two parameter tables that disagree, scans that are simply absent,
zeros counted in `NonZeroCount` -- are present rather than smoothed away.
"""

from __future__ import annotations

import sqlite3

import pytest

from synthetic import write_synthetic_uimf


def tables(path):
    conn = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()


def test_it_writes_the_tables_a_2026_file_carries(synthetic_uimf):
    names = tables(synthetic_uimf.path)
    assert {"Global_Params", "Frame_Param_Keys", "Frame_Params", "V_Frame_Params"} <= names
    assert {"Global_Parameters", "Frame_Parameters"} <= names, "legacy twins too"
    assert {"Frame_Scans", "pk_index_FrameScans"} <= names


def test_legacy_only_and_modern_only_are_what_they_say(tmp_path):
    legacy = write_synthetic_uimf(tmp_path / "legacy.uimf", legacy_only=True, frames=1, scans=8)
    modern = write_synthetic_uimf(tmp_path / "modern.uimf", modern_only=True, frames=1, scans=8)
    assert "Global_Params" not in tables(legacy.path)
    assert "Global_Parameters" in tables(legacy.path)
    assert "Global_Params" in tables(modern.path)
    assert "Global_Parameters" not in tables(modern.path)


def test_the_two_parameter_tables_disagree_about_frame_type(synthetic_uimf):
    """Our sample says 0 modern and 1 legacy for the same frame; a reader that picks the
    wrong table must be caught here rather than on real data (lab record, task 01)."""
    conn = sqlite3.connect("file:" + synthetic_uimf.path + "?mode=ro", uri=True)
    try:
        modern = conn.execute(
            "SELECT ParamValue FROM V_Frame_Params WHERE FrameNum = 1 AND ParamName = 'FrameType'"
        ).fetchone()
        legacy = conn.execute(
            "SELECT FrameType FROM Frame_Parameters WHERE FrameNum = 1"
        ).fetchone()
    finally:
        conn.close()
    assert modern[0] == "0"
    assert legacy[0] == 1


def test_not_every_scan_is_stored_and_numbering_does_not_start_at_zero(synthetic_uimf):
    stored = synthetic_uimf.stored_scans(1)
    assert stored, "a frame must have some signal"
    assert len(stored) < synthetic_uimf.scans, "empty scans are absent, as SLIMPHONY writes them"
    assert min(stored) > 0


def test_store_all_scans_writes_the_empty_ones_too(tmp_path):
    spec = write_synthetic_uimf(tmp_path / "all.uimf", frames=1, scans=8, store_all_scans=True)
    assert spec.stored_scans(1) == list(range(8))
    assert spec.scan(1, 0).bin_index.size == 0, "an empty scan has a row and no points"


def test_stored_columns_agree_with_the_points(synthetic_uimf):
    conn = sqlite3.connect("file:" + synthetic_uimf.path + "?mode=ro", uri=True)
    try:
        rows = list(conn.execute(
            "SELECT FrameNum, ScanNum, NonZeroCount, BPI, BPI_MZ, TIC, Intensities"
            " FROM Frame_Scans"
        ))
    finally:
        conn.close()
    assert len(rows) == len(synthetic_uimf.scan_rows)
    for frame, scan, non_zero, bpi, bpi_mz, tic, blob in rows:
        want = synthetic_uimf.scan(frame, scan)
        assert tic == int(want.intensity.sum())
        assert bpi == int(want.intensity.max()) if want.intensity.size else bpi == 0
        assert non_zero >= want.bin_index.size, "NonZeroCount is an upper bound"
        assert blob, "a scan with signal has a blob"
        if want.bin_index.size:
            assert bpi_mz > 0


def test_explicit_zeros_push_non_zero_count_above_the_point_count(tmp_path):
    with_zeros = write_synthetic_uimf(tmp_path / "z.uimf", frames=1, scans=8)
    without = write_synthetic_uimf(tmp_path / "nz.uimf", frames=1, scans=8, explicit_zeros=False)
    counted = sum(row.non_zero_count for row in with_zeros.scan_rows.values())
    points = sum(row.bin_index.size for row in with_zeros.scan_rows.values())
    assert counted > points
    assert all(r.non_zero_count == r.bin_index.size for r in without.scan_rows.values())


@pytest.mark.parametrize("intensity_type,width", [("ADC", 4), ("TDC", 2), ("FOLDED", 4)])
def test_every_intensity_element_type_can_be_written(tmp_path, intensity_type, width):
    spec = write_synthetic_uimf(
        tmp_path / f"{intensity_type}.uimf", frames=1, scans=8, tof_intensity_type=intensity_type
    )
    assert spec.dtype.itemsize == width
    assert spec.points(1) > 0


def test_bpi_mz_is_the_calibration_of_the_argmax_bin(synthetic_uimf):
    """Exact here, unlike every real writer -- which is the point of having a fixture."""
    row = synthetic_uimf.scan(1, synthetic_uimf.stored_scans(1)[0])
    argmax = int(row.intensity.argmax())
    t_us = float(row.bin_index[argmax]) * synthetic_uimf.bin_width_ns / 1000.0
    expected = (synthetic_uimf.slope * (t_us - synthetic_uimf.intercept)) ** 2
    assert row.bpi_mz == pytest.approx(expected, rel=1e-12)


def test_the_file_stays_small_enough_to_write_in_a_test(synthetic_uimf):
    import os

    assert os.path.getsize(synthetic_uimf.path) < 1_000_000
