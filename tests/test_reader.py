"""`UimfFile`: the parameters, the frame list, and a frame read back as points.

The measure of a reader is not that it produces arrays; it is that the arrays reproduce
the columns the writer put beside them. `TIC` and `BPI` are exact ground truth on every
row and `NonZeroCount` an upper bound (lab record, task 01), so those three comparisons
are what these tests are, on the synthetic file and on whatever real files this clone
has.

The other half is the awkwardness real files carry: two parameter tables that disagree,
a legacy-only 2011 file, scans that were never stored, and scan numbering that does not
start at zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from mainspring.uimf.reader import UimfFile
from synthetic import write_synthetic_uimf


def test_a_missing_file_is_a_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        UimfFile(tmp_path / "nothing.uimf")


def test_it_reads_the_global_parameters(synthetic_uimf):
    params = UimfFile(synthetic_uimf.path).global_params()
    assert params.instrument_name == "SLIM3"
    assert params.bins == synthetic_uimf.bins
    assert params.bin_width_ns == synthetic_uimf.bin_width_ns
    assert params.tof_intensity_type == "ADC"
    assert params.dtype == np.dtype("<i4")
    assert params.num_frames == len(synthetic_uimf.frames)


def test_time_offset_is_read_and_reported(synthetic_uimf):
    """Read, shown, and never applied to the calibration: the value being visible is how
    a reader of this code finds out it is deliberately unused (lab record, task 01)."""
    assert UimfFile(synthetic_uimf.path).global_params().time_offset_ns == 20000.0


def test_it_prefers_the_modern_frame_table_where_both_exist(synthetic_uimf):
    """The fixture writes FrameType 0 in the modern table and 1 in the legacy one, as our
    sample does; picking the wrong table is caught here rather than on real data."""
    uimf = UimfFile(synthetic_uimf.path)
    assert not uimf.is_legacy_only
    assert uimf.frame_params(1).frame_type == 0
    assert uimf.frame_types()[1] == 0


def test_a_legacy_only_file_falls_back(tmp_path):
    spec = write_synthetic_uimf(tmp_path / "legacy.uimf", frames=2, scans=12, legacy_only=True)
    uimf = UimfFile(spec.path)
    assert uimf.is_legacy_only
    assert uimf.frame_numbers() == [1, 2]
    assert uimf.global_params().instrument_name == "SLIM3", "Instrument_Name, one word over"
    params = uimf.frame_params(1)
    assert params.frame_type == 1, "the legacy table's own answer, since there is no other"
    assert params.scans == spec.scans
    assert params.calibration_done is True


def test_frame_type_zero_and_one_both_mean_ms1(synthetic_uimf):
    uimf = UimfFile(synthetic_uimf.path)
    assert uimf.frame_params(1).is_ms1


def test_frame_numbers_come_from_the_parameters_not_from_numframes(synthetic_uimf):
    assert UimfFile(synthetic_uimf.path).frame_numbers() == list(synthetic_uimf.frames)


def test_asking_for_a_frame_that_is_not_there(synthetic_uimf):
    with pytest.raises(KeyError):
        UimfFile(synthetic_uimf.path).frame_params(99)


def test_scan_summary_is_the_stored_columns_untouched(synthetic_uimf):
    scan, non_zero, bpi, tic = UimfFile(synthetic_uimf.path).scan_summary(1)
    assert scan.tolist() == synthetic_uimf.stored_scans(1)
    for i, number in enumerate(scan.tolist()):
        row = synthetic_uimf.scan(1, number)
        assert non_zero[i] == row.non_zero_count
        assert bpi[i] == row.bpi
        assert tic[i] == row.tic


def test_a_frame_reads_back_as_the_points_that_were_written(synthetic_uimf):
    frame = UimfFile(synthetic_uimf.path).read_frame(1)
    assert frame.extent == (synthetic_uimf.scans, synthetic_uimf.bins)
    assert len(frame) == synthetic_uimf.points(1)
    for scan in range(synthetic_uimf.scans):
        bins, values = frame.scan(scan)
        try:
            row = synthetic_uimf.scan(1, scan)
        except KeyError:
            assert bins.size == 0, "a scan the writer never stored is an empty slice"
            continue
        assert bins.tolist() == row.bin_index.tolist()
        assert values.tolist() == row.intensity.tolist()


def test_an_unstored_scan_is_an_empty_slice_not_a_missing_key(synthetic_uimf):
    frame = UimfFile(synthetic_uimf.path).read_frame(1)
    missing = min(synthetic_uimf.stored_scans(1)) - 1
    assert frame.scan(missing)[0].size == 0


def test_the_axis_extent_is_the_parameters_not_the_data(synthetic_uimf):
    """Signal stops before the last scan and starts after the first; the arrival-time
    axis still runs the full `Scans` (lab record, task 01)."""
    frame = UimfFile(synthetic_uimf.path).read_frame(1)
    stored = synthetic_uimf.stored_scans(1)
    assert min(stored) > 0 and max(stored) < synthetic_uimf.scans - 1
    assert frame.scans == synthetic_uimf.scans


def test_a_scan_range_reads_only_part_and_keeps_the_axis(synthetic_uimf):
    uimf = UimfFile(synthetic_uimf.path)
    whole = uimf.read_frame(1)
    part = uimf.read_frame(1, scan_range=(5, 9))
    assert part.extent == whole.extent
    assert len(part) == sum(int(np.diff(whole.scan_start)[s]) for s in range(5, 9))
    assert part.scan(4)[0].size == 0


def test_a_bin_range_reads_only_part_and_keeps_the_axis(synthetic_uimf):
    uimf = UimfFile(synthetic_uimf.path)
    whole = uimf.read_frame(1)
    part = uimf.read_frame(1, bin_range=(0, 100))
    assert part.extent == whole.extent
    assert part.bin_index.size and int(part.bin_index.max()) < 100
    assert len(part) == int(np.count_nonzero(whole.bin_index < 100))


@pytest.mark.parametrize("intensity_type", ["ADC", "TDC", "FOLDED"])
def test_every_intensity_element_type_reads_back(tmp_path, intensity_type):
    spec = write_synthetic_uimf(
        tmp_path / f"{intensity_type}.uimf", frames=1, scans=8,
        tof_intensity_type=intensity_type,
    )
    frame = UimfFile(spec.path).read_frame(1)
    assert frame.intensity.dtype == spec.dtype, "the file's own element type, not widened"
    scan = spec.stored_scans(1)[0]
    assert frame.scan(scan)[1].tolist() == spec.scan(1, scan).intensity.tolist()


def test_the_stored_columns_are_reproduced(synthetic_uimf):
    uimf = UimfFile(synthetic_uimf.path)
    scan, non_zero, bpi, tic = uimf.scan_summary(1)
    frame = uimf.read_frame(1)
    assert frame.tic()[scan].tolist() == tic.tolist(), "TIC is exact ground truth"
    assert frame.bpi()[scan].tolist() == bpi.tolist(), "BPI is exact ground truth"
    points = np.diff(frame.scan_start)[scan]
    assert (points <= non_zero).all(), "NonZeroCount is an upper bound"
    assert (points < non_zero).any(), "the fixture writes explicit zeros, so it is not tight"


# --- the same three comparisons, on files a real writer produced ----------------------


def test_a_real_file_reproduces_its_own_tic_and_bpi(real_uimf):
    uimf = UimfFile(real_uimf)
    for number in uimf.frame_numbers()[:4]:
        scan, non_zero, bpi, tic = uimf.scan_summary(number)
        frame = uimf.read_frame(number)
        assert np.array_equal(frame.tic()[scan], tic.astype(frame.tic().dtype))
        assert np.array_equal(frame.bpi()[scan].astype(np.float64), bpi)
        assert (np.diff(frame.scan_start)[scan] <= non_zero).all()


def test_a_real_file_declares_a_usable_calibration(real_uimf):
    uimf = UimfFile(real_uimf)
    bin_width = uimf.global_params().bin_width_ns
    calibration = uimf.frame_params(uimf.frame_numbers()[0]).calibration(bin_width)
    assert calibration.usable
    assert calibration.bin_of(calibration.mz(1000.0)) == pytest.approx(1000.0)
