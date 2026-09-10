"""The writer, the two-phase protocol, and the console's half of it.

mainspring writes a UIMF file in exactly one place, so these are the tests that say
what a file it creates is: a schema PNNL's own tools recognise, parameters under the
IDs UIMF-Library assigns, and a per-frame completion marker that makes "is this frame
finished" a question the file answers rather than one a clock guesses at.

The round trips go through the real reader, because agreement between our writer and
our reader is worth little on its own -- what makes it worth something is that the same
reader reproduces four other writers' files exactly (lab record, task 03).
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

from console_stub import ConsoleStub
from mainspring.uimf import UimfFile, sum_frames
from mainspring.uimf.cli import main as uimf_info
from mainspring.uimf.writer import (
    CLIENT_PARAM_ID_BASE,
    CUSTOM_PARAM_ID_BASE,
    DETECTOR_BITS,
    FRAME_COMPLETE,
    FRAME_KEYS,
    GLOBAL_KEYS,
    METHOD_FRAME,
    WRITER_STAMP,
    FrameSpec,
    GlobalSpec,
    ParamDef,
    UimfWriter,
)
from synthetic import write_synthetic_uimf

SLOPE = 0.738123
INTERCEPT = 0.07690495
TOF_LENGTH_NS = 129003.607843137


def a_frame(**overrides) -> FrameSpec:
    """A frame spec shaped like one repetition of a CLOCK experiment."""
    fields = dict(
        scans=16,
        accumulations=1,
        calibration_slope=SLOPE,
        calibration_intercept=INTERCEPT,
        average_tof_length_ns=TOF_LENGTH_NS,
    )
    fields.update(overrides)
    return FrameSpec(**fields)


def some_scans(count: int = 6, bins: int = 4096):
    """Scans whose base peaks walk across the bin axis, so that a check on the *spread*
    of a writer's `BPI_MZ` convention has something to measure."""
    for scan in range(2, 2 + count):
        centre = 50 + scan * (bins // (count + 4))
        bin_index = np.array([10, centre, centre + 3, bins - 5], dtype=np.int64)
        intensity = np.array([3, 900 + scan, 40, 7], dtype="<i4")
        yield scan, bin_index, intensity


@pytest.fixture
def written(tmp_path):
    """A two-frame file this writer created, both frames finalised."""
    path = tmp_path / "written.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096, instrument_name="SLIM3",
                                     detector_bits=14)) as writer:
        for frame in (1, 2):
            writer.add_frame(a_frame(method_frame=1, repetition=frame, repetitions=2))
            writer.write_scans(frame, some_scans())
            writer.finalise_frame(frame, duration_s=0.645)
    return str(path)


# --- what the file is ----------------------------------------------------------------


def test_it_writes_the_tables_and_indexes_a_real_file_carries(written):
    conn = sqlite3.connect("file:" + written + "?mode=ro", uri=True)
    try:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()
    assert {"Global_Params", "Frame_Param_Keys", "Frame_Params", "V_Frame_Params"} <= names
    assert {"Global_Parameters", "Frame_Parameters"} <= names
    assert {"Frame_Scans", "pk_index_FrameScans", "pk_index_FrameParams"} <= names


def test_the_parameter_ids_are_the_ones_uimf_library_assigns():
    """A downstream tool resolves a parameter by ID. Getting one wrong would not fail
    anywhere; it would quietly make a calibration slope into a temperature."""
    assert FRAME_KEYS["Accumulations"].param_id == 3
    assert FRAME_KEYS["FrameType"].param_id == 4
    assert FRAME_KEYS["Scans"].param_id == 7
    assert FRAME_KEYS["CalibrationSlope"].param_id == 12
    assert GLOBAL_KEYS["Bins"].param_id == 6
    assert GLOBAL_KEYS["TOFIntensityType"].param_id == 8


def test_our_own_parameters_are_clear_of_pnnls_numbering():
    """UIMF-Library skips an ID it does not know, so a custom key is free -- but an ID
    PNNL later assigns to something else would be read as that something else."""
    for keys, ceiling in ((FRAME_KEYS, 52), (GLOBAL_KEYS, 22)):
        ours = [d for d in keys.values() if d.name.startswith("Mainspring")]
        assert ours, "the custom keys are the point"
        assert all(d.param_id >= CUSTOM_PARAM_ID_BASE for d in ours)
        assert max(d.param_id for d in keys.values() if not d.name.startswith("Mainspring")) \
            == ceiling
        assert len({d.param_id for d in keys.values()}) == len(keys)


def test_the_dictionary_holds_the_keys_the_file_uses_and_not_the_rest(written):
    """`Frame_Param_Keys` is the file's own dictionary; our sample has 14 rows in it,
    not one per parameter that has ever existed."""
    conn = sqlite3.connect("file:" + written + "?mode=ro", uri=True)
    try:
        declared = {row[0] for row in conn.execute("SELECT ParamName FROM Frame_Param_Keys")}
        used = {row[0] for row in conn.execute(
            "SELECT DISTINCT ParamName FROM V_Frame_Params")}
    finally:
        conn.close()
    assert used <= declared
    assert declared < set(FRAME_KEYS), "not every key PNNL has ever defined"
    assert "VoltJetDist" not in declared


def test_a_file_is_written_in_wal_so_a_reader_can_follow_it(tmp_path):
    path = tmp_path / "wal.uimf"
    UimfWriter(path, GlobalSpec(bins=64)).close()
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()
    assert not os.path.exists(str(path) + "-wal"), "closing checkpoints and tidies up"


def test_the_journal_mode_can_be_asked_for_explicitly(tmp_path):
    """Every file PNNL's writers produce is in `delete`, and the fixture writes one so
    that the reader is exercised against both."""
    path = tmp_path / "delete.uimf"
    UimfWriter(path, GlobalSpec(bins=64), journal_mode="delete").close()
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        conn.close()


# --- round trips through the reader ---------------------------------------------------


def test_what_is_written_is_what_is_read_back(written):
    uimf = UimfFile(written)
    globals_ = uimf.global_params()
    assert globals_.bins == 4096
    assert globals_.instrument_name == "SLIM3"
    assert globals_.num_frames == 2
    assert uimf.frame_numbers() == [1, 2]

    params = uimf.frame_params(1)
    assert params.scans == 16
    assert params.accumulations == 1
    assert params.frame_type == 1, "MS (Regular), the value a downstream MS1 filter looks for"
    assert params.calibration_slope == pytest.approx(SLOPE)
    assert params.calibration_done

    frame = uimf.read_frame(1)
    written_scans = list(some_scans())
    assert len(frame) == sum(b.size for _, b, _ in written_scans)
    for scan, bin_index, intensity in written_scans:
        got_bins, got_intensity = frame.scan(scan)
        assert got_bins.tolist() == bin_index.tolist()
        assert got_intensity.tolist() == intensity.tolist()


def test_the_summary_columns_are_the_ones_verify_checks(written, capsys):
    """`uimf-info --verify` is the acceptance test the reader is written in terms of, so
    it is the one worth pointing at a file we produced ourselves."""
    assert uimf_info([written, "--verify"]) == 0
    assert "all frames agree" in capsys.readouterr().out


def test_the_two_tables_agree_when_nobody_asks_them_not_to(written):
    """Real files disagree about frame type; this writer emits one value in both, and
    the fixture is the only caller that overrides it."""
    conn = sqlite3.connect("file:" + written + "?mode=ro", uri=True)
    try:
        modern = conn.execute(
            "SELECT ParamValue FROM V_Frame_Params"
            " WHERE FrameNum = 1 AND ParamName = 'FrameType'").fetchone()[0]
        legacy = conn.execute(
            "SELECT FrameType FROM Frame_Parameters WHERE FrameNum = 1").fetchone()[0]
    finally:
        conn.close()
    assert int(modern) == legacy == 1


def test_a_frame_read_out_can_be_written_back_unchanged(written, tmp_path):
    """What the fold does: read repetitions, sum them, write the sum as one frame."""
    uimf = UimfFile(written)
    total = sum_frames([uimf.read_frame(n) for n in uimf.frame_numbers()])

    path = tmp_path / "summed.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096, instrument_name="SLIM3")) as writer:
        writer.add_frame(a_frame(accumulations=2))
        writer.write_sparse_frame(1, total)
        writer.finalise_frame(1, duration_s=1.29)

    back = UimfFile(path).read_frame(1)
    assert back.bin_index.tolist() == total.bin_index.tolist()
    assert back.intensity.tolist() == total.intensity.tolist()
    assert back.scan_start.tolist() == total.scan_start.tolist()
    assert UimfFile(path).frame_params(1).accumulations == 2


def test_empty_scans_are_left_out_unless_they_are_asked_for(written, tmp_path):
    frame = UimfFile(written).read_frame(1)
    dense = tmp_path / "dense.uimf"
    with UimfWriter(dense, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame())
        assert writer.write_sparse_frame(1, frame, store_empty_scans=True) == frame.scans
    sparse = tmp_path / "sparse.uimf"
    with UimfWriter(sparse, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame())
        assert writer.write_sparse_frame(1, frame) == len(list(some_scans()))


# --- the two phases --------------------------------------------------------------------


def test_a_frame_is_provisional_until_it_is_finalised(tmp_path):
    path = tmp_path / "twophase.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame())
        writer.write_scans(1, some_scans())
        uimf = UimfFile(path)
        assert uimf.has_completion_markers
        assert uimf.is_provisional(1)
        assert uimf.read_frame(1).provisional
        writer.finalise_frame(1, duration_s=0.645)
        # The same `UimfFile`: a frame that was provisional must not stay provisional in
        # a process that has already cached its parameters.
        assert not uimf.is_provisional(1)
        assert not uimf.read_frame(1).provisional


def test_a_frame_the_client_never_finalised_stays_provisional(tmp_path):
    """A power cut mid-acquisition. The file opens, every finished frame reads as
    finished, and the last one is honest about not being."""
    spec = write_synthetic_uimf(tmp_path / "cut.uimf", frames=3, scans=8, finalise=False)
    uimf = UimfFile(spec.path)
    assert uimf.frame_numbers() == [1, 2, 3]
    assert not uimf.is_provisional(1)
    assert not uimf.is_provisional(2)
    assert uimf.is_provisional(3)
    assert len(uimf.read_frame(3)) > 0, "and its data is still readable"


def test_finalising_without_the_marker_records_a_frame_that_was_cut_short(tmp_path):
    path = tmp_path / "short.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame())
        writer.finalise_frame(1, duration_s=0.2, complete=False)
    params = UimfFile(path).frame_params(1)
    assert not params.marked_complete
    assert float(params.extra["DurationSeconds"]) == pytest.approx(0.2)


def test_the_duration_row_exists_before_the_console_would_update_it(tmp_path):
    """The console's `update_timing_information` is an `UPDATE ... WHERE ParamID = ?`
    inside a `catch (...) {}`, so a row that is not there yet silently loses the timing
    the card's own clock produced."""
    path = tmp_path / "duration.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame())
        conn = sqlite3.connect("file:" + str(path) + "?mode=rw", uri=True)
        try:
            changed = conn.execute(
                "UPDATE Frame_Params SET ParamValue = ? WHERE FrameNum = ? AND ParamID = ?",
                ("0.6451", 1, FRAME_KEYS["DurationSeconds"].param_id),
            ).rowcount
            conn.commit()
        finally:
            conn.close()
    assert changed == 1
    assert UimfFile(path).frame_params(1).extra["DurationSeconds"] == "0.6451"


def test_a_file_with_no_marker_in_it_falls_back_to_the_heuristic(tmp_path):
    """Every file PNNL's writers produce is like this, and there the age of the file is
    still the only thing there is to go on (lab record, task 08)."""
    spec = write_synthetic_uimf(tmp_path / "pnnl.uimf", frames=2, scans=8)
    conn = sqlite3.connect(spec.path)
    try:
        conn.execute("DELETE FROM Global_Params WHERE ParamName = ?", (WRITER_STAMP,))
        conn.execute("DELETE FROM Frame_Params WHERE ParamID = ?",
                     (FRAME_KEYS[FRAME_COMPLETE].param_id,))
        conn.commit()
    finally:
        conn.close()
    uimf = UimfFile(spec.path)
    assert not uimf.has_completion_markers
    assert not uimf.is_provisional(1), "not the last frame"
    assert uimf.is_provisional(2), "the last frame of a file just written to"
    os.utime(spec.path, (0, 0))
    assert not UimfFile(spec.path).is_provisional(2), "and not once it has gone cold"


# --- the grouping and the bit depth ------------------------------------------------------


def test_the_grouping_is_readable_from_the_parameters_alone(written):
    """Task 17's navigation and the fold both read this without seeing the method."""
    uimf = UimfFile(written)
    assert [uimf.frame_params(n).method_frame for n in (1, 2)] == [1, 1]
    assert [uimf.frame_params(n).repetition for n in (1, 2)] == [1, 2]
    assert uimf.frame_params(2).repetitions == 2


def test_a_frame_can_cover_a_whole_method_frame_rather_than_one_repetition(tmp_path):
    """What a summed companion says about itself, and what an acquisition writes when
    one console frame holds every repetition instead of one each: a method frame with
    no repetition number, because it is not one of them."""
    path = tmp_path / "summed.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame(accumulations=8, method_frame=3, repetitions=8))
        writer.write_scans(1, some_scans())
        writer.finalise_frame(1, duration_s=5.16)
    params = UimfFile(path).frame_params(1)
    assert params.method_frame == 3
    assert params.repetition is None
    assert params.repetitions == 8
    assert params.accumulations == 8


def test_an_ungrouped_file_says_so_rather_than_guessing(tmp_path):
    spec = write_synthetic_uimf(tmp_path / "flat.uimf", frames=1, scans=8)
    params = UimfFile(spec.path).frame_params(1)
    assert params.method_frame is None and params.repetition is None


def test_the_detector_bit_depth_round_trips_and_is_absent_when_not_stored(written, tmp_path):
    assert UimfFile(written).global_params().detector_bits == 14
    plain = tmp_path / "plain.uimf"
    UimfWriter(plain, GlobalSpec(bins=64)).close()
    assert UimfFile(plain).global_params().detector_bits is None


def test_the_writer_stamps_the_file_with_what_wrote_it(written):
    import mainspring

    assert UimfFile(written).global_params().written_by == f"mainspring {mainspring.__version__}"


def test_num_frames_keeps_up_with_the_frames(tmp_path):
    path = tmp_path / "counting.uimf"
    with UimfWriter(path, GlobalSpec(bins=64)) as writer:
        for expected in (1, 2, 3):
            writer.add_frame(a_frame())
            assert UimfFile(path).global_params().num_frames == expected


# --- what it refuses ----------------------------------------------------------------------


def test_it_will_not_quietly_replace_an_acquisition(tmp_path):
    path = tmp_path / "once.uimf"
    UimfWriter(path, GlobalSpec(bins=64)).close()
    with pytest.raises(FileExistsError):
        UimfWriter(path, GlobalSpec(bins=64))
    UimfWriter(path, GlobalSpec(bins=64), overwrite=True).close()


def test_an_invented_parameter_name_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a UIMF frame parameter"):
        FrameSpec(scans=8, extra={"DriftVoltage": 1.0})
    with pytest.raises(ValueError, match="not a UIMF global parameter"):
        GlobalSpec(bins=64, extra={"DetectorBits": 14})


def test_a_parameter_pnnl_does_name_goes_through(tmp_path):
    path = tmp_path / "extra.uimf"
    with UimfWriter(path, GlobalSpec(bins=64, extra={"DriftGas": "N2"})) as writer:
        writer.add_frame(a_frame(extra={"AmbientTemperature": 21.5}))
        writer.finalise_frame(1, extra={"TOFLosses": 3})
    uimf = UimfFile(path)
    assert uimf.global_params().extra["DriftGas"] == "N2"
    assert float(uimf.frame_params(1).extra["AmbientTemperature"]) == pytest.approx(21.5)
    assert uimf.frame_params(1).extra["TOFLosses"] == "3"


@pytest.mark.parametrize("kwargs", [
    {"scans": 0},
    {"accumulations": 0},
    {"repetition": 2},                       # no method frame to belong to
    {"repetitions": 3},                      # likewise
    {"method_frame": 1, "repetition": 0},
    {"method_frame": 1, "repetition": 4, "repetitions": 3},
])
def test_a_frame_spec_that_cannot_mean_anything_is_refused(kwargs):
    with pytest.raises(ValueError):
        a_frame(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"bins": 0}, {"bin_width_ns": 0.0}, {"tof_intensity_type": "SOMETHING_NEW"},
    {"detector_bits": 0}, {"detector_bits": 65},
])
def test_a_global_spec_that_cannot_mean_anything_is_refused(kwargs):
    with pytest.raises(ValueError):
        GlobalSpec(**{"bins": 64, **kwargs})


# --- a client's own parameters ------------------------------------------------------------

METHOD_NAME = ParamDef(CLIENT_PARAM_ID_BASE + 1, "ClockworkMethodName", "System.String",
                       "Name of the method that produced this file")
METHOD_SHA = ParamDef(CLIENT_PARAM_ID_BASE + 2, "ClockworkMethodSha256", "System.String",
                      "sha256 of the canonical method TOML")
FRAME_LABEL = ParamDef(CLIENT_PARAM_ID_BASE + 1, "ClockworkStepLabel", "System.String",
                       "Which step of the method this frame ran")


def test_a_client_can_record_something_the_format_has_no_name_for(tmp_path):
    """The provenance a client wants travelling inside the file rather than beside it.
    mainspring's own registry stays closed; the client gets a block above it."""
    path = tmp_path / "stamped.uimf"
    with UimfWriter(path, GlobalSpec(
        bins=4096, extra={METHOD_NAME: "bradykinin-clock", METHOD_SHA: "a" * 64},
    )) as writer:
        writer.add_frame(a_frame(extra={FRAME_LABEL: "step-2"}))
        writer.finalise_frame(1, duration_s=0.645)

    uimf = UimfFile(path)
    assert uimf.global_params().extra["ClockworkMethodName"] == "bradykinin-clock"
    assert uimf.global_params().extra["ClockworkMethodSha256"] == "a" * 64
    assert uimf.frame_params(1).extra["ClockworkStepLabel"] == "step-2"

    conn = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
    try:
        declared = dict(conn.execute(
            "SELECT ParamName, ParamID FROM Frame_Param_Keys"))
        typed = dict(conn.execute(
            "SELECT ParamName, ParamDataType FROM Global_Params"))
    finally:
        conn.close()
    assert declared["ClockworkStepLabel"] == CLIENT_PARAM_ID_BASE + 1
    assert typed["ClockworkMethodName"] == "System.String"


def test_a_client_parameter_that_would_collide_is_refused():
    below = ParamDef(52, "ClockworkThing", "System.String")
    with pytest.raises(ValueError, match="start at 2000"):
        GlobalSpec(bins=64, extra={below: "x"})
    mainspring_block = ParamDef(CUSTOM_PARAM_ID_BASE + 1, "ClockworkThing", "System.String")
    with pytest.raises(ValueError, match="start at 2000"):
        FrameSpec(scans=8, extra={mainspring_block: "x"})
    renaming = ParamDef(CLIENT_PARAM_ID_BASE + 3, "Scans", "System.Int32")
    with pytest.raises(ValueError, match="already a frame parameter"):
        FrameSpec(scans=8, extra={renaming: 4})
    untyped = ParamDef(CLIENT_PARAM_ID_BASE + 4, "ClockworkThing", "System.Guid")
    with pytest.raises(ValueError, match="ParamDataType"):
        FrameSpec(scans=8, extra={untyped: "x"})
    with pytest.raises(ValueError, match="share ID"):
        FrameSpec(scans=8, extra={
            ParamDef(CLIENT_PARAM_ID_BASE + 5, "One", "System.String"): "a",
            ParamDef(CLIENT_PARAM_ID_BASE + 5, "Two", "System.String"): "b",
        })


def test_one_file_cannot_end_up_with_two_meanings_for_one_id(tmp_path):
    """Two frames declaring the same ID under different names would leave a dictionary
    the file's own `V_Frame_Params` view cannot resolve."""
    with UimfWriter(tmp_path / "clash.uimf", GlobalSpec(bins=64)) as writer:
        writer.add_frame(a_frame(extra={FRAME_LABEL: "step-1"}))
        other = ParamDef(FRAME_LABEL.param_id, "ClockworkSomethingElse", "System.String")
        with pytest.raises(ValueError, match="already"):
            writer.add_frame(a_frame(extra={other: "x"}))


@pytest.mark.parametrize("scan", [
    (0, np.array([1, 2]), np.array([1])),          # as many intensities as bins
    (0, np.array([-1]), np.array([1])),            # inside the axis
    (0, np.array([99999]), np.array([1])),
    (0, np.array([5, 5]), np.array([1, 2])),       # ascending, as the stream requires
    (0, np.array([9, 3]), np.array([1, 2])),
])
def test_a_scan_that_would_not_decode_is_refused(tmp_path, scan):
    with UimfWriter(tmp_path / "bad.uimf", GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame())
        with pytest.raises(ValueError):
            writer.write_scans(1, [scan])


def test_a_closed_writer_says_so(tmp_path):
    writer = UimfWriter(tmp_path / "closed.uimf", GlobalSpec(bins=64))
    writer.close()
    writer.close()  # idempotent
    with pytest.raises(ValueError, match="closed"):
        writer.add_frame(a_frame())


def test_the_same_frame_cannot_be_added_twice(tmp_path):
    with UimfWriter(tmp_path / "twice.uimf", GlobalSpec(bins=64)) as writer:
        writer.add_frame(a_frame(), frame=4)
        with pytest.raises(ValueError):
            writer.add_frame(a_frame(), frame=4)


def test_finalising_a_frame_that_was_never_added(tmp_path):
    with UimfWriter(tmp_path / "ghost.uimf", GlobalSpec(bins=64)) as writer:
        with pytest.raises(KeyError):
            writer.finalise_frame(7, duration_s=1.0)


# --- the console's half --------------------------------------------------------------------


def test_the_console_appends_to_a_file_it_did_not_create(tmp_path):
    path = tmp_path / "acquired.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096, instrument_name="SLIM3")) as writer:
        writer.add_frame(a_frame(method_frame=1, repetition=1, repetitions=1))
        assert ConsoleStub(path).acquire_frame(1, some_scans()) == len(list(some_scans()))
        writer.finalise_frame(1, duration_s=0.645)

    uimf = UimfFile(path)
    assert not uimf.is_provisional(1)
    frame = uimf.read_frame(1)
    for scan, bin_index, intensity in some_scans():
        got_bins, got_intensity = frame.scan(scan)
        assert got_bins.tolist() == bin_index.tolist()
        assert got_intensity.tolist() == intensity.tolist()


def test_the_console_refuses_to_create_the_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ConsoleStub(tmp_path / "nothing.uimf")


def test_the_console_numbers_scans_from_its_start_trigger(tmp_path):
    path = tmp_path / "triggered.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame(scans=32))
        ConsoleStub(path).acquire_frame(
            1, ((scan + 20, b, i) for scan, b, i in some_scans()), start_trigger=20
        )
        writer.finalise_frame(1)
    stored, *_ = UimfFile(path).scan_summary(1)
    assert stored.tolist() == [scan for scan, _, _ in some_scans()]


def test_the_console_writes_in_batches_and_the_frame_grows_between_them(tmp_path):
    """What task 08's poll will watch: a frame that is longer than it was a moment ago,
    in a file the writer has not finalised."""
    path = tmp_path / "batched.uimf"
    scans = list(some_scans(count=12))
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame(scans=32))
        console = ConsoleStub(path, batch_size=4)
        uimf = UimfFile(path)
        console.acquire_frame(1, scans[:4])
        first = len(uimf.read_frame(1))
        console.acquire_frame(1, scans[4:])
        assert len(uimf.read_frame(1)) > first
        assert uimf.is_provisional(1), "and none of it may be cached"


def test_the_console_drops_a_scan_with_nothing_in_it(tmp_path):
    """Its own rule is `encoded_spectra.size() > 1 || er.scan == 0`, which is why a
    5000-scan frame stores 1656 rows."""
    path = tmp_path / "quiet.uimf"
    empty = np.empty(0, dtype=np.int64)
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame(scans=32))
        written = ConsoleStub(path).acquire_frame(1, [
            (0, empty, empty.astype("<i4")),
            (5, empty, empty.astype("<i4")),
            (6, np.array([100, 200]), np.array([9, 8], dtype="<i4")),
        ])
    assert written == 2, "scan 0 always, scan 5 never, scan 6 because it has signal"
    stored, *_ = UimfFile(path).scan_summary(1)
    assert stored.tolist() == [0, 6]


def test_the_console_puts_a_bin_index_in_the_column_that_holds_an_mz(tmp_path, capsys):
    """Read off `uimfacquisitionrecord.cpp`, where `bpi_mz = index_max_intensity`, and
    not yet seen on a real file. `--verify` reads that column as an m/z and inverts it
    through the calibration, so on a console-written file the check that catches a
    decode error catches this instead -- while `TIC` and `BPI`, which are what the
    decode is actually judged on, agree exactly."""
    path = tmp_path / "console-mz.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame(scans=32))
        ConsoleStub(path).acquire_frame(1, some_scans())
        writer.finalise_frame(1)
    assert uimf_info([str(path), "--verify"]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out

    zeroed = tmp_path / "console-zero.uimf"
    with UimfWriter(zeroed, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(a_frame(scans=32))
        ConsoleStub(zeroed, bpi_mz_as_bin=False).acquire_frame(1, some_scans())
        writer.finalise_frame(1)
    assert uimf_info([str(zeroed), "--verify"]) == 0
    assert "all frames agree" in capsys.readouterr().out


def test_the_console_and_the_writer_do_not_fight_over_the_file(tmp_path):
    """Both connections are open at once, which is the arrangement an acquisition runs
    in: the client holding the file for the length of the run while the console inserts
    into it."""
    path = tmp_path / "shared.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        console = ConsoleStub(path)
        for frame in (1, 2, 3):
            writer.add_frame(a_frame(method_frame=1, repetition=frame, repetitions=3))
            console.acquire_frame(frame, some_scans())
            writer.finalise_frame(frame, duration_s=0.645)
    uimf = UimfFile(path)
    assert uimf.frame_numbers() == [1, 2, 3]
    assert not any(uimf.is_provisional(n) for n in (1, 2, 3))
    assert uimf_info([str(path)]) == 0
