"""`extra` cannot override what the writer writes from a spec field or keeps on its own.

`GlobalSpec.extra` and `FrameSpec.extra` used to accept a bare PNNL name the writer also
derives from a field. The modern table then held the `extra` value while the legacy row,
and for `TOFIntensityType` the encoded scans, followed the field: a file that said two
things about itself. Such a name is now refused with the field that sets it.
`finalise_frame(extra=)` refuses only what it has an argument for, since it updates both
tables from one set of values.
"""

from __future__ import annotations

import re
import sqlite3

import pytest

from mainspring.uimf import UimfFile
from mainspring.uimf.writer import (
    _FINALISE_OWNERS,
    _FRAME_OWNERS,
    _GLOBAL_OWNERS,
    _OUTCOME_GLOBALS,
    _OUTCOME_REFUSAL,
    _RESIDUAL_COEFFICIENTS,
    _WRITER_OWNED_FRAME_PARAMS,
    _WRITER_OWNED_GLOBALS,
    CLIENT_PARAM_ID_BASE,
    CUSTOM_PARAM_ID_BASE,
    FRAME_COMPLETE,
    FRAME_KEYS,
    FrameSpec,
    GlobalSpec,
    ParamDef,
    UimfWriter,
)


def frame_row(path, frame: int = 1) -> tuple[dict[str, str], sqlite3.Row]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        modern = {row["ParamName"]: row["ParamValue"] for row in conn.execute(
            "SELECT ParamName, ParamValue FROM V_Frame_Params WHERE FrameNum = ?", (frame,))}
        legacy = conn.execute(
            "SELECT * FROM Frame_Parameters WHERE FrameNum = ?", (frame,)).fetchone()
    return modern, legacy


# --- the global side -------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(_GLOBAL_OWNERS))
def test_every_owned_global_is_refused_at_creation_with_its_field(name):
    expected = f"{name} is {_GLOBAL_OWNERS[name]}, not through extra"
    with pytest.raises(ValueError, match=re.escape(expected)):
        GlobalSpec(bins=4096, extra={name: "1"})
    with pytest.raises(ValueError, match=re.escape(expected)):
        GlobalSpec(bins=4096, extra={ParamDef(CLIENT_PARAM_ID_BASE + 99, name,
                                              "System.String"): "1"})


@pytest.mark.parametrize("name", sorted(_OUTCOME_GLOBALS))
def test_an_outcome_global_keeps_its_own_refusal(name):
    with pytest.raises(ValueError) as refused:
        GlobalSpec(bins=4096, extra={name: "1"})
    assert str(refused.value) == _OUTCOME_REFUSAL


def test_the_refusal_names_each_field_and_reads_as_one_sentence():
    with pytest.raises(ValueError) as refused:
        GlobalSpec(bins=4096, extra={"TOFIntensityType": "TDC", "Bins": 1000,
                                     "DriftGas": "N2"})
    assert str(refused.value) == (
        "Bins is set by GlobalSpec.bins; TOFIntensityType is set by"
        " GlobalSpec.tof_intensity_type, not through extra")


def test_set_globals_refuses_with_the_same_table(tmp_path):
    with UimfWriter(tmp_path / "run.uimf", GlobalSpec(bins=4096)) as writer:
        with pytest.raises(ValueError) as refused:
            writer.set_globals({"Bins": 1000})
    assert str(refused.value) == (
        "Bins is set by GlobalSpec.bins, and not rewritten once the file exists")


def test_the_global_table_is_the_owned_set():
    assert set(_GLOBAL_OWNERS) | _OUTCOME_GLOBALS == _WRITER_OWNED_GLOBALS
    assert not set(_GLOBAL_OWNERS) & _OUTCOME_GLOBALS


def test_a_global_the_writer_does_not_write_reaches_both_tables(tmp_path):
    """The legacy row's columns with a modern twin follow `extra`, as the modern row does."""
    path = tmp_path / "run.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096, extra={
        "DriftGas": "N2", "TOFCorrectionTime": 2.5, "PrescanTICThreshold": 7,
        "PrescanContinuous": 1, "PrescanProfile": "profile"})) as writer:
        writer.add_frame(FrameSpec(scans=16))
    with sqlite3.connect(path) as conn:
        modern = dict(conn.execute("SELECT ParamName, ParamValue FROM Global_Params"))
        legacy = conn.execute(
            "SELECT TOFCorrectionTime, Prescan_TICThreshold, Prescan_Continuous,"
            " Prescan_Profile FROM Global_Parameters").fetchone()
    assert modern["DriftGas"] == "N2"
    assert legacy == (2.5, 7, 1, "profile")
    assert (float(modern["TOFCorrectionTime"]), int(modern["PrescanTICThreshold"]),
            int(modern["PrescanContinuous"]), modern["PrescanProfile"]) == legacy


# --- the frame side --------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(_WRITER_OWNED_FRAME_PARAMS))
def test_every_owned_frame_param_is_refused_at_creation_with_its_field(name):
    owners = {**_FRAME_OWNERS, **_FINALISE_OWNERS}
    expected = f"{name} is {owners[name]}, not through extra"
    with pytest.raises(ValueError, match=re.escape(expected)):
        FrameSpec(scans=16, extra={name: 1})
    with pytest.raises(ValueError, match=re.escape(expected)):
        FrameSpec(scans=16, extra={ParamDef(CLIENT_PARAM_ID_BASE + 99, name,
                                            "System.Int32"): 1})


@pytest.mark.parametrize("name", sorted(_FINALISE_OWNERS))
def test_finalise_refuses_what_it_has_an_argument_for(tmp_path, name):
    path = tmp_path / "run.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(FrameSpec(scans=16))
        before = frame_row(path)[0]
        with pytest.raises(ValueError, match=re.escape(f"{name} is {_FINALISE_OWNERS[name]}")):
            writer.finalise_frame(1, complete=False, extra={name: 1})
        assert frame_row(path)[0] == before
    assert UimfFile(str(path)).is_provisional(1)


def test_finalise_may_re_time_a_frame_in_both_tables(tmp_path):
    """clockwork's `begin_again`: a frame acquired again starts when the attempt does."""
    path = tmp_path / "run.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(FrameSpec(scans=16, start_time_minutes=1.0))
        writer.finalise_frame(1, complete=False, extra={"StartTimeMinutes": 2.5})
    modern, legacy = frame_row(path)
    assert float(modern["StartTimeMinutes"]) == legacy["StartTime"] == 2.5
    assert FRAME_COMPLETE not in modern


def test_a_frame_param_the_writer_does_not_write_reaches_both_tables(tmp_path):
    path = tmp_path / "run.uimf"
    residual = dict(zip(_RESIDUAL_COEFFICIENTS, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(FrameSpec(scans=16, extra={
            "TOFLosses": 3, "MPBitOrder": 5, "Decoded": 1, "AmbientTemperature": 21.5,
            **residual}))
    modern, legacy = frame_row(path)
    assert (legacy["TOFLosses"], legacy["MPBitOrder"], legacy["Decoded"]) == (3, 5, 1)
    assert (int(modern["TOFLosses"]), int(modern["MPBitOrder"]),
            int(modern["Decoded"])) == (3, 5, 1)
    assert float(modern["AmbientTemperature"]) == legacy["Temperature"] == 21.5
    for name, value in residual.items():
        assert float(modern[name]) == legacy[name.removeprefix("MassCalibrationCoefficient")]
        assert float(modern[name]) == value


def test_the_residual_coefficients_default_to_zero_in_both_tables(tmp_path):
    path = tmp_path / "run.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(FrameSpec(scans=16))
    modern, legacy = frame_row(path)
    for name in _RESIDUAL_COEFFICIENTS:
        assert float(modern[name]) == 0.0
        assert legacy[name.removeprefix("MassCalibrationCoefficient")] == 0.0
    assert (legacy["TOFLosses"], legacy["MPBitOrder"], legacy["Decoded"]) == (0, 0, 0)


def test_the_owned_frame_set_is_what_the_writer_writes(tmp_path):
    """Held to the code, so a field added to `FrameSpec` cannot slip past the refusal."""
    with UimfWriter(tmp_path / "run.uimf", GlobalSpec(bins=4096)) as writer:
        written = set(writer._frame_values(
            FrameSpec(scans=16, method_frame=1, repetition=1, repetitions=2)))
    # What `add_frame` writes, less the defaults `extra` may replace, is the fields'
    # names and the zero duration `finalise_frame(duration_s=)` later replaces.
    assert written - set(_RESIDUAL_COEFFICIENTS) == set(_FRAME_OWNERS) | {"DurationSeconds"}
    assert set(_RESIDUAL_COEFFICIENTS) <= written
    assert _WRITER_OWNED_FRAME_PARAMS == set(_FRAME_OWNERS) | set(_FINALISE_OWNERS)
    assert _WRITER_OWNED_FRAME_PARAMS <= set(FRAME_KEYS)
    own_block = {name for name, key in FRAME_KEYS.items()
                 if CUSTOM_PARAM_ID_BASE <= key.param_id < CLIENT_PARAM_ID_BASE}
    assert own_block <= _WRITER_OWNED_FRAME_PARAMS


def test_what_finalise_writes_is_its_own_set(tmp_path):
    path = tmp_path / "run.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096)) as writer:
        writer.add_frame(FrameSpec(scans=16))
        before = frame_row(path)[0]
        writer.finalise_frame(1, duration_s=0.5)
    after = frame_row(path)[0]
    changed = {name for name in after if after[name] != before.get(name)}
    assert changed == set(_FINALISE_OWNERS)
