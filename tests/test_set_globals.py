"""`UimfWriter.set_globals`: a client's own global parameters, written after creation.

`GlobalSpec.extra` takes a client's globals at creation only, and `set_outcome` is for
mainspring's own outcome keys. A fact learned mid-run -- how many times the run was held
between repetitions, and for how long -- needs the global twin of
`finalise_frame(extra=)`: the same keys `GlobalSpec.extra` takes, less the ones the writer
writes itself, upserted in one transaction, more than once if need be.
"""

from __future__ import annotations

import sqlite3

import pytest

from mainspring.uimf import UimfFile
from mainspring.uimf.writer import (
    _WRITER_OWNED_GLOBALS,
    CLIENT_PARAM_ID_BASE,
    CUSTOM_PARAM_ID_BASE,
    GLOBAL_KEYS,
    RUN_OUTCOME,
    FrameSpec,
    GlobalSpec,
    ParamDef,
    UimfWriter,
)

HOLDS = ParamDef(2018, "ClockworkHolds", "System.Int32", "Times the run was held")
HELD_SECONDS = ParamDef(2019, "ClockworkHeldSeconds", "System.Double", "Seconds held")
HOLD_TEXT = ParamDef(2020, "ClockworkHoldText", "System.String", "Why it was held")


def a_run(path, **kwargs) -> UimfWriter:
    writer = UimfWriter(path, GlobalSpec(bins=4096, instrument_name="SLIM3", **kwargs))
    writer.add_frame(FrameSpec(scans=16))
    return writer


def stored_globals(path) -> dict[str, str]:
    with sqlite3.connect(path) as conn:
        return dict(conn.execute("SELECT ParamName, ParamValue FROM Global_Params"))


def test_a_client_global_added_mid_run_reads_back_by_name(tmp_path):
    path = tmp_path / "run.uimf"
    with a_run(path) as writer:
        writer.set_globals({HOLDS: 1, HELD_SECONDS: 12.5, HOLD_TEXT: "operator hold"})
        extra = UimfFile(str(path)).global_params().extra
        assert extra["ClockworkHolds"] == "1"
        assert float(extra["ClockworkHeldSeconds"]) == pytest.approx(12.5)
        assert extra["ClockworkHoldText"] == "operator hold"
    with sqlite3.connect(path) as conn:
        rows = dict(conn.execute(
            "SELECT ParamName, ParamID FROM Global_Params WHERE ParamID >= ?",
            (CLIENT_PARAM_ID_BASE,)))
    assert rows == {"ClockworkHolds": 2018, "ClockworkHeldSeconds": 2019,
                    "ClockworkHoldText": 2020}


def test_a_second_call_overwrites_and_leaves_the_rest(tmp_path):
    path = tmp_path / "run.uimf"
    with a_run(path) as writer:
        writer.set_globals({HOLDS: 1, HELD_SECONDS: 12.5})
        writer.set_globals({HOLDS: 2, HELD_SECONDS: 30.0})
        writer.set_globals({HOLD_TEXT: "second hold"})
    stored = stored_globals(path)
    assert stored["ClockworkHolds"] == "2"
    assert float(stored["ClockworkHeldSeconds"]) == 30.0
    assert stored["ClockworkHoldText"] == "second hold"


def test_a_definition_declared_at_creation_can_be_set_again(tmp_path):
    path = tmp_path / "run.uimf"
    with a_run(path, extra={HOLDS: 0}) as writer:
        assert stored_globals(path)["ClockworkHolds"] == "0"
        writer.set_globals({HOLDS: 3})
    assert stored_globals(path)["ClockworkHolds"] == "3"


def test_a_bare_pnnl_name_is_accepted(tmp_path):
    path = tmp_path / "run.uimf"
    with a_run(path) as writer:
        writer.set_globals({"DriftGas": "N2"})
    assert UimfFile(str(path)).global_params().extra["DriftGas"] == "N2"


def test_a_file_already_open_keeps_its_globals_and_a_fresh_one_sees_the_new(tmp_path):
    """What `UimfFile.global_params` says about its cache, held to the code."""
    path = tmp_path / "run.uimf"
    with a_run(path) as writer:
        early = UimfFile(str(path))
        assert "ClockworkHolds" not in early.global_params().extra
        writer.set_globals({HOLDS: 1})
        early.refresh()
        assert "ClockworkHolds" not in early.global_params().extra
        assert UimfFile(str(path)).global_params().extra["ClockworkHolds"] == "1"


@pytest.mark.parametrize("key, match", [
    (ParamDef(52, "ClockworkThing", "System.String"), "start at 2000"),
    (ParamDef(CUSTOM_PARAM_ID_BASE + 9, "ClockworkThing", "System.String"), "start at 2000"),
    (ParamDef(2018, "ClockworkOther", "System.Int32"), "already 'ClockworkHolds'"),
    (ParamDef(2099, "ClockworkHolds", "System.Int32"), "already a global parameter"),
    (ParamDef(2099, "ClockworkThing", "System.Guid"), "ParamDataType"),
    ("NotAPnnlGlobal", "not a UIMF global parameter"),
])
def test_a_key_globalspec_extra_would_refuse_is_refused(tmp_path, key, match):
    with a_run(tmp_path / "run.uimf", extra={HOLDS: 0}) as writer:
        with pytest.raises(ValueError, match=match):
            writer.set_globals({key: "1"})


@pytest.mark.parametrize("name", sorted(_WRITER_OWNED_GLOBALS))
def test_every_global_the_writer_owns_is_refused(tmp_path, name):
    path = tmp_path / "run.uimf"
    with a_run(path) as writer:
        before = stored_globals(path)
        with pytest.raises(ValueError):
            writer.set_globals({name: "1"})
        with pytest.raises(ValueError):
            writer.set_globals({ParamDef(2099, name, "System.String"): "1"})
    assert stored_globals(path) == before


def test_an_outcome_key_is_refused_as_globalspec_refuses_it(tmp_path):
    with pytest.raises(ValueError) as at_creation:
        GlobalSpec(bins=4096, extra={RUN_OUTCOME: "completed"})
    with a_run(tmp_path / "run.uimf", records_outcome=True) as writer:
        with pytest.raises(ValueError) as later:
            writer.set_globals({RUN_OUTCOME: "completed"})
    assert str(later.value) == str(at_creation.value)


def test_the_owned_set_is_what_the_writer_writes(tmp_path):
    """Held to the code, so a field added to `GlobalSpec` cannot slip past the refusal."""
    with a_run(tmp_path / "run.uimf", detector_bits=14, records_outcome=True,
               repetitions_planned=4) as writer:
        written = set(writer._global_values())
    assert written <= _WRITER_OWNED_GLOBALS
    own_block = {name for name, key in GLOBAL_KEYS.items()
                 if CUSTOM_PARAM_ID_BASE <= key.param_id < CLIENT_PARAM_ID_BASE}
    assert own_block <= _WRITER_OWNED_GLOBALS
    assert _WRITER_OWNED_GLOBALS <= set(GLOBAL_KEYS)


@pytest.mark.parametrize("bad", [
    {"NotAPnnlGlobal": 1},
    {ParamDef(2021, "ClockworkOther", "System.String"): "x",
     ParamDef(2021, "ClockworkAnother", "System.String"): "y"},
    {ParamDef(2021, "ClockworkCount", "System.Int32"): "many"},
])
def test_one_bad_key_writes_and_registers_none_of_the_call(tmp_path, bad):
    path = tmp_path / "run.uimf"
    with a_run(path) as writer:
        before = stored_globals(path)
        registry = dict(writer.global_keys)
        with pytest.raises(ValueError):
            writer.set_globals({HOLDS: 1, HELD_SECONDS: 2.0, **bad})
        assert writer.global_keys == registry
        assert stored_globals(path) == before
        # and nothing it half-registered stands in the way of the corrected call
        writer.set_globals({HOLDS: 1, HELD_SECONDS: 2.0})
    assert stored_globals(path)["ClockworkHolds"] == "1"


def test_a_legacy_only_file_checks_the_call_and_writes_nothing(tmp_path):
    path = tmp_path / "legacy.uimf"
    with UimfWriter(path, GlobalSpec(bins=4096), tables="legacy") as writer:
        writer.set_globals({HOLDS: 1, "DriftGas": "N2"})
        with pytest.raises(ValueError):
            writer.set_globals({"NotAPnnlGlobal": 1})
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "Global_Params" not in tables
    assert "ClockworkHolds" not in UimfFile(str(path)).global_params().extra


def test_an_empty_call_is_nothing_and_a_closed_writer_refuses(tmp_path):
    path = tmp_path / "run.uimf"
    writer = a_run(path)
    before = stored_globals(path)
    writer.set_globals({})
    assert stored_globals(path) == before
    writer.close()
    with pytest.raises(ValueError, match="closed"):
        writer.set_globals({HOLDS: 1})
