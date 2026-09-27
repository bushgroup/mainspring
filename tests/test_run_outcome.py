"""The run's outcome: written `incomplete` at create, replaced at close, read and filtered.

A frame's completion marker says one frame is finished. Nothing in it says whether the
run that wrote the file finished, and a run stopped after twelve of fifty repetitions is
twelve frames that each look complete. So the file carries one more thing, in
`Global_Params`: how the run ended and how much of it was acquired. These tests hold the
three halves of that together: what the writer puts there and when, what the reader
makes of it on a file that has it and on one that does not, and the filter `uimf-info`
offers an analysis.
"""

from __future__ import annotations

import json

import pytest

from mainspring.uimf import (
    OUTCOME_COMPLETED,
    OUTCOME_FAILED,
    OUTCOME_INCOMPLETE,
    OUTCOME_STOPPED,
    OUTCOME_UNKNOWN,
    RUN_OUTCOMES,
    UimfFile,
)
from mainspring.uimf.cli import main as uimf_info
from mainspring.uimf.writer import (
    CUSTOM_PARAM_ID_BASE,
    GLOBAL_KEYS,
    REPETITIONS_ACQUIRED,
    REPETITIONS_PLANNED,
    RUN_OUTCOME,
    RUN_REASON,
    FrameSpec,
    GlobalSpec,
    UimfWriter,
)
from synthetic import write_synthetic_uimf


def a_run(path, *, planned: int = 4) -> UimfWriter:
    return UimfWriter(path, GlobalSpec(bins=4096, instrument_name="SLIM3",
                                       records_outcome=True, repetitions_planned=planned))


def test_the_names_are_mainsprings_own_and_clear_of_pnnls_numbering():
    assert (RUN_OUTCOME, RUN_REASON, REPETITIONS_PLANNED, REPETITIONS_ACQUIRED) == (
        "MainspringRunOutcome", "MainspringRunReason", "MainspringRepetitionsPlanned",
        "MainspringRepetitionsAcquired")
    assert [GLOBAL_KEYS[name].param_id for name in (
        RUN_OUTCOME, RUN_REASON, REPETITIONS_PLANNED, REPETITIONS_ACQUIRED)] == [
        CUSTOM_PARAM_ID_BASE + n for n in (3, 4, 5, 6)]
    assert RUN_OUTCOMES == ("completed", "stopped", "failed", "incomplete")
    assert OUTCOME_UNKNOWN not in RUN_OUTCOMES


def test_a_run_reads_incomplete_from_its_first_commit_until_it_is_closed(tmp_path):
    """A crash or a power cut leaves exactly this, which is the honest reading."""
    path = tmp_path / "run.uimf"
    writer = a_run(path)
    try:
        writer.add_frame(FrameSpec(scans=16, method_frame=1, repetition=1, repetitions=4))
        found = UimfFile(str(path)).global_params()
        assert found.run_outcome == OUTCOME_INCOMPLETE
        assert (found.repetitions_planned, found.repetitions_acquired) == (4, 0)
        assert found.run_reason == ""
    finally:
        writer.close()
    assert UimfFile(str(path)).global_params().run_outcome == OUTCOME_INCOMPLETE


@pytest.mark.parametrize("outcome, reason, acquired", [
    (OUTCOME_COMPLETED, "", 4),
    (OUTCOME_STOPPED, "", 2),
    (OUTCOME_FAILED, "BoxLost: cormorant stopped answering on its port", 1),
])
def test_the_outcome_is_replaced_at_close_with_its_count_and_reason(
        tmp_path, outcome, reason, acquired):
    path = tmp_path / "run.uimf"
    with a_run(path) as writer:
        writer.set_outcome(outcome, reason=reason, repetitions_acquired=acquired)
    found = UimfFile(str(path)).global_params()
    assert (found.run_outcome, found.run_reason) == (outcome, reason)
    assert (found.repetitions_planned, found.repetitions_acquired) == (4, acquired)


def test_a_second_outcome_replaces_the_first_rather_than_adding_a_row(tmp_path):
    path = tmp_path / "run.uimf"
    with a_run(path) as writer:
        writer.set_outcome(OUTCOME_STOPPED, repetitions_acquired=2)
        writer.set_outcome(OUTCOME_FAILED, reason="the fold failed")
    found = UimfFile(str(path)).global_params()
    assert (found.run_outcome, found.repetitions_acquired) == (OUTCOME_FAILED, 2)
    assert list(found.extra).count(RUN_OUTCOME) == 1


def test_what_the_writer_refuses(tmp_path):
    with a_run(tmp_path / "run.uimf") as writer:
        with pytest.raises(ValueError, match="one of completed, stopped"):
            writer.set_outcome("interrupted")
        with pytest.raises(ValueError, match="only for a failed run"):
            writer.set_outcome(OUTCOME_STOPPED, reason="the operator pressed Stop")
        with pytest.raises(ValueError, match="negative"):
            writer.set_outcome(OUTCOME_COMPLETED, repetitions_acquired=-1)
    with UimfWriter(tmp_path / "plain.uimf", GlobalSpec(bins=4096)) as writer:
        with pytest.raises(ValueError, match="without records_outcome"):
            writer.set_outcome(OUTCOME_COMPLETED)
    with pytest.raises(ValueError, match="only with records_outcome"):
        GlobalSpec(bins=4096, repetitions_planned=3)
    with pytest.raises(ValueError, match="not through extra"):
        GlobalSpec(bins=4096, records_outcome=True, extra={RUN_OUTCOME: "completed"})


def test_a_file_that_records_no_outcome_reads_unknown(tmp_path):
    """Every PNNL file, and every file the lab wrote before this was recorded."""
    spec = write_synthetic_uimf(tmp_path / "older.uimf", frames=1, scans=8)
    found = UimfFile(spec.path).global_params()
    assert found.run_outcome == OUTCOME_UNKNOWN
    assert found.repetitions_planned is None and found.repetitions_acquired is None
    with UimfWriter(tmp_path / "plain.uimf", GlobalSpec(bins=4096)) as writer:
        pass
    assert UimfFile(writer.path).global_params().run_outcome == OUTCOME_UNKNOWN


def test_the_list_filters_a_folder_by_how_each_run_ended(tmp_path, capsys):
    folder = tmp_path / "runs"
    folder.mkdir()
    for name, outcome, acquired, reason in (
        ("a_001.uimf", OUTCOME_COMPLETED, 4, ""),
        ("a_002.uimf", OUTCOME_STOPPED, 2, ""),
        ("a_003.uimf", OUTCOME_FAILED, 1, "BoxLost: cormorant"),
    ):
        with a_run(folder / name) as writer:
            writer.set_outcome(outcome, reason=reason, repetitions_acquired=acquired)
    a_run(folder / "a_004.uimf").close()
    write_synthetic_uimf(folder / "older.uimf", frames=1, scans=8)
    (folder / "notes.txt").write_text("not a file to list", encoding="utf-8")

    assert uimf_info([str(folder), "--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("  ")[:2] for line in lines] == [
        ["a_001.uimf", "completed"], ["a_002.uimf", "stopped"], ["a_003.uimf", "failed"],
        ["a_004.uimf", "incomplete"], ["older.uimf", "unknown"]]
    assert "2 of 4 repetitions" in lines[1] and lines[2].endswith("BoxLost: cormorant")

    assert uimf_info([str(folder), "--outcome", "completed,stopped", "--json", "-"]) == 0
    out = capsys.readouterr().out
    stamped = json.loads(out[out.index("{"):])
    assert [row["file"] for row in stamped["results"]["files"]] == ["a_001.uimf", "a_002.uimf"]

    assert uimf_info([str(folder), "--outcome", "interrupted"]) == 2
    assert "no such outcome: interrupted" in capsys.readouterr().err


def test_the_summary_says_how_the_run_ended(tmp_path, capsys):
    path = tmp_path / "run.uimf"
    with a_run(path) as writer:
        writer.set_outcome(OUTCOME_STOPPED, repetitions_acquired=3)
    assert uimf_info([str(path), "--json", "-"]) == 0
    out = capsys.readouterr().out
    summary = json.loads(out[out.index("{\n"):])["results"]["summary"]
    assert (summary["run_outcome"], summary["repetitions_acquired"]) == ("stopped", 3)


def test_the_panels_words_for_each_outcome():
    from mainspring.uimf import GlobalParams

    pytest.importorskip("PySide6")
    from mainspring.viewer.info_panel import run_words

    assert run_words(GlobalParams()) == "not recorded in this file"
    assert run_words(GlobalParams(run_outcome="stopped", repetitions_planned=50,
                                  repetitions_acquired=12)) == "stopped, 12 of 50 repetitions"
    assert run_words(GlobalParams(run_outcome="failed", repetitions_planned=4,
                                  repetitions_acquired=1, run_reason="BoxLost: cormorant")
                     ) == "failed, 1 of 4 repetitions: BoxLost: cormorant"
    assert run_words(GlobalParams(run_outcome="incomplete")).startswith("never closed")
