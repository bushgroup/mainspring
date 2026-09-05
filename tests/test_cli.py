"""`uimf-info`: the reader's acceptance test, tested on a file whose answers we know.

`--verify` is the check a user runs in the field to find out whether mainspring
understands their file, so what matters here is that it *fails* when it should. A
verifier that only ever passes is a decoration, so one test corrupts a stored `TIC` in a
copy of the synthetic file and insists on a non-zero exit status.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from mainspring.uimf.cli import main
from synthetic import write_synthetic_uimf


def test_a_missing_file_is_reported_not_traced_back(tmp_path, capsys):
    assert main([str(tmp_path / "nothing.uimf")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_the_summary_names_what_is_in_the_file(synthetic_uimf, capsys):
    assert main([synthetic_uimf.path]) == 0
    out = capsys.readouterr().out
    assert "SLIM3" in out
    assert str(synthetic_uimf.bins) in out
    assert "ADC" in out
    assert "TimeOffset" in out and "not applied" in out


def test_verify_passes_on_a_file_we_wrote(synthetic_uimf, capsys):
    assert main([synthetic_uimf.path, "--verify"]) == 0
    assert "all frames agree" in capsys.readouterr().out


def test_verify_fails_when_a_stored_column_disagrees(tmp_path, capsys):
    """The test that makes the other one mean something."""
    spec = write_synthetic_uimf(tmp_path / "broken.uimf", frames=1, scans=8)
    conn = sqlite3.connect(spec.path)
    try:
        conn.execute("UPDATE Frame_Scans SET TIC = TIC + 1 WHERE ScanNum = ?",
                     (spec.stored_scans(1)[0],))
        conn.commit()
    finally:
        conn.close()
    assert main([spec.path, "--verify"]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_verify_can_be_pointed_at_one_frame(synthetic_uimf, capsys):
    assert main([synthetic_uimf.path, "--verify", "--frame", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("  --verify") == 1


def test_asking_for_a_frame_that_is_not_there(synthetic_uimf, capsys):
    assert main([synthetic_uimf.path, "--frame", "99"]) == 2
    assert "frame 99" in capsys.readouterr().err


def test_bench_times_the_decode_and_the_raster(synthetic_uimf, capsys):
    assert main([synthetic_uimf.path, "--bench"]) == 0
    out = capsys.readouterr().out
    assert "decode_pure" in out and "raster_800x600" in out and "profile_x" in out


def test_json_carries_the_provenance_stamp(synthetic_uimf, tmp_path, capsys):
    target = tmp_path / "results.json"
    assert main([synthetic_uimf.path, "--verify", "--json", str(target)]) == 0
    capsys.readouterr()
    stamped = json.loads(target.read_text(encoding="utf-8"))
    assert set(stamped) >= {"task", "date", "mainspring_version", "results"}
    assert stamped["results"]["summary"]["bins"] == synthetic_uimf.bins
    assert stamped["results"]["verify"]["frames"][0]["tic_mismatches"] == 0


def test_json_to_stdout(synthetic_uimf, capsys):
    assert main([synthetic_uimf.path, "--json"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{"):])["results"]["summary"]["dtype"] == "int32"


def test_a_legacy_only_file_summarises(tmp_path, capsys):
    spec = write_synthetic_uimf(tmp_path / "legacy.uimf", frames=1, scans=8, legacy_only=True)
    assert main([spec.path, "--verify"]) == 0
    assert "legacy only" in capsys.readouterr().out


@pytest.mark.parametrize("intensity_type", ["TDC", "FOLDED"])
def test_the_element_types_no_real_file_uses_verify_too(tmp_path, intensity_type, capsys):
    spec = write_synthetic_uimf(
        tmp_path / f"{intensity_type}.uimf", frames=1, scans=8,
        tof_intensity_type=intensity_type,
    )
    assert main([spec.path, "--verify"]) == 0
    assert "all frames agree" in capsys.readouterr().out


def test_verify_passes_on_every_real_file_this_clone_has(real_uimf, capsys):
    """Milestone M1 in one line: every scan's `TIC` and `BPI` reproduced exactly, and
    `NonZeroCount` never exceeded, on files four different writers produced."""
    assert main([real_uimf, "--verify"]) == 0
    assert "all frames agree" in capsys.readouterr().out
