"""Self-check for a fresh clone: no data file, no lab repo needed.

Every check here must pass in a bare public clone. Checks that need something a
clone does not ship (PNNL's test excerpts in external/, a lab sample named by
MAINSPRING_SMOKE_UIMF, the lab repo) are reported as SKIPPED when it is absent,
never as FAIL.

What it covers today: the package imports, the `uimf` layer stays free of Qt,
the reporting stamp, and the lab-directory resolution. The synthetic-UIMF round
trip and the real-file decode checks arrive with the reader layer (lab record,
tasks 02 and 03).

Run:  uv run tools/check_public.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

FAIL: list[str] = []
SKIPPED: list[str] = []


def check_true(name: str, cond: object) -> None:
    print("{:4s} {}".format("OK" if cond else "FAIL", name))
    if not cond:
        FAIL.append(name)


def skip(name: str, why: str) -> None:
    print(f"SKIP {name} ({why})")
    SKIPPED.append(name)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    # --- imports and the no-Qt seam -------------------------------------------
    import mainspring
    import mainspring.report as report
    import mainspring.uimf  # noqa: F401

    check_true("mainspring imports and has a version", bool(mainspring.__version__))
    qt_loaded = [m for m in sys.modules if m.startswith(("PySide6", "pyqtgraph", "shiboken6"))]
    check_true("importing mainspring.uimf loads no Qt module", not qt_loaded)

    # --- reporting stamp ----------------------------------------------------------
    stamped = report.stamp({"x": 1}, task="check")
    check_true("report.stamp carries task, date, version, commit, results",
               set(stamped) >= {"task", "date", "mainspring_version", "mainspring_commit", "results"})
    check_true("report.stamp wraps the results untouched", stamped["results"] == {"x": 1})

    # --- lab directory resolution ----------------------------------------------------
    lab = mainspring.lab_dir()
    if lab is None:
        skip("lab_dir resolves", "no lab checkout beside this clone; a public clone has none")
    else:
        check_true("lab_dir points at a task system", os.path.isfile(os.path.join(lab, "tasks", "README.md")))
        check_true("lab_dir('data') resolves or is None", mainspring.lab_dir("data") in (None,) or
                   os.path.isdir(mainspring.lab_dir("data")))

    # --- data-dependent checks (not built yet) -------------------------------------
    testdata = os.path.join(mainspring.EXTERNAL_DIR, "pnnl-testdata")
    if not os.path.isdir(testdata) or not os.listdir(testdata):
        skip("PNNL excerpt decode", "run tools/fetch_testdata.py; the decode check itself arrives with the reader layer")
    else:
        skip("PNNL excerpt decode", "excerpts present; the decode check arrives with the reader layer")
    if os.environ.get("MAINSPRING_SMOKE_UIMF"):
        skip("lab sample decode", "the decode check arrives with the reader layer")
    else:
        skip("lab sample decode", "MAINSPRING_SMOKE_UIMF not set")

    print()
    print(f"{len(FAIL)} failed, {len(SKIPPED)} skipped")
    for name in FAIL:
        print("  FAIL", name)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
