"""Reporting conventions: stamped `results.json`, commented CSV tables, figures.

The small set of habits that make a number found in an exploration directory six
months later still traceable, gathered here so every lab-side script writes them the
same way. Mirrors the convention of the lab's other packages (schamp.report).

* **`results.json` is stamped** with the date, the mainspring version, and the commit
  of the code repo and -- when a lab checkout resolves -- of the lab repo, wrapped
  around the results under a `results` key.
* **A written table carries a `#` comment header** saying what it is, which task wrote
  it and where its numbers came from.
* **Figures are headless and deterministic**: matplotlib is imported inside the
  functions that need it and switched to Agg before pyplot exists. matplotlib is not a
  dependency of the package; a lab-side script that plots installs it there.

No numpy at import, no I/O at import, no Qt anywhere.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from . import ROOT, __version__, lab_dir

__all__ = [
    "commit_of",
    "figure_defaults",
    "stamp",
    "use_headless_matplotlib",
    "write_results",
    "write_table",
]


def commit_of(repo: str | os.PathLike[str]) -> str | None:
    """The short HEAD commit of a git repository, or None if it is not one.

    Never raises: a tarball rather than a checkout, or a machine with no `git`, gets
    None and a results file that says so honestly.
    """
    try:
        done = subprocess.run(
            ["git", "-C", os.fspath(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


def stamp(results: Any, *, task: str = "") -> dict[str, Any]:
    """Wrap results in their provenance: `{task, date, ..., results}`."""
    lab = lab_dir()
    return {
        "task": task,
        "date": _dt.date.today().isoformat(),
        "mainspring_version": __version__,
        "mainspring_commit": commit_of(ROOT),
        "lab_commit": commit_of(lab) if lab else None,
        "results": results,
    }


def write_results(
    results: Any,
    out_dir: str | os.PathLike[str],
    *,
    task: str = "",
    name: str = "results.json",
) -> str:
    """Write a stamped `results.json` into `out_dir` and return its path.

    `results` must be JSON-serialisable: convert numpy scalars and arrays first.
    """
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(stamp(results, task=task), handle, indent=2)
        handle.write("\n")
    return path


def write_table(
    path: str | os.PathLike[str],
    header: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    comment: str = "",
) -> str:
    """Write a CSV with an optional `#` comment block above the header (LF, no BOM)."""
    import csv  # noqa: PLC0415 -- stdlib, but nothing at import time needs it

    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for line in comment.splitlines():
            handle.write(f"# {line}\n" if line.strip() else "#\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def figure_defaults() -> Mapping[str, object]:
    """The rcParams every mainspring figure starts from: plain, vector text kept as text."""
    return {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def use_headless_matplotlib() -> None:
    """Select the Agg backend and apply `figure_defaults`. Call before importing pyplot."""
    import matplotlib  # noqa: PLC0415 -- deferred; matplotlib is not a package dependency

    matplotlib.use("Agg")
    matplotlib.rcParams.update(figure_defaults())
