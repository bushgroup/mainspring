"""Reporting conventions: stamped `results.json`, commented CSV tables, figures.

The small set of habits that make a number found in an exploration directory six
months later still traceable, gathered here so every lab-side script writes them the
same way. Mirrors the convention of the lab's other packages (schamp.report).

* **`results.json` is stamped** with the date, the mainspring version, and the commit
  of the code repo and -- when a lab checkout resolves -- of the lab repo, wrapped
  around the results under a `results` key. A commit is recorded only where one is
  knowable: run from anything but a checkout of the repository in question it is
  `None`, never a neighbouring repository's.
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


def _is_same_dir(a: str, b: str) -> bool:
    """Whether two paths name the same directory, as git would have to agree they do.

    `realpath` on both sides, not `normpath`: git answers `--show-toplevel` in long
    form, so a caller holding an 8.3 short name (`C:/Users/BUSH-L~1/...`) compares
    unequal against git's `C:/Users/bush-lab-admin/...` and a real checkout would
    silently lose its commit.
    """
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def commit_of(repo: str | os.PathLike[str]) -> str | None:
    """The short HEAD commit of the git repository *rooted at* `repo`, else None.

    Git searches parent directories, so asking it for a commit inside an installed
    package answers for whatever checkout happens to enclose it: a wheel under
    `<venv>/Lib` reports the host project's HEAD as if it were mainspring's. The
    toplevel is read in the same call and the commit kept only when that toplevel is
    `repo` itself.

    Never raises, and answers None rather than someone else's commit: a wheel install,
    the packaged `.exe`, a tarball rather than a checkout, a directory inside an
    unrelated repository, or a machine with no `git`, all get None and a results file
    that says so honestly.
    """
    try:
        done = subprocess.run(
            ["git", "-C", os.fspath(repo), "rev-parse", "--show-toplevel", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    # splitlines, never split(): a repository path may contain spaces.
    lines = done.stdout.splitlines()
    if len(lines) != 2:
        return None
    toplevel, commit = lines[0].strip(), lines[1].strip()
    if not toplevel or not commit:
        return None
    try:
        if not _is_same_dir(toplevel, os.fspath(repo)):
            return None
    except OSError:
        return None
    return commit


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
