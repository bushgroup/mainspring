# mainspring — public-role code repo

Package `mainspring`: an interactive viewer and a Python reader for UIMF ion mobility mass
spectrometry files (PNNL's SQLite-based format, written by SLIM instruments and by the lab's
`clockwork` acquisition software). Named for the spring that drives a clockwork. This is the
**public-role half of a two-repo arrangement**: it carries `src/mainspring/`, `tools/`, `tests/`,
`docs/`, `packaging/` and the public self-check — nothing else. The development record lives in the
private sibling **`bushgroup/mainspring-lab`** (tasks, notes, explorations, sample `.uimf` files,
and the working CLAUDE.md with the full project state and never-do list).

**Sessions are based here.** A gitignored `CLAUDE.local.md` imports the lab repo's CLAUDE.md; if
you are reading this file *without* that import, you have a public clone — the package and
`tools/check_public.py` are fully usable, the lab material is not yours to see.

## Rules that live with the code

- **Run anything with `uv run <path/to/script.py>`** — no flags, no venv activation. Never rely on
  the `python` on PATH.
- **`uv run tools/check_public.py`** after touching `src/mainspring/`. Data-free: a clone with no
  `.uimf` file must pass it. Checks that need a real file report SKIPPED, never FAIL, when none is
  present; `uv run tools/fetch_testdata.py` fetches PNNL's small test excerpts into gitignored
  `external/pnnl-testdata/`, and `MAINSPRING_SMOKE_UIMF` may name any other file.
- **Two layers, one seam.** `mainspring.uimf` is the data layer (numpy, optional numba) and
  **never imports Qt**; `mainspring.viewer` is PySide6 + pyqtgraph on top of it. A pipeline that
  installs the package for the reader must not pull a GUI into its import path.
- **Every control the user can touch explains itself.** Build actions and toolbar widgets through
  `mainspring.viewer.controls` (`make_action`, `add_labelled`, `describe`), which requires a
  one-sentence tooltip; `tip=None` is a deliberate, reviewable waiver. `controls.unexplained()`
  walks the live window by widget type and `tools/check_public.py` fails on anything mute.
- **Never materialise a dense frame.** A frame is scans × TOF bins (5000 × 114688 on SLIMPHONY,
  2.3 GB as int32). Frames stay sparse (`SparseFrame`); the heatmap is rasterised to the viewport.
- **The format reference is PNNL's UIMF-Library** (C#, actively maintained). Decode, calibration
  and parameter semantics follow it; where this code differs deliberately, the docstring says so
  and the lab record says why (lab record, task 01).
- **No `.uimf` file is ever committed here** — `.gitignore` excludes the extension and
  `.githooks/pre-commit` refuses the path. Sample data live in the lab repo.
- **Windows 11 only.** The instruments and the users run Windows 11; the `.exe` is the primary
  deliverable. Nothing should *break* elsewhere, but nothing else is tested.
- **Lab-side paths resolve through `mainspring.lab_dir()`**: `$MAINSPRING_LAB`, then this root,
  then the sibling `../mainspring-lab`. Code in `src/` refers to lab material only in opaque form
  ("lab record, task NN"), never by a path that only resolves lab-side.
- **Public commit messages are self-contained statements of the change.** Task IDs may appear as
  opaque references ("lab record, task 04") at most. Trailer is `Assisted-by: <model name>`, no
  email — never `Co-Authored-By:`. `.githooks/commit-msg` rewrites, `.githooks/pre-commit` rejects
  staged files over 5 MiB and any `.uimf` path; both need `git config core.hooksPath .githooks`
  once per clone.
- **`.gitattributes` pins `* text=auto eol=lf`.**
- **Outward-facing prose (README, `docs/`, user guides) follows the `manuscript-voice` skill**
  from the lab repo. Repo-internal prose (this file, docstrings, commit messages) does not.
- **BSD 3-Clause, `LICENSE`, copyright University of Washington.** Public since 2026-09-06,
  released as `v1.0.0`; written from the first commit as if public. The version is declared
  independently in `pyproject.toml`, `mainspring.__version__` and `packaging/mainspring.iss`, and
  `check_public.py` fails unless the three agree; bump them together. The installer and the wheel
  reach users as GitHub release assets, never through the repo.

## Maintaining this file

This file stays lean: rules for working *in this repo*, nothing about the science or the project's
state. Those belong in the lab repo's CLAUDE.md and notes. Keep it under 60 lines.
