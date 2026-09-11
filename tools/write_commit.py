"""Write `src/mainspring/_commit.py`: the commit a built artefact was built from.

A wheel and the `.exe` leave the checkout behind, so `report.commit_of(ROOT)` can only
answer None for them (lab record, task 19) and every number they produce identifies the
code as "some 1.0.0". The version moves once per release and the code moves every
commit, so that is not enough to get from a result back to what computed it. This script
records the commit at build time, and records it as a *commit*: nothing here reads,
derives or touches a version, which is the arrangement `declared_versions` in
`check_public.py` exists to protect.

Run from `hatch_build.py` for the wheel and the sdist and from `tools/build_exe.ps1`
before PyInstaller, so both artefacts carry the commit by the same route. The generated
module is gitignored and rewritten before every build, like `packaging/numba_cache_seed/`.

Three ways this ends:

* **a checkout** -- the short HEAD, with `-dirty` appended when the tree carries
  uncommitted changes, since a bare sha would claim the build matched that commit;
* **no checkout, but a generated module already there** -- left alone. That is how a
  wheel keeps a commit: `uv build` builds the sdist from the checkout and then the wheel
  *from the sdist*, where there is no git left to ask;
* **neither** -- `COMMIT = None`, the same honest answer a tarball has always given.

Stdlib only, and no import of an installed mainspring: this runs inside hatchling's
isolated build environment, where the only thing importable besides the standard library
is the source tree being built.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "src", "mainspring", "_commit.py")

_TEMPLATE = '''\
"""The commit this artefact was built from -- generated, gitignored, never edited.

Written by `tools/write_commit.py` at build time and imported defensively by
`mainspring.report`, which prefers a checkout's live HEAD over this and falls back to
None when neither answers (lab record, task 20). A `-dirty` suffix means the build tree
carried uncommitted changes, so the sha names the commit the build was *closest to*
rather than one that reproduces it.
"""

COMMIT: str | None = {value!r}
'''

_VALUE = re.compile(r"^COMMIT: str \| None = (.+)$", re.MULTILINE)


def commit_of(root: str) -> str | None:
    """`report.commit_of`, imported from the tree being built rather than reimplemented.

    The guard matters here for the same reason it matters at run time: git searches
    parent directories, so a source tree unpacked inside some other checkout would
    otherwise bake *that* repository's HEAD into the wheel -- task 19's bug, moved to
    build time. `mainspring.report` imports nothing outside the standard library, which
    is what makes this reachable from an isolated build environment.
    """
    sys.path.insert(0, os.path.join(root, "src"))
    try:
        from mainspring.report import commit_of as guarded  # noqa: PLC0415 -- see above
    finally:
        sys.path.pop(0)
    return guarded(root)


def is_dirty(root: str) -> bool:
    """Whether the checkout at `root` carries uncommitted changes.

    Ignored files are not changes -- `--porcelain` leaves them out, which is why the
    generated module itself, `dist/` and the numba seed never make a build dirty.
    Untracked files do not count either (`--untracked-files=no`): hatchling selects a
    wheel's contents by what git tracks, so an untracked file cannot reach the wheel and
    cannot make the sha misdescribe what was built -- which is also why `uv`'s own
    untracked marker in a git-dependency checkout must not stamp a clean build `-dirty`
    (lab record, task 22).
    """
    try:
        done = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(done.stdout.strip())


def read(target: str = TARGET) -> tuple[bool, str | None]:
    """`(the module is there, the commit it states)`; `(False, None)` when it is not."""
    try:
        with open(target, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return False, None
    found = _VALUE.search(text)
    if not found:
        return False, None
    try:
        value = ast.literal_eval(found.group(1))
    except (SyntaxError, ValueError):
        return False, None
    return True, value if isinstance(value, str) else None


def write(root: str = ROOT, target: str = TARGET) -> str | None:
    """Generate the module for the tree at `root` and return the commit it now states."""
    commit = commit_of(root)
    if commit is None:
        present, existing = read(target)
        if present:
            return existing  # an sdist's own stamp; there is no git here to better it
    elif is_dirty(root):
        commit += "-dirty"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_TEMPLATE.format(value=commit))
    return commit


def remove(target: str = TARGET) -> bool:
    """Delete the generated module; True if there was one. A checkout keeps none."""
    try:
        os.remove(target)
    except OSError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--remove", action="store_true",
                        help="delete the generated module instead of writing it")
    parser.add_argument("--quiet", action="store_true", help="say nothing on success")
    args = parser.parse_args(argv)

    if args.remove:
        gone = remove()
        if not args.quiet:
            print(f"{TARGET}: {'removed' if gone else 'nothing to remove'}")
        return 0
    commit = write()
    if not args.quiet:
        print(f"{TARGET}: COMMIT = {commit!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
