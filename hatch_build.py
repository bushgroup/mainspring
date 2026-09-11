"""Hatchling build hook: put the commit into the wheel and into the sdist.

Hatchling looks for this file at the project root by name, so registering the hook is one
empty table per target in `pyproject.toml`. All it does is run `tools/write_commit.py` and
force the module it generates into the artefact -- the module is gitignored, and hatchling
selects files by what git tracks, so it would otherwise be left behind exactly where it is
needed.

**A commit is not a version, and this hook never touches one.** hatch-vcs is the reflex
answer to "stamp the build" and it is the wrong shape: it derives the *version* from git
tags, which would collide head-on with the three hand-carried literals that
`tools/check_public.py:declared_versions` exists to cross-check -- and with the Inno Setup
define that also names the installer's own file (lab record, task 20).

Two things here are not obvious:

* **The sdist is stamped too, and has to be.** `uv build` builds the sdist from the
  checkout and then the wheel *from that sdist*, in an unpacked tree with no `.git` in it.
  A hook on the wheel alone would produce a wheel that honestly reports None every time.
* **An editable install is skipped.** It resolves to the checkout, which answers from git
  and should not be handed a second, staler answer -- and `uv run` re-syncs often enough
  that stamping there would rewrite the file under every command.
"""

from __future__ import annotations

import importlib.util
import os

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CommitBuildHook(BuildHookInterface):
    """Generate `mainspring/_commit.py` for a standard build and clean up after it."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        if version != "standard":
            return
        writer = self._write_commit()
        target = os.path.join(self.root, "src", "mainspring", "_commit.py")
        self._was_there = os.path.isfile(target)
        commit = writer.write(self.root, target)
        # The sdist keeps the source layout; the wheel installs the package itself.
        inside = ("src/mainspring/_commit.py" if self.target_name == "sdist"
                  else "mainspring/_commit.py")
        build_data["force_include"][target] = inside
        self.app.display_info(f"{inside}: COMMIT = {commit!r}")

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        """Leave the tree as the build found it: a checkout carries no generated module."""
        if version != "standard" or getattr(self, "_was_there", False):
            return
        self._write_commit().remove(os.path.join(self.root, "src", "mainspring", "_commit.py"))

    def _write_commit(self):
        """`tools/write_commit.py`, loaded by path -- the build environment is isolated."""
        path = os.path.join(self.root, "tools", "write_commit.py")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"{path} is missing, so this build cannot record the commit it was built "
                "from (lab record, task 20)."
            )
        spec = importlib.util.spec_from_file_location("mainspring_write_commit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
