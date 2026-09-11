"""`tools/write_commit.py`: the commit a wheel or the `.exe` is built with.

Build machinery rather than package code, and tested here because every way it can go
wrong is silent. A missing `-dirty` claims a build reproduces a commit it does not; a
lost "keep what is there" turns every wheel's commit into None, since `uv build` makes
the sdist from the checkout and then the wheel *from the sdist*, where no git is left to
ask; a template that stops parsing takes the stamp with it (lab record, task 20).

Loaded by path: `tools/` is not a package, and the hatchling build hook loads it the same
way. `commit_of` and `is_dirty` are the seam every test monkeypatches -- they are the two
questions that need a real repository, and only the pair of them is asked of one here.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess

import pytest

import mainspring

_GIT_IDENTITY = [
    "-c", "user.email=test@example.invalid",
    "-c", "user.name=Test",
    "-c", "commit.gpgsign=false",
]


def _load():
    path = os.path.join(mainspring.ROOT, "tools", "write_commit.py")
    if not os.path.isfile(path):
        pytest.skip("no tools/write_commit.py beside this install; not a source tree")
    spec = importlib.util.spec_from_file_location("write_commit_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def write_commit():
    return _load()


@pytest.fixture
def target(tmp_path):
    return str(tmp_path / "pkg" / "_commit.py")


def _stated(module, path):
    """What the generated module says, read back the way `mainspring.report` would."""
    namespace: dict = {}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), namespace)  # noqa: S102
    stated = namespace["COMMIT"]
    assert module.read(path) == (True, stated)  # the parser and the interpreter agree
    return stated


def test_a_clean_checkout_is_stamped_with_its_head(write_commit, target, monkeypatch):
    monkeypatch.setattr(write_commit, "commit_of", lambda root: "abc1234")
    monkeypatch.setattr(write_commit, "is_dirty", lambda root: False)
    assert write_commit.write("ignored", target) == "abc1234"
    assert _stated(write_commit, target) == "abc1234"


def test_a_dirty_checkout_says_so(write_commit, target, monkeypatch):
    """A bare sha would claim the build matched that commit. It did not."""
    monkeypatch.setattr(write_commit, "commit_of", lambda root: "abc1234")
    monkeypatch.setattr(write_commit, "is_dirty", lambda root: True)
    assert write_commit.write("ignored", target) == "abc1234-dirty"
    assert _stated(write_commit, target) == "abc1234-dirty"


def test_no_repository_and_nothing_there_stamps_None(write_commit, target, monkeypatch):
    monkeypatch.setattr(write_commit, "commit_of", lambda root: None)
    assert write_commit.write("ignored", target) is None
    assert _stated(write_commit, target) is None


def test_no_repository_keeps_a_stamp_that_is_already_there(write_commit, target, monkeypatch):
    """The wheel-from-sdist case: the sdist's commit is the only one left to carry."""
    monkeypatch.setattr(write_commit, "commit_of", lambda root: "abc1234")
    monkeypatch.setattr(write_commit, "is_dirty", lambda root: False)
    write_commit.write("ignored", target)

    monkeypatch.setattr(write_commit, "commit_of", lambda root: None)
    assert write_commit.write("ignored", target) == "abc1234"
    assert _stated(write_commit, target) == "abc1234"


def test_a_repository_overwrites_a_stamp_that_is_already_there(write_commit, target, monkeypatch):
    """A second build in the same tree records the second build, not the first."""
    monkeypatch.setattr(write_commit, "commit_of", lambda root: "abc1234")
    monkeypatch.setattr(write_commit, "is_dirty", lambda root: False)
    write_commit.write("ignored", target)

    monkeypatch.setattr(write_commit, "commit_of", lambda root: "def5678")
    assert write_commit.write("ignored", target) == "def5678"
    assert _stated(write_commit, target) == "def5678"


def test_reading_something_that_is_not_a_stamp(write_commit, tmp_path):
    absent = str(tmp_path / "absent.py")
    assert write_commit.read(absent) == (False, None)
    other = tmp_path / "other.py"
    other.write_text("COMMIT = 'not the generated form'\n", encoding="utf-8")
    assert write_commit.read(str(other)) == (False, None)


def test_remove_takes_the_stamp_away_and_does_not_mind_its_absence(
        write_commit, target, monkeypatch):
    monkeypatch.setattr(write_commit, "commit_of", lambda root: None)
    write_commit.write("ignored", target)
    assert write_commit.remove(target) is True
    assert not os.path.exists(target)
    assert write_commit.remove(target) is False


@pytest.mark.skipif(shutil.which("git") is None, reason="no git on this machine")
def test_is_dirty_counts_changes_and_not_ignored_files(write_commit, tmp_path):
    """The generated stamp, `dist/` and the numba seed are ignored, so no build is dirty
    merely for having been built."""
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        subprocess.run(["git", *_GIT_IDENTITY, "-C", os.fspath(root), *args],
                       capture_output=True, text=True, check=True)

    git("init", "-q")
    (root / ".gitignore").write_text("built.py\n", encoding="utf-8")
    (root / "file.txt").write_text("x\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "initial")
    assert write_commit.is_dirty(os.fspath(root)) is False

    (root / "built.py").write_text("COMMIT = None\n", encoding="utf-8")
    assert write_commit.is_dirty(os.fspath(root)) is False

    (root / "file.txt").write_text("y\n", encoding="utf-8")
    assert write_commit.is_dirty(os.fspath(root)) is True


@pytest.mark.skipif(shutil.which("git") is None, reason="no git on this machine")
def test_is_dirty_ignores_untracked_files_but_not_modified_ones(write_commit, tmp_path):
    """`uv` leaves an untracked marker in the checkout it builds a git dependency from
    (lab record, task 22); a clean build must not stamp `-dirty` merely because it is
    there. A modified tracked file is a real change and still earns the suffix."""
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        subprocess.run(["git", *_GIT_IDENTITY, "-C", os.fspath(root), *args],
                       capture_output=True, text=True, check=True)

    git("init", "-q")
    (root / "file.txt").write_text("x\n", encoding="utf-8")
    git("add", "file.txt")
    git("commit", "-q", "-m", "initial")
    assert write_commit.is_dirty(os.fspath(root)) is False

    (root / ".ok").write_text("", encoding="utf-8")
    assert write_commit.is_dirty(os.fspath(root)) is False

    (root / "file.txt").write_text("y\n", encoding="utf-8")
    assert write_commit.is_dirty(os.fspath(root)) is True


@pytest.mark.skipif(shutil.which("git") is None, reason="no git on this machine")
def test_the_guard_is_the_one_report_states_rather_than_a_second_copy(write_commit, tmp_path):
    """Imported from the tree being built, so the two can never drift apart.

    The guard matters at build time for the reason it matters at run time: git searches
    parent directories, so a source tree unpacked inside some other checkout would
    otherwise bake that repository's HEAD into the wheel (lab record, task 19).
    """
    from mainspring import report

    assert write_commit.commit_of(mainspring.ROOT) == report.commit_of(mainspring.ROOT)
    inside = tmp_path / "not-a-repo"
    inside.mkdir()
    assert write_commit.commit_of(os.fspath(inside)) is None
