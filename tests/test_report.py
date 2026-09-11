"""`report.commit_of`: a commit is that repository's own commit, or it is None.

Git searches parent directories, so the obvious `rev-parse --short HEAD` in an installed
package's directory answers for whatever checkout encloses it -- under a wheel in
`<venv>/Lib`, `mainspring_commit` was the *host* project's HEAD (lab record, task 19).
The pin for that is `test_a_subdirectory_is_not_the_repository`: it is the case that used
to return a wrong sha rather than nothing.

Everything here builds its own repository in `tmp_path` with `git -c` overrides for
identity and signing, so nothing depends on the machine's global git config, and the
module skips whole when there is no `git` to run.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from mainspring import report

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="no git on this machine")

# Enough identity to commit, and no signing, whatever the machine's global config says.
_GIT_IDENTITY = [
    "-c", "user.email=test@example.invalid",
    "-c", "user.name=Test",
    "-c", "commit.gpgsign=false",
]


def _git(cwd, *args: str) -> str:
    done = subprocess.run(
        ["git", *_GIT_IDENTITY, "-C", os.fspath(cwd), *args],
        capture_output=True, text=True, check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A one-commit git repository, returned as (path, its short HEAD)."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "file.txt").write_text("x\n", encoding="utf-8")
    _git(root, "add", "file.txt")
    _git(root, "commit", "-q", "-m", "initial")
    return root, _git(root, "rev-parse", "--short", "HEAD")


def test_a_checkout_answers_its_own_commit(repo):
    root, head = repo
    assert report.commit_of(root) == head


def test_a_subdirectory_is_not_the_repository(repo):
    """The regression pin: this used to answer the enclosing repository's commit."""
    root, _ = repo
    inside = root / "sub" / "deeper"
    inside.mkdir(parents=True)
    assert report.commit_of(inside) is None


def test_a_directory_that_is_no_repository_answers_none(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert report.commit_of(plain) is None


def test_a_path_that_is_not_there_answers_none(tmp_path):
    assert report.commit_of(tmp_path / "absent") is None


def test_a_repository_with_no_commits_answers_none(tmp_path):
    """An unborn HEAD is not a commit, and `rev-parse` failing is the whole guard."""
    root = tmp_path / "unborn"
    root.mkdir()
    _git(root, "init", "-q")
    assert report.commit_of(root) is None


def test_a_worktree_answers_its_own_commit(repo):
    """The test that fails if someone reaches for an `isdir(.git)` shortcut.

    A `git worktree` root carries `.git` as a *file* pointing into the parent
    repository's admin directory, so a pre-check for a `.git` directory rejects a
    perfectly real checkout.
    """
    root, _ = repo
    linked = root.parent / "linked"
    _git(root, "worktree", "add", "-q", "-b", "side", os.fspath(linked))
    assert not (linked / ".git").is_dir()
    assert report.commit_of(linked) == _git(linked, "rev-parse", "--short", "HEAD")


def test_a_path_spelled_differently_is_still_the_same_repository(repo):
    """Git answers in long form; a caller may hold anything that resolves to the same place.

    Compared with `realpath` rather than `normpath` for exactly this reason -- on Windows
    an 8.3 short name (`C:/Users/BUSH-L~1/...`) is the shape that appears in practice, and
    it must not cost a real checkout its commit (lab record, task 19).
    """
    root, head = repo
    spelled = os.path.join(os.fspath(root), "sub", "..")
    os.mkdir(os.path.join(os.fspath(root), "sub"))
    assert report.commit_of(spelled) == head


def test_stamp_carries_a_commit_or_an_honest_none():
    stamped = report.stamp({"x": 1}, task="test")
    for key in ("mainspring_commit", "lab_commit"):
        assert stamped[key] is None or isinstance(stamped[key], str)
    assert stamped["results"] == {"x": 1}


# --- Which commit a stamp reports ------------------------------------------------------
# `commit_of` above answers about a *directory*; `mainspring_commit` answers about *this
# mainspring*, and has two sources: the checkout it is running out of and, failing that,
# the commit the build that produced this wheel or `.exe` was made from
# (`tools/write_commit.py`, lab record, task 20). The order is checkout first, because git
# is live and a generated module is a snapshot -- so a build artefact left behind in a
# source tree can never speak over the tree it is sitting in.


def test_a_checkout_wins_over_the_commit_a_build_recorded(monkeypatch):
    monkeypatch.setattr(report, "commit_of", lambda repo: "1111111")
    monkeypatch.setattr(report, "_BUILT_COMMIT", "2222222")
    assert report.mainspring_commit() == "1111111"


def test_the_build_answers_when_there_is_no_checkout(monkeypatch):
    """The wheel and the `.exe`: `ROOT` is inside an install, so only the build can say."""
    monkeypatch.setattr(report, "commit_of", lambda repo: None)
    monkeypatch.setattr(report, "_BUILT_COMMIT", "2222222")
    assert report.mainspring_commit() == "2222222"


def test_neither_is_still_None(monkeypatch):
    monkeypatch.setattr(report, "commit_of", lambda repo: None)
    monkeypatch.setattr(report, "_BUILT_COMMIT", None)
    assert report.mainspring_commit() is None


def test_running_from_a_checkout_reports_that_checkout(monkeypatch):
    """The path every test run from a source tree takes, asserted rather than assumed.

    The generated module is absent in a clean checkout and present in one that has just
    built an `.exe`; neither may change the answer, which is this repository's own HEAD.
    """
    head = report.commit_of(report.ROOT)
    if head is None:
        pytest.skip("not running out of a checkout; a wheel or a tarball has no HEAD")
    for built in (None, "2222222"):
        monkeypatch.setattr(report, "_BUILT_COMMIT", built)
        assert report.mainspring_commit() == head


def test_a_dirty_build_commit_is_still_a_plain_string(monkeypatch):
    """`-dirty` is a suffix on the sha, not a second field: the stamp promises `str | None`."""
    monkeypatch.setattr(report, "commit_of", lambda repo: None)
    monkeypatch.setattr(report, "_BUILT_COMMIT", "2222222-dirty")
    assert report.stamp({"x": 1})["mainspring_commit"] == "2222222-dirty"
