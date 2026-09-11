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
