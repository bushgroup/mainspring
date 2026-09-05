"""`viewer.app`'s frozen-build helpers: the numba cache location and its seeding.

No `QApplication` needed -- `mainspring.viewer.app` imports Qt lazily, inside `main()`,
so these two functions are plain Python (lab record, task 07).
"""

from __future__ import annotations

import os

from mainspring.viewer import app


def test_numba_cache_dir_is_under_localappdata(monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", r"C:\fake\localappdata")
    cache_dir = app._numba_cache_dir()
    assert cache_dir == os.path.join(r"C:\fake\localappdata", "mainspring", "numba-cache")


def test_seed_numba_cache_does_nothing_outside_a_frozen_build(tmp_path, monkeypatch):
    monkeypatch.delattr(app.sys, "frozen", raising=False)
    seed = tmp_path / "bundle" / "numba_cache_seed"
    seed.mkdir(parents=True)
    (seed / "warm.nbi").write_text("compiled")
    cache_dir = tmp_path / "cache"

    app._seed_numba_cache(str(cache_dir))

    assert not cache_dir.exists()


def test_seed_numba_cache_copies_the_bundled_seed_once(tmp_path, monkeypatch):
    monkeypatch.setattr(app.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    seed = tmp_path / "bundle" / "numba_cache_seed"
    seed.mkdir(parents=True)
    (seed / "warm.nbi").write_text("compiled")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    app._seed_numba_cache(str(cache_dir))

    assert (cache_dir / "warm.nbi").read_text() == "compiled"


def test_seed_numba_cache_leaves_an_already_populated_cache_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(app.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    seed = tmp_path / "bundle" / "numba_cache_seed"
    seed.mkdir(parents=True)
    (seed / "warm.nbi").write_text("from the build")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "already-compiled.nbi").write_text("from a real run")

    app._seed_numba_cache(str(cache_dir))

    assert not (cache_dir / "warm.nbi").exists()
    assert (cache_dir / "already-compiled.nbi").read_text() == "from a real run"


def test_seed_numba_cache_is_a_no_op_with_no_bundled_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(app.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    app._seed_numba_cache(str(cache_dir))  # must not raise with no bundle at all

    assert list(cache_dir.iterdir()) == []
