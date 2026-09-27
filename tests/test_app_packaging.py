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


def test_seed_numba_cache_copies_the_bundled_seed(tmp_path, monkeypatch):
    frozen_as(monkeypatch, tmp_path)
    seed = tmp_path / "bundle" / "numba_cache_seed"
    seed.mkdir(parents=True)
    (seed / "warm.nbi").write_text("compiled")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    app._seed_numba_cache(str(cache_dir))

    assert (cache_dir / "warm.nbi").read_text() == "compiled"


def frozen_as(monkeypatch, tmp_path, exe_bytes=b"exe"):
    """A frozen build whose executable is a real file, so its stamp can be read."""
    exe = tmp_path / "bundle" / "mainspring.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(exe_bytes)
    monkeypatch.setattr(app.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setattr(app.sys, "executable", str(exe))
    return exe


def test_seed_numba_cache_seeds_a_populated_cache_for_a_new_executable(tmp_path, monkeypatch):
    """An upgrade lands on a folder the previous version filled; the new seed must still
    arrive, and what numba compiled there stays beside it."""
    frozen_as(monkeypatch, tmp_path)
    seed = tmp_path / "bundle" / "numba_cache_seed"
    seed.mkdir(parents=True)
    (seed / "warm.nbi").write_text("from the build")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "already-compiled.nbi").write_text("from a real run")

    app._seed_numba_cache(str(cache_dir))

    assert (cache_dir / "warm.nbi").read_text() == "from the build"
    assert (cache_dir / "already-compiled.nbi").read_text() == "from a real run"


def test_seed_numba_cache_copies_once_per_executable(tmp_path, monkeypatch):
    exe = frozen_as(monkeypatch, tmp_path)
    seed = tmp_path / "bundle" / "numba_cache_seed"
    seed.mkdir(parents=True)
    (seed / "warm.nbi").write_text("from the build")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    app._seed_numba_cache(str(cache_dir))

    (cache_dir / "warm.nbi").write_text("numba wrote this since")
    app._seed_numba_cache(str(cache_dir))
    assert (cache_dir / "warm.nbi").read_text() == "numba wrote this since"

    exe.write_bytes(b"a different, larger executable")
    app._seed_numba_cache(str(cache_dir))
    assert (cache_dir / "warm.nbi").read_text() == "from the build"


def test_seed_numba_cache_is_a_no_op_with_no_bundled_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(app.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    app._seed_numba_cache(str(cache_dir))  # must not raise with no bundle at all

    assert list(cache_dir.iterdir()) == []


# --- numba's cache in a frozen program ------------------------------------------------


def test_a_frozen_programs_cache_folder_does_not_depend_on_the_launch_folder(
        tmp_path, monkeypatch):
    """PyInstaller's relative `co_filename` made numba name the folder after the working
    directory; this one is named after the program and the package only."""
    from mainspring.uimf import decode

    monkeypatch.setattr(decode.sys, "executable", r"C:\Program Files\mainspring\mainspring.exe")
    seen = set()
    for cwd in (tmp_path / "a", tmp_path / "b"):
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        seen.add(decode.frozen_cache_folder(r"C:\cache", os.path.join("mainspring", "uimf",
                                                                       "decode.py")))
    assert seen == {os.path.join(r"C:\cache", "mainspring_uimf")}


def test_the_frozen_locator_answers_only_in_a_frozen_program(tmp_path, monkeypatch):
    from numba.core import caching, config

    from mainspring.uimf import decode

    monkeypatch.setattr(decode.sys, "frozen", True, raising=False)
    monkeypatch.setattr(decode.sys, "executable", str(tmp_path / "clockwork.exe"))
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    assert decode.install_frozen_cache_locator()
    impl = getattr(caching, "CacheImpl", None) or getattr(caching, "_CacheImpl")
    locator = impl._locator_classes[0]

    def kernel():  # pragma: no cover -- never called, only located
        return 0

    found = locator.from_function(kernel, os.path.join("mainspring", "uimf", "frame.py"))
    assert found is not None
    assert found.get_cache_path() == str(tmp_path / "cache" / "clockwork_uimf")
    monkeypatch.delattr(decode.sys, "frozen")
    assert locator.from_function(kernel, "frame.py") is None


def test_warming_from_a_source_tree_compiles_and_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_CACHE_DIR", os.environ.get("NUMBA_CACHE_DIR", ""))
    assert app.main([app.WARM_OPTION, str(tmp_path / "seed")]) == 0
