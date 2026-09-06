# PyInstaller build spec for the mainspring viewer.
#
# Run through `tools/build_exe.ps1`, not `pyinstaller` directly, so the working and dist
# paths land under `dist/`/`build/` (both gitignored) regardless of the caller's cwd, and so
# `-Mode onedir` reaches this file the same way it reaches the packaging decision.
#
# The excludes below are PySide6 submodules the viewer never imports -- WebEngine, Qml/Quick,
# Multimedia, the 3D and remaining device-facing modules -- cut because PySide6 ships every
# Qt module in one wheel and PyInstaller's default Qt hook otherwise pulls each one's own
# native Qt libraries in with it (lab record, task 07). `mainspring.uimf.decode`'s numba path
# and PySide6/pyqtgraph themselves are covered by `pyinstaller-hooks-contrib`'s `hook-numba.py`
# and PyInstaller's own Qt hook utility; neither needs a hand-written hook here.
#
# MAINSPRING_PACKAGE_MODE selects onefile (default, single .exe, extracts on every start) or
# onedir (a folder, no extraction, for an Inno Setup installer) -- `tools/build_exe.ps1 -Mode`
# sets it. Task 07 measured onefile's extract-every-start cost too slow to ship (progress log).

import os

from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
MODE = os.environ.get("MAINSPRING_PACKAGE_MODE", "onedir")
if MODE not in ("onefile", "onedir"):
    raise ValueError(f"MAINSPRING_PACKAGE_MODE must be 'onefile' or 'onedir', got {MODE!r}")
# SPECPATH is injected by PyInstaller into this file's exec namespace; resolving the
# entry script against it means the build works from any cwd, not just this directory.
ENTRYPOINT = os.path.join(SPECPATH, "entrypoint.py")

# The icon is package data, not a packaging asset: the viewer sets it on its own windows
# through `mainspring.viewer.app`, so it has to travel with the package for a source run
# and a wheel install as well as for this build, and `collect_data_files("mainspring")`
# below already brings it into the bundle. This is the same file, handed to the Windows
# resource section of the .exe. `tools/make_icon.py` regenerates it from packaging/icon/.
ICON = os.path.join(
    os.path.dirname(SPECPATH), "src", "mainspring", "viewer", "resources", "mainspring.ico"
)
if not os.path.isfile(ICON):
    raise RuntimeError(f"{ICON} is missing -- run `uv run tools/make_icon.py`.")

# `mainspring.viewer.app._seed_numba_cache` copies this into NUMBA_CACHE_DIR on a frozen
# build's first launch, so the 2.5-4.7 s first-ever-launch JIT cost (notes/reader-layer.md,
# task 04) is paid at build time instead. Required, not optional: an unwarmed build would
# still work, just slowly on its first launch, and that regression should fail the build
# rather than ship quietly.
SEED_DIR = os.path.join(SPECPATH, "numba_cache_seed")
if not os.path.isdir(SEED_DIR) or not os.listdir(SEED_DIR):
    raise RuntimeError(
        "packaging/numba_cache_seed is missing or empty -- run "
        "`uv run tools/warm_numba_cache.py` before building (lab record, task 07)."
    )
NUMBA_SEED_DATAS = Tree(SEED_DIR, prefix="numba_cache_seed")

PYSIDE6_EXCLUDES = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.QtWebView",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickTest",
    "PySide6.QtQuickWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtSpatialAudio",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtSerialBus",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtRemoteObjects",
    "PySide6.QtNetworkAuth",
    "PySide6.QtHttpServer",
    "PySide6.QtScxml",
    "PySide6.QtStateMachine",
    "PySide6.QtHelp",
    "PySide6.QtDesigner",
    "PySide6.QtUiTools",
    "PySide6.QtAxContainer",
    "PySide6.QtCanvasPainter",
    "PySide6.QtTextToSpeech",
    "PySide6.QtDBus",
]

a = Analysis(
    [ENTRYPOINT],
    pathex=[],
    binaries=[],
    datas=collect_data_files("mainspring"),
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=PYSIDE6_EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
# Tree() yields the 3-entry TOC form (dest, src, typecode), unlike Analysis(datas=...)'s
# 2-entry (src, dest_dir) form, so it is appended to the already-built TOC rather than
# passed into Analysis itself.
a.datas += NUMBA_SEED_DATAS

# --- Provenance guard --------------------------------------------------------------------
# PyInstaller resolves each collected binary's DLL dependencies by searching the binary's
# own directory, then sys.path, then PATH. A PATH carrying another Python distribution
# substitutes its DLLs quietly: with anaconda3/Library/bin on the build shell's PATH, its
# ICU 73 `icuuc.dll` (versioned symbols) shadowed the Windows `icuuc.dll` (unversioned
# symbols) that PySide6's Qt6Core links, and every launch of the frozen build -- on the
# build machine and on a clean one -- died with "DLL load failed while importing QtCore:
# The specified procedure could not be found". The same PATH also swapped in anaconda's
# OpenSSL, TBB, MSVCP140 and UCRT forwarders. `tools/build_exe.ps1` strips PATH down to the
# Windows directories and uv's for the build; this refuses to ship anything that still gets
# through. Legitimate sources are exactly the venv, the interpreter it was created from, and
# this repo (Windows system DLLs never appear in the TOC; PyInstaller excludes them).
import sys

def _root(path):
    return os.path.normcase(os.path.realpath(path)).rstrip(os.sep) + os.sep

ALLOWED_ROOTS = tuple(_root(p) for p in (sys.prefix, sys.base_prefix, os.path.dirname(SPECPATH)))

def _foreign(toc):
    for entry in toc:
        src = os.path.normcase(os.path.realpath(entry[1]))
        if not src.startswith(ALLOWED_ROOTS):
            yield entry

foreign = list(_foreign(a.binaries)) + list(_foreign(a.datas))
if foreign:
    listing = "\n".join(f"  {dest}  <-  {src}" for dest, src, *_ in foreign)
    raise RuntimeError(
        "Refusing to build: these files come from outside the venv, its base interpreter and "
        "this repo (a foreign directory on PATH, most likely another Python distribution):\n"
        + listing
    )

# --- One C++ runtime at the bundle root ---------------------------------------------------
# numba's and llvmlite's extension modules import MSVCP140.dll by name and get whichever
# copy PyInstaller found first; Windows then reuses that already-loaded copy for Qt6Core,
# which needs the newer 14.44 exports PySide6 ships with. Pin the root copy to PySide6's,
# the newest in the venv, so every importer in the bundle shares one runtime that is at
# least as new as anything expects. (VCRUNTIME140 comes from the base interpreter and is
# already the same 14.44 line.)
import PySide6

_pyside_msvcp = os.path.join(os.path.dirname(PySide6.__file__), "MSVCP140.dll")
if not os.path.isfile(_pyside_msvcp):
    raise RuntimeError(f"PySide6 no longer ships MSVCP140.dll at {_pyside_msvcp}; revisit this pin.")
a.binaries = [e for e in a.binaries if e[0].lower() != "msvcp140.dll"] + [
    ("MSVCP140.dll", _pyside_msvcp, "BINARY")
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

_exe_common = dict(
    name="mainspring",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)

if MODE == "onedir":
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **_exe_common)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, upx_exclude=[], name="mainspring")
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], runtime_tmpdir=None, **_exe_common)
