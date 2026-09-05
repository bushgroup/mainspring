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
    icon=None,
)

if MODE == "onedir":
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **_exe_common)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, upx_exclude=[], name="mainspring")
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], runtime_tmpdir=None, **_exe_common)
