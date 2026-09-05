"""The seam, the layout, and the honesty of the stubs.

These are the checks that make the module layout a rule rather than a habit: every
module named in the layout exists and imports, the data layer stays free of Qt, and a
function that has not been written yet says so instead of returning something plausible
(lab record, task 02).
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

UIMF_MODULES = ("cache", "calib", "cli", "decode", "frame", "raster", "reader")
VIEWER_MODULES = (
    "app", "heatmap", "info_panel", "main_window", "settings", "side_plots", "workers",
)


@pytest.mark.parametrize("name", UIMF_MODULES)
def test_uimf_modules_import(name):
    assert importlib.import_module(f"mainspring.uimf.{name}") is not None


@pytest.mark.parametrize("name", VIEWER_MODULES)
def test_viewer_modules_import(name):
    assert importlib.import_module(f"mainspring.viewer.{name}") is not None


def test_uimf_layer_imports_no_qt_in_a_fresh_interpreter():
    """The seam, checked where it cannot be faked by an already-imported module.

    A test in this process can only see what this process has loaded, and pytest-qt has
    loaded Qt long before we get here. So ask a fresh interpreter, which is also what a
    pipeline machine's import looks like.
    """
    code = (
        "import sys, mainspring.uimf, mainspring.uimf.reader, mainspring.uimf.raster;"
        "leaked = [m for m in sys.modules if m.startswith(('PySide6', 'pyqtgraph', 'shiboken6'))];"
        "print(leaked)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]", done.stdout


def test_viewer_does_not_leak_into_the_uimf_layer():
    """Dependencies point one way: nothing under `uimf` may name the viewer."""
    for name in UIMF_MODULES:
        module = importlib.import_module(f"mainspring.uimf.{name}")
        assert "mainspring.viewer" not in (module.__doc__ or "").replace("`", ""), name
        source = getattr(module, "__file__", None)
        assert source is not None
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert "from ..viewer" not in text and "import mainspring.viewer" not in text, name


def test_unbuilt_functions_raise_rather_than_guess():
    """An unwritten body raises `NotImplementedError` and names the task that fills it."""
    from mainspring.uimf import calib, decode, reader

    calls = (
        lambda: decode.decode_intensities(b""),
        lambda: decode.lzf_decompress(b""),
        lambda: calib.Calibration(1.0, 0.0, 1.0).mz(0),
        lambda: calib.arrival_time_ms(0, 1.0),
        lambda: reader.UimfFile.global_params(object()),
    )
    for call in calls:
        with pytest.raises(NotImplementedError, match="task"):
            call()
