"""The seam, the layout, and the honesty of the stubs.

These are the checks that make the module layout a rule rather than a habit: every
module named in the layout exists and imports, the data layer stays free of Qt, and a
function that has not been written yet says so instead of returning something plausible
(lab record, task 02).
"""

from __future__ import annotations

import importlib
import os
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
    """An unwritten body raises `NotImplementedError` and names the task that fills it.

    The whole `uimf` layer is written now (lab record, task 03); `app.main` and the rest
    of what task 04 built are no longer on this list -- `test_viewer_smoke.py` exercises
    them instead. What is left to hold honest is what tasks 05 and 06 still owe.
    """
    from mainspring.viewer import heatmap, info_panel, settings, workers

    calls = (
        lambda: heatmap.UimfViewBox(),
        lambda: info_panel.per_push(0.0, 1, 8),
        lambda: settings.load_settings(),
        lambda: workers.RenderMailbox.put(object(), object()),
    )
    for call in calls:
        with pytest.raises(NotImplementedError, match="task"):
            call()


def test_the_data_layer_no_longer_has_stubs_in_it():
    """Task 03 filled `uimf` end to end; a `NotImplementedError` left behind there would
    be a module the reader cannot actually use."""
    import mainspring.uimf

    directory = os.path.dirname(mainspring.uimf.__file__)
    for name in UIMF_MODULES:
        with open(os.path.join(directory, f"{name}.py"), encoding="utf-8") as handle:
            source = handle.read()
        assert "raise NotImplementedError" not in source, name
