"""Import-level checks that hold in a bare clone with no data."""

import os
import sys


def test_uimf_layer_does_not_import_qt(monkeypatch):
    for mod in [m for m in sys.modules if m.startswith(("PySide6", "pyqtgraph", "shiboken6"))]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    import mainspring.uimf  # noqa: F401

    assert not any(m.startswith(("PySide6", "pyqtgraph")) for m in sys.modules)


def test_version_and_lab_dir():
    import mainspring

    assert mainspring.__version__
    lab = mainspring.lab_dir()
    assert lab is None or os.path.isdir(lab)
