"""Shared fixtures. The synthetic ones need no data file and no lab repository.

Real files are a separate matter and a separate fixture: PNNL's excerpts are fetched
into `external/` and never committed, and the lab's own sample lives in the private
repository, so every test that wants one is parametrised over whatever this clone
happens to have and skips when that is nothing. A bare public clone runs the whole
suite green on synthetic files alone.
"""

from __future__ import annotations

import glob
import os
import sys

import pytest

from synthetic import write_synthetic_uimf

# Set before pytest-qt's `qapp` fixture builds a `QApplication` (lazily, on first use),
# so the viewer tests run with no display -- this workstation has one, but CI and a
# contributor's headless clone may not (lab record, task 04). A caller that wants a
# real, visible window sets `QT_QPA_PLATFORM` before running pytest, which this leaves
# alone.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path):
    """Every test gets its own `QSettings` store, never the user's real one.

    `ViewerSettings` persists through `QSettings(ORGANISATION, APPLICATION)`
    (`viewer/settings.py`), which defaults to the Windows registry on this platform -- a
    test that wrote there would pollute a real user's saved preferences and leak state
    between test runs and between this suite and a session's actual viewer. Redirecting
    the *default* format to a per-test ini file keeps the round trip real without ever
    touching anything outside `tmp_path` (lab record, task 06). Autouse and unconditional:
    every test gets this, including ones that never import `settings.py`, because it is
    cheap and because a test added later that does touch it should not have to remember
    to ask.
    """
    from PySide6.QtCore import QSettings

    previous_format = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    yield
    QSettings.setDefaultFormat(previous_format)


@pytest.fixture(autouse=True)
def _isolated_live_pointer(tmp_path):
    """No test reads or writes the pointer a real acquisition would publish.

    `Live` resolves the run in progress through `mainspring.interface.live_pointer_path`,
    which is a per-user file under `%LOCALAPPDATA%` (lab record, task 32). This session's
    workstation is the instrument PC, so that file can genuinely be there and name a real
    acquisition, and a suite that read it would pass or fail depending on whether someone
    was running the rig. Redirected per test for the reason `QSettings` is, and autouse
    for the same reason: a window built by any viewer test resolves a target the moment
    `Live` is touched.

    **By hand rather than with `monkeypatch`.** Naming that fixture from an autouse one
    puts it in front of every test in the suite, and the whole run then dies partway
    through with an access violation inside pyqtgraph's `GraphicsView` constructor -- on
    every run, at the same test, and never with the *same* environment variable set by
    the four lines below instead. So it is the fixture and not the redirection, whatever
    the interaction with pytest-qt's teardown turns out to be, and a save and a restore
    do not need it (lab record, task 32).
    """
    from mainspring.interface import LIVE_POINTER_ENV

    previous = os.environ.get(LIVE_POINTER_ENV)
    os.environ[LIVE_POINTER_ENV] = str(tmp_path / "live-run.json")
    yield
    if previous is None:
        os.environ.pop(LIVE_POINTER_ENV, None)
    else:
        os.environ[LIVE_POINTER_ENV] = previous


@pytest.fixture(autouse=True)
def _restore_the_default_palette():
    """No test leaves a plot palette on the process.

    `theme.apply` writes pyqtgraph's `background` and `foreground` config options, which
    are read when an item is constructed and are process-wide -- so a test that ends
    under the light palette hands the next one a window whose items are born light.
    Everything the viewer paints is set explicitly afterwards, so today that is invisible
    rather than wrong; it is still state leaking between tests, and the walk that would
    catch the consequences is itself one of the tests that would be leaking it.

    Only if the plot layer has already been imported: a data-layer test should not pull
    Qt in through a fixture that runs for every test in the suite (lab record, task 14).
    """
    yield
    if "mainspring.viewer.theme" not in sys.modules:
        return
    import pyqtgraph as pg

    from mainspring.viewer import theme

    pg.setConfigOptions(background=theme.DARK.background, foreground=theme.DARK.foreground)
    theme._ACTIVE = theme.DARK


@pytest.fixture(autouse=True)
def _restore_the_default_text_size():
    """No test leaves a text scale on the process.

    `fonts.apply` writes `QApplication.setFont`, which is process-wide and reaches every
    widget that has not been given a font of its own -- so a test that ends at 200 per
    cent hands the next one a window whose menus, panel and axes are all half again too
    big, and whose layout assertions then measure the wrong thing.

    Guarded like the palette fixture above, and for the same reason: a data-layer test
    should not pull Qt in through a fixture that runs for every test in the suite.
    """
    yield
    if "mainspring.viewer.fonts" not in sys.modules:
        return
    from PySide6.QtWidgets import QApplication

    from mainspring.viewer import fonts

    fonts._ACTIVE = 1.0
    if QApplication.instance() is not None and fonts._BASE is not None:
        QApplication.setFont(fonts._BASE)


def real_uimf_paths() -> list[str]:
    """Every `.uimf` this clone can see: PNNL's excerpts, plus `MAINSPRING_SMOKE_UIMF`.

    `tools/fetch_testdata.py` puts the excerpts in place; the environment variable names
    one more file, which is how a lab-side session points the suite at real acquisitions
    without any path to them living in this repository.
    """
    from mainspring import EXTERNAL_DIR

    paths = sorted(glob.glob(os.path.join(EXTERNAL_DIR, "pnnl-testdata", "*.uimf")))
    smoke = os.environ.get("MAINSPRING_SMOKE_UIMF")
    if smoke and os.path.isfile(smoke):
        paths.append(os.path.abspath(smoke))
    return paths


@pytest.fixture
def synthetic_uimf(tmp_path):
    """A two-frame synthetic UIMF file, and the `SyntheticFile` describing it.

    Small enough to write per test: a couple of dozen scans of a few hundred points.
    """
    return write_synthetic_uimf(tmp_path / "synthetic.uimf", frames=2, scans=16, bins=4096)


@pytest.fixture(
    params=real_uimf_paths() or [None],
    ids=lambda path: os.path.basename(path) if path else "no-real-file",
)
def real_uimf(request):
    """One real file per parametrisation, or a skip that says why there are none.

    The `None` placeholder is deliberate: an empty parameter list would make these tests
    vanish from the report, and a check that disappears when its input is missing is
    worse than one that says what it would have needed.
    """
    if request.param is None:
        pytest.skip(
            "no .uimf file here; run tools/fetch_testdata.py or set MAINSPRING_SMOKE_UIMF"
        )
    return request.param
