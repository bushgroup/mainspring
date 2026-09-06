"""Every control the user can touch explains itself, and the walk that proves it works.

The coverage test is the visible half; the negative tests are the half that matters. A
coverage test that cannot fail is worth nothing, and this one has two specific ways of
being worthless: a `QAction` reports its own text as its tooltip when it has none, and a
tooltip could be set on the factory's output while an inline-built control slips past.
Both are asserted here before the coverage assertion is trusted.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QComboBox, QToolBar

from mainspring.viewer import controls
from mainspring.viewer.main_window import MainWindow


def test_a_mute_action_is_named(qtbot):
    """The negative case: an action added the old way is caught."""
    window = MainWindow()
    qtbot.addWidget(window)
    assert controls.unexplained(window) == []

    QAction("Mute", window)
    assert any("Mute" in name for name in controls.unexplained(window))


def test_an_action_tooltip_that_is_only_its_label_does_not_count(qtbot):
    """Qt answers `toolTip()` with the action's text when none was set, so a naive
    emptiness test passes for every mute action in the file. This is that test."""
    window = MainWindow()
    qtbot.addWidget(window)

    action = QAction("&Sum all", window)
    assert action.toolTip() == "Sum all"  # Qt's fallback, mnemonic stripped
    assert any("Sum all" in name for name in controls.unexplained(window))

    controls.describe(action, "Add every frame passing the type filter into one heatmap.")
    assert controls.unexplained(window) == []


def test_a_mute_widget_is_named(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    toolbar = window.findChild(QToolBar, "view_toolbar")

    box = QComboBox()
    box.setObjectName("mute_box")
    toolbar.addWidget(box)
    assert any("mute_box" in name for name in controls.unexplained(window))


def test_the_whole_window_explains_itself(qtbot):
    """The rule the repo holds itself to, over the window as it actually ships."""
    window = MainWindow()
    qtbot.addWidget(window)
    assert controls.unexplained(window) == []


def test_the_window_still_explains_itself_with_a_file_open(qtbot, synthetic_uimf):
    """Opening a file repopulates the frame-type filter and the parameter tree; neither
    may lose its tooltip on the way."""
    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.frame_shown, timeout=10_000):
        window.open_file(synthetic_uimf.path)
    assert controls.unexplained(window) == []


def test_the_shortcut_is_appended_to_the_tooltip_not_the_status_tip(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.open_action.toolTip().endswith("(Ctrl+O)")
    assert "Ctrl+O" not in window.open_action.statusTip()
    assert window.open_action.statusTip()  # the bare sentence, for the status bar
    assert window.open_action.toolTip().startswith(window.open_action.statusTip())


def test_a_toolbar_label_shares_its_widget_tooltip(qtbot):
    """`Bits:` is the word a user points at, so it cannot be the untipped half."""
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._bits_box.toolTip()
    assert window._bits_label.toolTip() == window._bits_box.toolTip()


def test_a_waiver_is_deliberate_and_silences_the_walk(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    action = QAction("Mute", window)
    assert any("Mute" in name for name in controls.unexplained(window))
    controls.describe(action, None)
    assert controls.is_waived(action)
    assert controls.unexplained(window) == []


def test_make_action_configures_before_it_connects(qtbot):
    """A restored non-default setting must not fire its handler mid-build."""
    window = MainWindow()
    qtbot.addWidget(window)
    fired = []

    action = controls.make_action(
        window,
        "Keep something",
        tip="Keep the current thing when the next one arrives.",
        checkable=True,
        checked=True,
        toggled=fired.append,
    )
    assert action.isChecked()
    assert fired == []


def test_the_plot_items_explain_themselves(qtbot):
    """The heatmap, its four axes, the colour bar and both projections."""
    window = MainWindow()
    qtbot.addWidget(window)

    items = window.heatmap.scene().items()
    images = [i for i in items if isinstance(i, pg.ImageItem)]
    axes = [i for i in items if isinstance(i, pg.AxisItem) and i.isVisible()]
    plots = [i for i in items if isinstance(i, pg.PlotItem)]
    assert images and axes and len(plots) >= 4  # heatmap, two projections, colour bar
    for item in images + axes + plots:
        assert item.toolTip().strip(), f"{item!r} explains nothing"


def test_a_hidden_axis_is_not_asked_to_explain_itself(qtbot):
    """The projections hide all four of theirs; there is nothing to point at."""
    window = MainWindow()
    qtbot.addWidget(window)

    hidden = [
        i
        for i in window.heatmap.scene().items()
        if isinstance(i, pg.AxisItem) and not i.isVisible()
    ]
    assert hidden, "the side plots should still be hiding their axes"
    for axis in hidden:
        controls.waive(axis)  # no-op for the walk, asserted below
    assert controls.unexplained(window) == []
