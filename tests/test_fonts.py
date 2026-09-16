"""Task 24: one text size over the whole application, and what it has to reach.

The half that Qt does -- `QApplication.setFont` reaching every widget -- is asserted once
and then trusted. Everything else here is about the half Qt does not do: a
`QGraphicsTextItem` ignores an application font change, an `AxisItem` caches what it drew,
and the space reserved for an axis is a number in a layout that no font ever touches. Each
of those is a separate way for a scale to be half-applied, and `fonts.sized` is the walk
that names them, so the negative cases below come before the coverage assertion that
rests on it.

The compounding test is the one that would go wrong silently: applying a scale to the
*current* font rather than to the base gives 200 per cent followed by 150 per cent as
300, which looks like a bug in whichever step the user took second.
"""

from __future__ import annotations

import pyqtgraph as pg
import pytest
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from mainspring.viewer import fonts, theme
from mainspring.viewer.heatmap import AXIS_HEIGHT, AXIS_WIDTH
from mainspring.viewer.main_window import MainWindow
from mainspring.viewer.settings import TEXT_SCALES, ViewerSettings


@pytest.fixture
def window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)
    return window


def _points(font: QFont) -> float:
    """A font's size in whichever unit it carries, so a test can compare two of them."""
    return font.pointSizeF() if font.pointSizeF() > 0 else float(font.pixelSize())


# --- the scale itself -------------------------------------------------------------------

def test_the_menu_offers_every_scale_with_the_stored_one_ticked(qtbot):
    window = MainWindow(ViewerSettings(text_scale=1.5))
    qtbot.addWidget(window)
    view_menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")
    menu = next(a.menu() for a in view_menu.actions() if a.text() == "Text size")

    actions = menu.actions()
    assert [a.text() for a in actions] == [f"{round(s * 100)}%" for s in TEXT_SCALES]
    assert [a.text() for a in actions if a.isChecked()] == ["150%"]


def test_a_scale_is_applied_to_the_base_font_and_never_compounds(window):
    base = _points(fonts._base())

    fonts.apply(window, 2.0)
    assert _points(QApplication.font()) == pytest.approx(2.0 * base)

    fonts.apply(window, 1.5)
    assert _points(QApplication.font()) == pytest.approx(1.5 * base)  # not 3.0

    fonts.apply(window, 1.0)
    assert _points(QApplication.font()) == pytest.approx(base)


def test_picking_a_size_applies_it_and_is_remembered(window):
    view_menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")
    menu = next(a.menu() for a in view_menu.actions() if a.text() == "Text size")
    bigger = next(a for a in menu.actions() if a.text() == "200%")

    bigger.trigger()

    assert window.settings.text_scale == 2.0
    assert fonts.active() == 2.0
    assert _points(QApplication.font()) == pytest.approx(2.0 * _points(fonts._base()))


# --- what Qt does not reach on its own ---------------------------------------------------

def test_the_tick_fonts_and_the_overlay_follow_the_scale(window):
    fonts.apply(window, 2.0)
    expected = 2.0 * _points(fonts._base())

    for axis in window.heatmap._axes():
        assert _points(axis.style["tickFont"]) == pytest.approx(expected)
    assert _points(window.heatmap._debug.textItem.font()) == pytest.approx(expected)


def test_the_axis_label_carries_the_size_and_the_palette_together(window):
    """The claim `theme.label_style` is written around: a palette change re-reads the
    scale and a scale change re-reads the palette, so neither puts the other's old value
    back."""
    fonts.apply(window, 1.5)
    theme.apply(window, "light")

    style = window.heatmap.plot_item.getAxis("left").labelStyle
    assert style["font-size"] == fonts.css_size()
    assert style["color"].lower() == theme.LIGHT.foreground

    fonts.apply(window, 2.0)

    style = window.heatmap.plot_item.getAxis("left").labelStyle
    assert style["font-size"] == fonts.css_size()
    assert style["color"].lower() == theme.LIGHT.foreground  # still light


def test_the_axis_extents_grow_on_both_sides_at_once(window, qtbot):
    """The heatmap and the projections are aligned by these two numbers, so a scale
    applied to one side alone is a layout bug rather than a smaller one."""
    fonts.apply(window, 2.0)
    window.heatmap.ci.layout.activate()
    qtbot.wait(20)

    assert window.heatmap.plot_item.getAxis("left").fixedWidth == 2 * AXIS_WIDTH
    assert window.heatmap.plot_item.getAxis("bottom").fixedHeight == 2 * AXIS_HEIGHT
    assert window.side_plots.x_plot.layout.columnMaximumWidth(0) == 2 * AXIS_WIDTH
    assert window.side_plots.y_plot.layout.rowMaximumHeight(3) == 2 * AXIS_HEIGHT


@pytest.mark.parametrize("scale", TEXT_SCALES)
def test_a_spin_box_still_shows_its_value_at_every_scale(qtbot, scale):
    """The style carves the up and down buttons out of whatever width the control ends
    up with, and they follow the font while Qt's own text allowance does not: at 200 per
    cent a plain `QSpinBox` gave its edit field five pixels and drew no number at all."""
    from mainspring.viewer.controls import SpinBox

    box = SpinBox()
    box.setRange(1, 5000)
    box.setValue(4321)
    qtbot.addWidget(box)
    box.setFont(fonts.scaled_font(scale))
    box.resize(box.sizeHint())
    box.show()

    assert box.lineEdit().width() >= box.fontMetrics().horizontalAdvance("4321")


def test_the_info_panels_floor_follows_the_scale(window):
    from mainspring.viewer.settings import INFO_PANEL_WIDTH

    fonts.apply(window, 2.0)

    assert window.info_panel._container.minimumWidth() == 2 * INFO_PANEL_WIDTH


# --- the walk ---------------------------------------------------------------------------

def test_a_tick_font_left_behind_is_named(window):
    """The negative case, at a non-default scale: at 100 per cent this walk passes for
    anything that simply never asked."""
    fonts.apply(window, 1.5)
    assert fonts.sized(window) == []

    window.heatmap.plot_item.getAxis("left").setTickFont(QFont(fonts._base()))

    assert any("tick font" in name for name in fonts.sized(window))


def test_a_stale_label_size_is_named(window):
    fonts.apply(window, 1.5)
    axis = window.heatmap.plot_item.getAxis("bottom")

    axis.setLabel("m/z", **{"color": "#969696", "font-size": "9.0pt"})

    assert any("label size" in name for name in fonts.sized(window))


def test_a_text_item_left_at_the_size_it_was_born_at_is_named(window):
    """A `QGraphicsTextItem` takes the application font when it is *constructed* and
    never hears about a later change, which is the whole reason
    `HeatmapView.set_text_scale` sets the overlay by hand. An item made at 100 per cent
    and not told is what this stands in for."""
    stray = pg.TextItem("stray")
    stray.setFont(QFont(fonts._base()))
    window.heatmap.scene().addItem(stray)
    try:
        fonts.apply(window, 1.5)
        assert any("TextItem" in name for name in fonts.sized(window))
    finally:
        window.heatmap.scene().removeItem(stray)


@pytest.mark.parametrize("scale", TEXT_SCALES)
def test_every_scale_reaches_every_piece_of_text_on_the_canvas(window, scale):
    fonts.apply(window, scale)
    assert fonts.sized(window) == []


def test_a_stored_scale_starts_applied(qtbot):
    window = MainWindow(ViewerSettings(text_scale=1.25))
    qtbot.addWidget(window)

    assert fonts.active() == 1.25
    assert fonts.sized(window) == []
    assert _points(QApplication.font()) == pytest.approx(1.25 * _points(fonts._base()))
