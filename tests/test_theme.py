"""Light mode: what the toggle changes, what it must not change, and the walk.

The coverage assertion is `test_nothing_on_the_canvas_is_painted_outside_its_palette`,
and on its own it is worth very little: run under the dark palette it passes for every
colour hardcoded to the value the viewer has always drawn, which is most of what the
walk exists to catch. So it is run under both palettes, and the negative tests below put
each of the four colours the tree carried before this feature back in place and assert
the walk names it. Those are what make the coverage assertion mean something.

The other half is that a toggle is *free*: the open file, the frame, the view ranges and
the pinned colour levels all survive it, because the palette is set on the items that
are already on screen rather than by rebuilding them.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from mainspring.viewer import theme
from mainspring.viewer.main_window import MainWindow
from mainspring.viewer.settings import THEMES, ViewerSettings, load_settings, save_settings


def _light(window: MainWindow) -> None:
    window._light_action.setChecked(True)


def test_every_theme_name_has_a_palette():
    """`settings.THEMES` is what a hand-edited setting is clamped against and
    `theme.PALETTES` is what gets painted; a name in one and not the other is a theme
    that validates and then falls back to dark without saying so."""
    assert set(THEMES) == set(theme.PALETTES)
    assert all(name == theme.PALETTES[name].name for name in THEMES)


def test_dark_is_the_default_and_is_what_the_viewer_always_drew():
    """An existing user sees no change: the defaults are pyqtgraph's own `k` and `d`,
    the pale curve blue and the debug overlay's near-white, spelled out."""
    assert ViewerSettings().theme == "dark"
    assert theme.DARK.background == "#000000"
    assert theme.DARK.foreground == "#969696"
    assert theme.DARK.curve == "#bed2ff"
    assert theme.DARK.debug == "#dcdcdc"


def test_a_hand_edited_theme_is_clamped():
    assert ViewerSettings(theme="solarized").validate().theme == "dark"
    assert ViewerSettings(theme="light").validate().theme == "light"


def test_the_setting_round_trips():
    settings = ViewerSettings(theme="light")
    save_settings(settings)
    assert load_settings().theme == "light"


def test_a_new_window_starts_dark(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.settings.theme == "dark"
    assert not window._light_action.isChecked()
    assert window.heatmap.backgroundBrush().color().name() == theme.DARK.background


def test_the_menu_entry_paints_the_canvas_and_writes_the_setting(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    _light(window)
    assert window.settings.theme == "light"
    assert window.heatmap.backgroundBrush().color().name() == theme.LIGHT.background

    window._light_action.setChecked(False)
    assert window.settings.theme == "dark"
    assert window.heatmap.backgroundBrush().color().name() == theme.DARK.background


def test_a_restored_setting_opens_the_window_light(qtbot):
    save_settings(ViewerSettings(theme="light"))
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._light_action.isChecked()
    assert window.heatmap.backgroundBrush().color().name() == theme.LIGHT.background
    assert theme.themed(window) == []


def test_nothing_on_the_canvas_is_painted_outside_its_palette(qtbot):
    """The rule, over the window as it actually ships, under both palettes. The light
    one is the half that can fail."""
    window = MainWindow()
    qtbot.addWidget(window)
    for name in THEMES:
        theme.apply(window, name)
        assert theme.themed(window) == [], name


def test_the_walk_still_holds_with_a_file_open(qtbot, synthetic_uimf):
    """Opening a file sets both axis labels and both projections' data, which is the
    first moment anything on the canvas has a colour it did not start with."""
    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.frame_shown, timeout=20000):
        window.open_file(synthetic_uimf.path)
    for name in THEMES:
        theme.apply(window, name)
        assert theme.themed(window) == [], name


def test_the_curve_colour_the_viewer_used_to_hardcode_is_named(qtbot):
    """`side_plots._PEN`, a pale blue chosen for a black background and close to
    unreadable on white. The first thing the walk was written to find."""
    window = MainWindow()
    qtbot.addWidget(window)
    theme.apply(window, "light")

    window.side_plots.curves[0].setPen(pg.mkPen(color=(190, 210, 255), width=1))
    assert any("curve pen" in name for name in theme.themed(window))


def test_the_debug_overlay_is_walked_even_though_it_is_hidden(qtbot):
    """It is drawn only under `MAINSPRING_DEBUG_RENDER`, so a walk that skipped what is
    not currently on screen would never look at it."""
    window = MainWindow()
    qtbot.addWidget(window)
    theme.apply(window, "light")
    assert not window.heatmap._debug.isVisible()

    window.heatmap._debug.setColor(pg.mkColor(220, 220, 220))
    assert any("text colour" in name for name in theme.themed(window))


def test_a_tick_pen_left_at_the_dark_foreground_is_named(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    theme.apply(window, "light")

    window.heatmap.plot_item.getAxis("left").setTickPen(pg.mkPen("#969696", width=1.5))
    assert any("tick pen" in name for name in theme.themed(window))


def test_an_axis_label_that_lost_its_colour_is_named(qtbot):
    """The invisible-label bug, in the direction light mode would produce it: a label
    set with `font-weight` alone falls back to Qt's rich-text black."""
    window = MainWindow()
    qtbot.addWidget(window)
    theme.apply(window, "light")

    window.heatmap.plot_item.setLabel("bottom", "m/z", color="#000000", **{"font-weight": "bold"})
    assert any("label colour" in name for name in theme.themed(window))


def test_two_stray_items_of_one_type_are_both_named(qtbot):
    """Naming one of them would look like one problem, and fixing it like fixing both."""
    window = MainWindow()
    qtbot.addWidget(window)
    theme.apply(window, "light")

    for curve in window.side_plots.curves:
        curve.setPen(pg.mkPen("#ff0000"))
    named = [name for name in theme.themed(window) if "curve pen" in name]
    assert len(named) == 2
    assert named[0] != named[1]


def test_the_colour_map_is_not_part_of_the_theme(qtbot, synthetic_uimf):
    """The map is a statement about the data, the palette one about the furniture. A
    toggle must not move the user's choice, and the walk must not ask the image or the
    colour bar's gradient to be palette members."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._on_colour_map_changed("magma", True)
    with qtbot.waitSignal(window.frame_shown, timeout=20000):
        window.open_file(synthetic_uimf.path)

    _light(window)
    assert window.settings.colour_map == "magma"
    assert theme.themed(window) == []


def test_the_axis_labels_are_repainted_when_the_theme_changes(qtbot, synthetic_uimf):
    """Set once by `set_image` under one palette, and they have to follow the other --
    this is the failure the walk's `label colour` entry exists for."""
    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.frame_shown, timeout=20000):
        window.open_file(synthetic_uimf.path)

    axis = window.heatmap.plot_item.getAxis("bottom")
    assert axis.labelText  # `set_image` has been through
    assert axis.labelStyle["color"] == theme.DARK.foreground

    _light(window)
    assert axis.labelStyle["color"] == theme.LIGHT.foreground
    assert axis.labelStyle["font-weight"] == "bold"  # and it is still bold
    assert axis.labelText


def test_a_frame_drawn_after_the_toggle_keeps_the_light_label_colour(qtbot, synthetic_uimf):
    """The other order: the theme changes first, then a render sets the labels again.
    A module-level style constant would put the dark colour back here."""
    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.frame_shown, timeout=20000):
        window.open_file(synthetic_uimf.path)
    _light(window)

    with qtbot.waitSignal(window.frame_shown, timeout=20000):
        window.heatmap.view_box.setRange(
            xRange=window.heatmap.view_range()[0], yRange=(0.0, 1.0), padding=0.0
        )
    assert window.heatmap.plot_item.getAxis("bottom").labelStyle["color"] == theme.LIGHT.foreground


def test_toggling_keeps_the_file_the_frame_the_ranges_and_the_levels(qtbot, synthetic_uimf):
    """A repaint, not a rebuild. Nothing the user set up costs anything to change theme."""
    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.frame_shown, timeout=20000):
        window.open_file(synthetic_uimf.path)

    (x0, x1), (y0, y1) = window.heatmap.view_range()
    zoom = (x0 + 0.25 * (x1 - x0), x0 + 0.5 * (x1 - x0))
    with qtbot.waitSignal(window.frame_shown, timeout=20000):
        window.heatmap.view_box.setRange(xRange=zoom, yRange=(y0, y1), padding=0.0)
    window.heatmap.set_levels(3.0, 44.0)

    path, frame, image = window._path, window._current_frame_number, window.heatmap.image_item.image
    ranges = window.heatmap.view_range()

    _light(window)

    assert window._path == path
    assert window._current_frame_number == frame
    assert window.heatmap.levels() == (3.0, 44.0)
    assert window.heatmap.view_range() == ranges
    assert np.array_equal(window.heatmap.image_item.image, image)


def test_the_auto_range_button_is_gone(qtbot):
    """pyqtgraph's own "A" button enables auto-range, which the heatmap forbids, and it
    is a white pixmap no palette can reach. Hidden rather than themed."""
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.heatmap.plot_item.buttonsHidden
