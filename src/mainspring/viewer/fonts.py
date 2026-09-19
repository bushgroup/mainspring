"""One text size for the whole application, the path that sets it, and the walk that
proves nothing bypassed it.

The viewer's text was sized for a 1000x700 window at 96 dpi and nothing followed when
the window or the figure got bigger. On a 4K panel across a bench that is a readout
nobody can read from where they are standing, and it is the same complaint as an
exported figure whose type is a magnified screenshot (lab record, task 24).
`View > Text size` answers it with one scale over everything: the menus, the toolbar,
the info panel, the axis labels and the tick values all move together, in four steps
from 100 to 200 per cent, persisted like every other preference.

This module mirrors `theme.py`'s shape on purpose -- the one path that sets the thing
(`apply`) and the walk that proves nothing was missed (`sized`) -- and is deliberately
**not** folded into it. `theme.py`'s first paragraph promises it touches the plot canvas
only and leaves the Qt chrome to the operating system; this reaches the whole
application, which is a different promise. A text *size* is not a `QPalette`, so the
non-goal that keeps the chrome's colors Windows's does not keep its type at 100 per
cent.

Three things have to happen for a scale to take effect, and only the first is Qt's:

1. `QApplication.setFont` reaches every widget that has not been given a font of its
   own, which is every widget here. That covers the menus, the toolbar, the status bar,
   the info panel and the dialogs, and it is why none of them appears below.
2. **A `QGraphicsTextItem` does not follow an application font change.** The axis labels
   and the render-time overlay live in the plot scene and have to be told, which is what
   `HeatmapView.set_text_scale` is for. Tick values go through `AxisItem.setTickFont`,
   which also drops the cached `QPicture` an axis draws from -- and that cache is what an
   export would otherwise replay at the old size.
3. The space reserved around the plots does not follow the type in it. `AXIS_WIDTH` and
   `AXIS_HEIGHT` stay the numbers they are at 100 per cent and are multiplied by
   `extent()` at the moment they are applied, on the heatmap **and** on the projections'
   layout grids, since those two numbers are what aligns a linked plot in pixels and a
   scale applied to one side alone would pull them apart.

**Every scale is applied to the base font, never to the current one.** `_base()` captures
`QApplication.font()` once, the first time anything here asks, before `apply` has ever
written to it -- otherwise 200 per cent followed by 150 per cent would compound to 300.
The base may carry a point size or a pixel size depending on the platform and the style,
so `scaled_font` branches on which, rather than assuming the point size that is merely
usual on Windows.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from .controls import name_of, numbered

__all__ = ["active", "apply", "css_size", "extent", "scaled_font", "sized"]

_ACTIVE = 1.0
"""The scale `apply` last applied. Module state for the same reason `theme._ACTIVE` is:
there is one application font per process, and a second idea of what it is would only
ever disagree with the first."""

_BASE: "QFont | None" = None
"""The application font as it was before this module ever touched it, captured lazily so
that importing this module costs nothing and so that the capture happens after Qt has
finished choosing a default for the platform."""


def _base() -> QFont:
    """The font every scale is applied to. Captured once, on first use."""
    global _BASE
    if _BASE is None:
        _BASE = QFont(QApplication.font())
    return _BASE


def active() -> float:
    """The text scale in force, as a multiplier."""
    return _ACTIVE


def scaled_font(scale: "float | None" = None) -> QFont:
    """The base font at `scale`, or at the active scale.

    A `QFont` carries either a point size or a pixel size, never both, and asking for
    the one it does not have answers -1. Which it is depends on the platform and the
    style, so both are handled rather than the usual one.
    """
    font = QFont(_base())
    factor = _ACTIVE if scale is None else float(scale)
    if font.pointSizeF() > 0:
        font.setPointSizeF(font.pointSizeF() * factor)
    else:
        font.setPixelSize(max(1, round(font.pixelSize() * factor)))
    return font


def css_size() -> str:
    """The active text size as a CSS `font-size`, for an axis label's style dict.

    An axis label is HTML inside a `<span>` whose style comes from `AxisItem.labelStyle`
    (`labelString`), so this is the only way to size one. `theme.label_style` is what
    puts it there, which is what makes a palette change re-read the scale and a scale
    change re-read the palette.
    """
    font = scaled_font()
    if font.pointSizeF() > 0:
        return f"{font.pointSizeF():.1f}pt"
    return f"{font.pixelSize()}px"


def extent() -> float:
    """The multiplier for the space an axis reserves for its type.

    The scale itself. An axis's width is mostly its tick values, so it has to grow with
    them or the values are clipped; the margins and the label inside that width grow a
    little more than they need to, which costs a few pixels of plot area and is invisible
    beside type that does not fit.
    """
    return _ACTIVE


def apply(window: object, scale: float) -> float:
    """Set the whole application's text size, live, and return the scale used.

    The one path a text-size change takes, and the order is load-bearing. The
    application font goes first, because that is what every widget in the window reads
    and because the plot layer's own sizing is measured against it; then the three
    owners of text Qt will not reach are told in turn.

    Nothing is rebuilt, so the open file, the frame, the view ranges and the color
    levels all survive a change untouched, exactly as they survive a theme toggle.
    """
    global _ACTIVE
    _base()  # captured before the first write, or every later scale compounds
    _ACTIVE = float(scale)
    QApplication.setFont(scaled_font())
    window.heatmap.set_text_scale(_ACTIVE)
    window.side_plots.set_text_scale(_ACTIVE)
    window.info_panel.set_text_scale(_ACTIVE)
    window.chromatogram.set_text_scale(_ACTIVE)
    return _ACTIVE


def sized(window: object) -> "list[str]":
    """Names of everything on the plot canvas not drawn at the active text size.

    The analogue of `theme.themed`, and confined to the plot canvas for the same reason
    that walk is: a widget's font is Qt's to propagate and `QApplication.setFont` is one
    call that cannot half-succeed, while every piece of text in the `QGraphicsScene` is
    something this module had to remember to set. Those are the ones worth walking.

    Hidden axes are walked too. The projections hide all eight of theirs and a size that
    is only wrong while it cannot be seen is still wrong: an axis shown later, or an
    export, would draw it.

    **Over every canvas `window.plot_canvases()` names**, and not `window.heatmap`
    alone, for the reason `theme.themed` walks the same list: a second plot widget in a
    dock of its own would otherwise be outside both checks, silently (lab record,
    task 31).
    """
    expected = scaled_font()
    wrong: list[str] = []
    for view in window.plot_canvases():
        for item in view.scene().items():
            if isinstance(item, pg.AxisItem):
                tick_font = item.style.get("tickFont")
                if tick_font is None or not _same_size(tick_font, expected):
                    wrong.append(f"{name_of(item)} tick font")
                label_size = item.labelStyle.get("font-size")
                if label_size is not None and label_size != css_size():
                    wrong.append(f"{name_of(item)} label size ({label_size})")
            elif isinstance(item, pg.TextItem):
                if not _same_size(item.textItem.font(), expected):
                    wrong.append(f"{name_of(item)} font")
    return numbered(wrong)


def _same_size(font: QFont, expected: QFont) -> bool:
    """Whether `font` is the expected size, in whichever unit it carries it."""
    if expected.pointSizeF() > 0:
        return abs(font.pointSizeF() - expected.pointSizeF()) < 0.01
    return font.pixelSize() == expected.pixelSize()
