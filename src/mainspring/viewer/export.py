"""`File > Export`: the display as it stands, at a chosen resolution, without the colour bar.

Everything on screen is one `QGraphicsScene`, so an export is one
`QGraphicsScene.render` into whatever paint device the format asks for -- a `QImage` for
PNG, a `QPdfWriter` for PDF. There is no second drawing path to keep in step with the
first: what is exported is the scene the user is looking at, rendered again.

**Leaving the colour bar out is a matter of which rectangle is rendered**, not of taking
anything apart. The bar is a `PlotItem` of its own in the layout's last column
(`heatmap.py`), so the heatmap and its two projections are a contiguous rectangle to the
left of it; `content_rect` unites their three scene rects and the render is clipped to
that. Detaching and re-attaching the bar around a render would disturb the levels wiring
it owns, for a rectangle we can simply not draw.

**The image is re-rasterised for the export, and that is the whole of the work here.**
The `ImageItem` holds an array the size of the *viewport* -- that is the point of the
sparse frame, and it is why a gesture repaints at all (`uimf/raster.py`,
`workers.RenderRequest`). Rendering that scene at 3x would give crisp vector axes around
a heatmap upscaled 3x, which is the one artefact a reader of the figure would notice. So
the export asks `rasterise` for the same window at the export's pixel size, shows it
through `HeatmapView.substituted` for as long as the painter needs it, and puts the
screen's array back.

Past one output pixel per source element there is nothing more to get: `rasterise` caps
its cell count at the number of elements in view, so a zoomed-in export returns the same
image at every resolution and stretches it, correctly, rather than inventing detail.

**Re-rasterising moves the numbers the colour levels are expressed in**, which is why
`export_levels` exists. A `sum` pixel covering a quarter of the area holds about a
quarter of the intensity, so the screen's levels would render the export nearly black.
The rule is one line and has no special cases: the levels sit at some fraction of the
screen image's own range, and the export puts them at the same fraction of *its* range.
Levels left to auto-scale are the fractions 0 and 1, so they come out auto-scaled again;
levels the user pinned to bring up a faint feature keep the contrast that pinning chose.

**`BASE_DPI` is the resolution the on-screen layout is taken to be actual size at.** A
figure exported at 96 dpi is the window's own pixels; at 300 it is the same figure with
3.125 times as many. Qt resolves a point size against the paint device and the painter's
transform scales it again, so the two formats reach the same result by different routes
and neither may double-count:

* PNG scales the painter, and the `QImage`'s logical DPI is the screen's, so text comes
  out the same fraction of the figure that it is on screen. The file's DPI metadata is
  stamped *after* the render, since setting it first would change that conversion.
* PDF does not scale the painter at all. `QPdfWriter` is set to `BASE_DPI` so that one
  scene unit is one 96th of an inch, which puts text at its true point size and the
  page at the figure's true size; the requested resolution reaches the file only as the
  sample count of the one embedded image, which is exactly what "dpi" means for it.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from PySide6.QtCore import QMarginsF, QRectF, QSizeF, Qt
from PySide6.QtGui import QImage, QPageLayout, QPageSize, QPainter, QPdfWriter
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from ..uimf.raster import RasterResult, rasterise
from .controls import describe
from .heatmap import scaled
from .settings import EXPORT_DPIS

__all__ = [
    "BASE_DPI",
    "FORMATS",
    "MAX_PIXELS",
    "ExportDialog",
    "content_rect",
    "export_display",
    "export_levels",
    "export_pixels",
]

BASE_DPI = 96.0
"""The resolution the on-screen layout is taken to be actual size at, so that a figure's
physical size is its pixel size divided by this. 96 rather than a number read off the
screen: it is what Qt itself assumes wherever no paint device has said otherwise, so
matching it is what keeps a point size meaning the same thing in a `QImage`, in a PDF
page and on screen. An export at 96 dpi is the window's own pixels, one for one."""

MAX_PIXELS = 50_000_000
"""The largest figure that will be written, in pixels. Not arbitrary: the rasteriser
allocates a float64 accumulator over the output grid before it narrows to float32, so
the pixel count is paid for several times over, and a maximised window on a 4K display
asks for 300 megapixels at 600 dpi without the user doing anything unusual. The dialog
does the arithmetic in front of them and refuses before anything is allocated."""

FORMATS = ("png", "pdf")
"""What `export_display` can write. The menu carries one entry per format rather than a
format chooser in the dialog, because "Export PNG" is what a user goes to the File menu
looking for."""


# --- the geometry ---------------------------------------------------------------------

def content_rect(heatmap: object, side_plots: object) -> QRectF:
    """The scene rectangle holding the heatmap and its two projections, and nothing else.

    The union of three items rather than the scene's own rectangle minus a column: it
    stays right if the layout ever gains a fourth cell, and it crops to what is drawn
    rather than to what the widget happens to be sized to.

    **The axis labels are added by hand because they hang outside the plot that owns
    them.** `AxisItem` reserves space for its ticks and its values, and then draws its
    label in a text item that overhangs that by a few pixels; a `PlotItem`'s bounding
    rect is its layout geometry, so `m/z` and `Arrival time (ms)` fall a little outside
    it and a crop to the three plots alone shaves the outer edge off both. The scene's
    own `childrenBoundingRect` would have caught them and also the image item, which
    extends well past the view box that clips it whenever the user is zoomed in.
    """
    rect = QRectF()
    for plot in (heatmap.plot_item, side_plots.x_plot, side_plots.y_plot):
        rect = rect.united(plot.mapRectToScene(plot.boundingRect()))
        for name in ("left", "bottom", "top", "right"):
            label = plot.getAxis(name).label
            if label.isVisible():
                rect = rect.united(label.mapRectToScene(label.boundingRect()))
    return rect


def export_pixels(rect: QRectF, dpi: int) -> "tuple[int, int]":
    """`(width, height)` in pixels of `rect` exported at `dpi`."""
    scale = float(dpi) / BASE_DPI
    return max(1, round(rect.width() * scale)), max(1, round(rect.height() * scale))


# --- the levels -----------------------------------------------------------------------

def export_levels(
    on_screen: np.ndarray, exported: np.ndarray, levels: "tuple[float, float]"
) -> "tuple[float, float]":
    """Where `levels` fall on `exported`'s range, given where they fall on `on_screen`'s.

    Both images are already in display space (`heatmap.scaled`), which is where a level
    lives; interpolating in the intensities' own space would be wrong under the log and
    sqrt colour scales, where the transform is not linear and a fraction of the range is
    not a fraction of the intensity.

    A degenerate range at either end falls back to the export's own full range, which is
    what `HeatmapView.set_image` does with an image that is all one value.
    """
    if exported.size == 0:
        return 0.0, 1.0
    export_low, export_high = float(exported.min()), float(exported.max())
    if not export_high > export_low:
        export_high = export_low + 1.0
    if on_screen.size == 0:
        return export_low, export_high
    screen_low, screen_high = float(on_screen.min()), float(on_screen.max())
    if not screen_high > screen_low:
        return export_low, export_high
    span = (export_high - export_low) / (screen_high - screen_low)
    low = export_low + (float(levels[0]) - screen_low) * span
    high = export_low + (float(levels[1]) - screen_low) * span
    return low, (high if high > low else low + 1.0)


# --- writing it out -------------------------------------------------------------------

def export_display(
    heatmap: object,
    side_plots: object,
    frame: object,
    result: RasterResult,
    colour_scale: str,
    path: str,
    fmt: str,
    dpi: int,
) -> "tuple[int, int]":
    """Write the display to `path` at `dpi`, and answer with its pixel size.

    `result` is what is on screen -- it carries the window, the axes and the aggregate
    the export has to reproduce, so nothing about "which picture is this" has to be
    passed separately or could disagree with what the user is looking at.

    Raises `ValueError` for an unknown format or a figure over `MAX_PIXELS`, and `OSError`
    if the file cannot be written. Nothing is allocated before those checks.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; one of {FORMATS}")
    rect = content_rect(heatmap, side_plots)
    width, height = export_pixels(rect, dpi)
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"{width} x {height} is {width * height / 1e6:.0f} megapixels, over the "
            f"{MAX_PIXELS / 1e6:.0f} this will write; lower the resolution or make the "
            "window smaller"
        )

    # The viewport's size scaled, not the figure's: that is what the on-screen render is
    # asked for (`HeatmapView.pixel_size`), so scaling it is what makes the export's
    # sample density exactly `scale` times the screen's rather than approximately.
    scale = float(dpi) / BASE_DPI
    viewport_width, viewport_height = heatmap.pixel_size()
    exported = rasterise(
        frame,
        result.axes,
        result.x_range,
        result.y_range,
        max(1, round(viewport_width * scale)),
        max(1, round(viewport_height * scale)),
        aggregate=result.aggregate,
    )
    displayed = scaled(exported.image, colour_scale)
    levels = export_levels(scaled(result.image, colour_scale), displayed, heatmap.levels())

    x0, x1 = result.x_range
    y0, y1 = result.y_range
    with heatmap.substituted(displayed, (x0, y0, x1 - x0, y1 - y0), levels):
        if fmt == "pdf":
            _write_pdf(heatmap, rect, path)
        else:
            _write_png(heatmap, rect, (width, height), path, dpi)
    return width, height


def _write_png(
    view: object, source: QRectF, size: "tuple[int, int]", path: str, dpi: int
) -> None:
    width, height = size
    image = QImage(width, height, QImage.Format.Format_RGB32)
    if image.isNull():
        raise OSError(f"could not allocate a {width} x {height} image")
    image.fill(view.backgroundBrush().color())
    with _painter_on(image) as painter:
        _render(view, painter, QRectF(0, 0, width, height), source)
    # After the render, never before: Qt converts a font's point size against the paint
    # device's logical DPI, so an image that already claimed 300 would scale the text a
    # second time on top of the painter's own scaling.
    dots_per_metre = round(float(dpi) / 0.0254)
    image.setDotsPerMeterX(dots_per_metre)
    image.setDotsPerMeterY(dots_per_metre)
    if not image.save(path, "PNG"):
        raise OSError(f"could not write {path}")


def _write_pdf(view: object, source: QRectF, path: str) -> None:
    writer = QPdfWriter(path)
    writer.setCreator("mainspring")
    # Set before the painter exists: `QPdfWriter` fixes its page when one is constructed
    # on it. Resolution `BASE_DPI` makes one device unit one scene unit, so the render
    # below is 1:1 and nothing is scaled twice -- the module docstring says why.
    writer.setResolution(int(BASE_DPI))
    writer.setPageSize(QPageSize(
        QSizeF(source.width() / BASE_DPI * 72.0, source.height() / BASE_DPI * 72.0),
        QPageSize.Unit.Point,
        "mainspring",
        QPageSize.SizeMatchPolicy.ExactMatch,
    ))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Point)
    with _painter_on(writer) as painter:
        target = QRectF(0, 0, writer.width(), writer.height())
        painter.fillRect(target, view.backgroundBrush())
        _render(view, painter, target, source)


@contextmanager
def _painter_on(device: object):
    """A `QPainter` that is ended even if the render raises.

    Qt warns, and on some platforms writes nothing, if a painter is still active when
    its device is destroyed, so this is not tidiness -- a raise partway through a render
    would otherwise cost the file as well.
    """
    painter = QPainter(device)
    if not painter.isActive():
        raise OSError("could not open a painter on the export")
    try:
        yield painter
    finally:
        painter.end()


def _render(view: object, painter: QPainter, target: QRectF, source: QRectF) -> None:
    """Draw `source` of the view's scene into `target`.

    Antialiasing through render hints rather than through `pg.setConfigOptions`: the
    interactive path turns it off for a heatmap redrawn on every wheel tick (`app.py`),
    and the items that read that option read it once, when they are constructed.

    `IgnoreAspectRatio` with a target computed from the source: the two agree to within
    the pixel that rounding to whole pixels costs, and letting Qt letterbox that pixel
    would leave an unpainted strip down one edge of the figure.
    """
    view.scene().prepareForPaint()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    view.scene().render(painter, target, source, Qt.AspectRatioMode.IgnoreAspectRatio)


# --- the dialog -----------------------------------------------------------------------

class ExportDialog(QDialog):
    """Choose the resolution, having seen what it will cost in pixels and in inches.

    A resolution is only meaningful next to the size it produces -- 600 dpi is the right
    answer for a two-column figure and the wrong one for a maximised window -- so the
    size is on screen and changes as the choice does, and a figure over `MAX_PIXELS`
    disables Save with the reason showing rather than failing after the file dialog has
    already been answered.

    Fixed choices rather than a free number: 96 is the window's own pixels, 300 and 600
    are what a journal asks for, and 150 is a slide. A spin box would offer a thousand
    values that only differ from these in the file size.
    """

    def __init__(
        self, parent: object, fmt: str, dpi: int, rect: QRectF
    ) -> None:
        super().__init__(parent)
        self._fmt = fmt
        self._rect = rect
        self.setWindowTitle(f"Export {fmt.upper()}")
        self.setModal(True)

        self._dpi_box = QComboBox()
        for preset in EXPORT_DPIS:
            self._dpi_box.addItem(f"{preset} dpi", preset)
        index = self._dpi_box.findData(dpi)
        self._dpi_box.setCurrentIndex(index if index >= 0 else self._dpi_box.count() - 2)
        describe(self._dpi_box, "How finely the figure is drawn, and the heatmap resampled.")

        self._size_label = QLabel()
        self._size_label.setWordWrap(True)
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Resolution:", self._dpi_box)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._size_label)
        note = QLabel("The colour bar is not part of the exported figure.")
        note.setEnabled(False)  # a footnote, not a control
        layout.addWidget(note)
        layout.addWidget(self._buttons)

        # Connected after the restored choice is in place, the order every control in
        # this viewer is built in (`controls.make_action` says why), and then called by
        # hand so the label describes that choice rather than staying empty until the
        # user changes it.
        self._dpi_box.currentIndexChanged.connect(lambda _: self._describe_size())
        self._describe_size()

    def dpi(self) -> int:
        """The chosen resolution."""
        return int(self._dpi_box.currentData())

    def _describe_size(self) -> None:
        dpi = self.dpi()
        width, height = export_pixels(self._rect, dpi)
        inches = f"{self._rect.width() / BASE_DPI:.1f} x {self._rect.height() / BASE_DPI:.1f} in"
        too_big = width * height > MAX_PIXELS
        if self._fmt == "pdf":
            # The page is the same size at every resolution; what the choice buys a PDF
            # is the sample count of the heatmap it embeds, and saying so is the only
            # way "600 dpi" means anything to someone looking at a fixed page size.
            self._size_label.setText(f"{inches} page, heatmap at {dpi} dpi")
        else:
            self._size_label.setText(f"{width:,} x {height:,} pixels, {inches}")
        if too_big:
            self._size_label.setText(
                self._size_label.text()
                + f"\nOver the {MAX_PIXELS / 1e6:.0f} megapixels this can write."
                " Choose a lower resolution, or make the window smaller."
            )
        self._buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(not too_big)
