"""What `File > Export` writes, and the two things about it that are easy to get wrong.

The colour bar is left out by cropping rather than by taking anything apart, so the crop
is what is tested: not only that the bar is outside it, but that nothing else is --
`m/z` and `Arrival time (ms)` hang a few pixels outside the plot item that owns them,
and a crop to the plots alone shaved the outer edge off both.

The figure is the scene as it stands, so what is tested about the heatmap is that the
export does not touch it: the same array, under the same levels, before and after. Task 24
deleted the re-rasterise that used to stand a finer image in for the screen's, which moved
the colour levels and made the figure a different picture from the one the user exported.

Everything here runs on the synthetic fixture, offscreen, with no data file. The window
paints boxes for glyphs under the offscreen platform, which is why nothing below looks
at pixels: what is asserted is geometry, sizes and the state the window is left in.
"""

from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QGraphicsTextItem

from mainspring.viewer import theme
from mainspring.viewer.export import (
    BASE_DPI,
    ExportDialog,
    content_rect,
    export_display,
    export_pixels,
)
from mainspring.viewer.main_window import MainWindow


@pytest.fixture
def viewer(qtbot, synthetic_uimf):
    """A window showing the synthetic file's first frame, laid out and painted once.

    `show()` and not a bare `resize()`: the export crops to where the items actually
    are, and a window that has never been laid out puts them at a default size that has
    nothing to do with the one under test.
    """
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1000, 700)
    window.show()
    with qtbot.waitSignal(window.frame_shown, timeout=20000):
        window.open_file(synthetic_uimf.path)
    return window


def _bar_of(window: MainWindow) -> pg.ColorBarItem:
    """The colour bar, found by type rather than by reaching for a private attribute."""
    bars = [
        item for item in window.heatmap.scene().items()
        if isinstance(item, pg.ColorBarItem)
    ]
    assert len(bars) == 1
    return bars[0]


# --- the crop ---------------------------------------------------------------------------

def test_the_crop_excludes_the_colour_bar(viewer):
    bar = _bar_of(viewer)
    rect = content_rect(viewer.heatmap, viewer.side_plots)

    assert not rect.intersects(bar.mapRectToScene(bar.boundingRect()))
    for plot in (viewer.heatmap.plot_item, viewer.side_plots.x_plot, viewer.side_plots.y_plot):
        assert rect.contains(plot.mapRectToScene(plot.boundingRect()))


def test_the_crop_keeps_the_axis_labels_the_plots_do_not_contain(viewer):
    """The regression this rectangle exists in its current form for.

    `AxisItem` draws its label in a text item that overhangs the space the plot's layout
    reserved, so a crop to the three plot items alone cut into `m/z` and
    `Arrival time (ms)`. Asserted over every text item in the scene that is not the
    colour bar's, so a label added later is covered without being named here.
    """
    bar = _bar_of(viewer)
    rect = content_rect(viewer.heatmap, viewer.side_plots)

    labels = [
        item for item in viewer.heatmap.scene().items()
        if isinstance(item, QGraphicsTextItem) and item.isVisible()
        and not _under(item, bar) and item.toPlainText().strip()
    ]
    assert labels, "the heatmap's axes should be labelled by now"
    for label in labels:
        assert rect.contains(label.mapRectToScene(label.boundingRect())), label.toPlainText()


def _under(item: object, ancestor: object) -> bool:
    node = item
    while node is not None:
        if node is ancestor:
            return True
        node = node.parentItem()
    return False


def test_the_pixel_size_scales_with_the_resolution():
    rect = QRectF(0.0, 0.0, 800.0, 600.0)

    assert export_pixels(rect, int(BASE_DPI)) == (800, 600)
    assert export_pixels(rect, 2 * int(BASE_DPI)) == (1600, 1200)
    # A degenerate rectangle still has to name a size a QImage can be built at.
    assert export_pixels(QRectF(0.0, 0.0, 0.0, 0.0), 300) == (1, 1)


# --- what is written ---------------------------------------------------------------------

def _export(window: MainWindow, path: str, fmt: str, dpi: int) -> "tuple[int, int]":
    return export_display(window.heatmap, window.side_plots, str(path), fmt, dpi)


@pytest.mark.parametrize("dpi", [96, 300])
def test_a_png_is_written_at_the_promised_size_and_carries_its_resolution(viewer, tmp_path, dpi):
    path = tmp_path / f"figure-{dpi}.png"
    rect = content_rect(viewer.heatmap, viewer.side_plots)

    width, height = _export(viewer, path, "png", dpi)

    assert (width, height) == export_pixels(rect, dpi)
    written = QImage(str(path))
    assert (written.width(), written.height()) == (width, height)
    # Stamped after the render, so the file says how big the figure is in inches without
    # having changed how large its text came out.
    assert written.dotsPerMeterX() == round(dpi / 0.0254)


@pytest.mark.parametrize("name", ["dark", "light"])
def test_the_png_background_follows_the_theme(viewer, tmp_path, name):
    """`export.py` fills from `view.backgroundBrush()` and so needs no theme code of its
    own. That makes "the export follows the theme" a property of one line rather than of
    an intention, which is what this asserts: a light figure on a dark canvas would be
    the most visible way for `View > Light mode` to be half-implemented.

    The corners rather than the middle: the plot area holds the image, and the margins
    around the axes are where the canvas itself shows.
    """
    theme.apply(viewer, name)
    path = tmp_path / f"figure-{name}.png"

    _export(viewer, path, "png", 96)

    written = QImage(str(path))
    expected = QColor(theme.PALETTES[name].background).rgb()
    assert written.pixel(0, 0) == expected
    assert written.pixel(written.width() - 1, written.height() - 1) == expected


def test_the_pdf_background_follows_the_theme(viewer, tmp_path):
    """The PDF fills its page from the same brush, and a vector page cannot be sampled
    for a pixel -- so what is asserted is that the two themes write different bytes and
    that the light one names white."""
    theme.apply(viewer, "dark")
    dark = tmp_path / "dark.pdf"
    _export(viewer, dark, "pdf", 96)

    theme.apply(viewer, "light")
    light = tmp_path / "light.pdf"
    _export(viewer, light, "pdf", 96)

    assert dark.read_bytes() != light.read_bytes()
    assert viewer.heatmap.backgroundBrush().color().name() == theme.LIGHT.background


def test_a_pdf_is_written(viewer, tmp_path):
    path = tmp_path / "figure.pdf"

    _export(viewer, path, "pdf", 300)

    assert path.read_bytes()[:5] == b"%PDF-"


def test_the_pdf_page_is_the_same_size_at_every_resolution(viewer, tmp_path):
    """The page is the figure's own size in inches whatever it is asked for, which is why
    the PDF dialog offers no resolution at all (`export.py`)."""
    small = tmp_path / "small.pdf"
    large = tmp_path / "large.pdf"

    _export(viewer, small, "pdf", 96)
    _export(viewer, large, "pdf", 600)

    def media_box(path):
        marker = b"/MediaBox"
        raw = path.read_bytes()
        start = raw.index(marker)
        return raw[start:raw.index(b"]", start) + 1]

    assert media_box(small) == media_box(large)


# --- what the export leaves behind -------------------------------------------------------

def test_the_export_is_the_picture_on_screen(viewer, tmp_path):
    """The reason task 24 deleted the re-rasterise. A heatmap pixel is an aggregate over
    the bins and scans inside it, so a finer image is a different picture: the colour map
    moves, blobs separate, faint features appear. An export must be the figure the user
    looked at and decided to export."""
    # Copied, and compared by value: `ImageItem.setImage` keeps a `view()` of what it is
    # given rather than the array itself, so identity says nothing about either end.
    before = np.array(viewer.heatmap.image_item.image, copy=True)
    levels = viewer.heatmap.levels()

    _export(viewer, tmp_path / "figure.png", "png", 3 * int(BASE_DPI))

    assert np.array_equal(viewer.heatmap.image_item.image, before)
    assert viewer.heatmap.levels() == levels


def test_the_upscaled_heatmap_is_not_interpolated(viewer):
    """`ImageItem.paint` ends in `drawImage`, which honours `SmoothPixmapTransform`, so
    the hint's absence from `_render` is what makes an enlarged sample a square block.
    A gradient invented between two aggregates would read as data, so this is asserted
    on the painter rather than left to whoever edits the two lines next to it."""
    from PySide6.QtGui import QPainter

    from mainspring.viewer import export

    canvas = QImage(64, 64, QImage.Format.Format_RGB32)
    painter = QPainter(canvas)
    try:
        export._render(
            viewer.heatmap,
            painter,
            QRectF(0, 0, 64, 64),
            content_rect(viewer.heatmap, viewer.side_plots),
        )
        assert not painter.testRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        assert painter.testRenderHint(QPainter.RenderHint.Antialiasing)
    finally:
        painter.end()


def test_nothing_is_left_of_the_re_rasterise(viewer):
    """The deletion itself. `export_levels` existed only to follow the levels a finer
    image moved, and `HeatmapView.substituted` only to stand that image in; both are
    gone, and `hidden_debug` is the part of the second that an export still needs."""
    from mainspring.viewer import export

    assert not hasattr(export, "export_levels")
    assert not hasattr(export, "rasterise")
    assert not hasattr(viewer.heatmap, "substituted")
    assert hasattr(viewer.heatmap, "hidden_debug")


def test_the_debug_overlay_is_not_part_of_a_figure(viewer):
    """A developer's render time is not something to write into a paper, and a raise
    part way through must not leave it off for the rest of the session."""
    viewer.heatmap._debug.setVisible(True)

    with pytest.raises(RuntimeError):
        with viewer.heatmap.hidden_debug():
            assert not viewer.heatmap._debug.isVisible()
            raise RuntimeError("the painter fell over")

    assert viewer.heatmap._debug.isVisible()


# --- what is refused -----------------------------------------------------------------------

def test_an_unknown_format_is_refused_before_anything_is_rendered(viewer, tmp_path):
    with pytest.raises(ValueError, match="unknown export format"):
        _export(viewer, tmp_path / "figure.tiff", "tiff", 300)


def test_a_figure_over_the_pixel_cap_is_refused(viewer, tmp_path, monkeypatch):
    monkeypatch.setattr("mainspring.viewer.export.MAX_PIXELS", 1000)

    with pytest.raises(ValueError, match="megapixels"):
        _export(viewer, tmp_path / "figure.png", "png", 300)
    assert not (tmp_path / "figure.png").exists()


# --- the dialog ------------------------------------------------------------------------------

def test_the_dialog_offers_the_stored_resolution_and_says_what_it_costs(viewer, qtbot):
    rect = content_rect(viewer.heatmap, viewer.side_plots)
    dialog = ExportDialog(viewer, fmt="png", dpi=150, rect=rect)
    qtbot.addWidget(dialog)

    assert dialog.dpi() == 150
    width, height = export_pixels(rect, 150)
    assert f"{width:,} x {height:,} pixels" in dialog._size_label.text()


def test_the_pdf_dialog_offers_no_resolution_and_names_the_page(viewer, qtbot):
    """A PDF page is vector at the figure's own size whatever is asked for, so the only
    thing a resolution could change is nothing."""
    rect = content_rect(viewer.heatmap, viewer.side_plots)
    dialog = ExportDialog(viewer, fmt="pdf", dpi=600, rect=rect)
    qtbot.addWidget(dialog)

    assert dialog._dpi_box is None
    assert dialog.dpi() == int(BASE_DPI)
    assert "dpi" not in dialog._size_label.text()
    assert f"{rect.width() / BASE_DPI:.1f} x" in dialog._size_label.text()
    assert not dialog.findChildren(QComboBox)


def test_a_pdf_export_does_not_move_the_remembered_png_resolution(viewer, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog, QFileDialog

    viewer.settings.export_dpi = 600
    path = tmp_path / "chosen.pdf"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    monkeypatch.setattr(ExportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    viewer._prompt_export("pdf")

    assert path.read_bytes()[:5] == b"%PDF-"
    assert viewer.settings.export_dpi == 600


def test_the_dialog_refuses_a_figure_it_would_not_be_able_to_write(viewer, qtbot, monkeypatch):
    """Refused here, before the file dialog has been answered, rather than by an error
    after the user has already named a file."""
    monkeypatch.setattr("mainspring.viewer.export.MAX_PIXELS", 1000)
    rect = content_rect(viewer.heatmap, viewer.side_plots)
    dialog = ExportDialog(viewer, fmt="png", dpi=600, rect=rect)
    qtbot.addWidget(dialog)

    save = dialog._buttons.button(QDialogButtonBox.StandardButton.Save)
    assert not save.isEnabled()
    assert "megapixels" in dialog._size_label.text()


# --- the menu --------------------------------------------------------------------------------

def test_the_menu_entry_asks_where_then_at_what_resolution_then_writes_it(
    viewer, tmp_path, monkeypatch
):
    """The wiring between the two dialogs and the export, which nothing else covers.

    Both dialogs are answered for the test rather than driven: a native save dialog is
    the platform's, not this window's, and `ExportDialog` is exercised on its own above.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog

    path = tmp_path / "chosen.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    monkeypatch.setattr(ExportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(ExportDialog, "dpi", lambda self: 150)

    viewer._prompt_export("png")

    assert QImage(str(path)).size().isValid()
    assert viewer.settings.export_dpi == 150  # remembered for the next export
    assert "150 dpi" in viewer.statusBar().currentMessage()


def test_a_cancelled_save_dialog_writes_nothing(viewer, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    opened: list = []
    monkeypatch.setattr(ExportDialog, "exec", lambda self: opened.append(True))

    viewer._prompt_export("pdf")

    assert not opened


def test_the_default_name_is_the_file_and_the_frame(viewer):
    """A figure is nearly always named after both, so the save dialog opens on that."""
    name = os.path.basename(viewer._export_default_path("png"))

    assert name.startswith("synthetic-frame")
    assert name.endswith(".png")



def test_the_export_entries_are_grey_until_something_has_been_drawn(qtbot, synthetic_uimf):
    window = MainWindow()
    qtbot.addWidget(window)

    assert not window.export_png_action.isEnabled()
    assert not window.export_pdf_action.isEnabled()

    with qtbot.waitSignal(window.frame_shown, timeout=20000):
        window.open_file(synthetic_uimf.path)

    assert window.export_png_action.isEnabled()
    assert window.export_pdf_action.isEnabled()
