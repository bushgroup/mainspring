"""What `File > Export` writes, and the two things about it that are easy to get wrong.

The colour bar is left out by cropping rather than by taking anything apart, so the crop
is what is tested: not only that the bar is outside it, but that nothing else is --
`m/z` and `Arrival time (ms)` hang a few pixels outside the plot item that owns them,
and a crop to the plots alone shaved the outer edge off both.

The image is re-rasterised at the export's own resolution, so the levels it is shown
under have to be recomputed: a `sum` pixel covering a quarter of the area holds about a
quarter of the intensity, and reusing the screen's levels would write a nearly black
figure. `export_levels` is tested directly, because a level is a number no assertion
about a PNG could name.

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
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QDialogButtonBox, QGraphicsTextItem

from mainspring.viewer.export import (
    BASE_DPI,
    ExportDialog,
    content_rect,
    export_display,
    export_levels,
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


# --- the levels ---------------------------------------------------------------------------

def test_auto_scaled_levels_come_out_auto_scaled_again():
    """The screen's levels at its own full range are the fractions 0 and 1, so the
    export's are its own full range -- no special case in `export_levels` for it."""
    on_screen = np.array([[0.0, 10.0]])
    exported = np.array([[0.0, 2.5]])

    assert export_levels(on_screen, exported, (0.0, 10.0)) == (0.0, 2.5)


def test_pinned_levels_keep_the_contrast_they_were_pinned_at():
    """Levels held at the bottom half of the screen image's range come out at the bottom
    half of the export's, which is the point: a user pins levels to bring up a faint
    feature, and a figure that lost that has lost what they were looking at."""
    on_screen = np.array([[0.0, 100.0]])
    exported = np.array([[0.0, 25.0]])

    assert export_levels(on_screen, exported, (0.0, 50.0)) == (0.0, 12.5)
    assert export_levels(on_screen, exported, (25.0, 75.0)) == (6.25, 18.75)


def test_a_degenerate_range_falls_back_to_the_exports_own():
    flat = np.zeros((4, 4))

    assert export_levels(flat, np.array([[0.0, 8.0]]), (0.0, 1.0)) == (0.0, 8.0)
    assert export_levels(np.array([[0.0, 8.0]]), flat, (0.0, 8.0)) == (0.0, 1.0)
    assert export_levels(np.empty(0), np.empty(0), (0.0, 1.0)) == (0.0, 1.0)


# --- what is written ---------------------------------------------------------------------

def _export(window: MainWindow, path: str, fmt: str, dpi: int) -> "tuple[int, int]":
    return export_display(
        window.heatmap, window.side_plots, window._current_frame,
        window.last_render.result, window.settings.colour_scale, str(path), fmt, dpi,
    )


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


def test_a_pdf_is_written(viewer, tmp_path):
    path = tmp_path / "figure.pdf"

    _export(viewer, path, "pdf", 300)

    assert path.read_bytes()[:5] == b"%PDF-"


def test_the_pdf_page_is_the_same_size_at_every_resolution(viewer, tmp_path):
    """The resolution buys a PDF the sample count of the heatmap it embeds and nothing
    else: the page is the figure's own size in inches either way, so text and rules come
    out identical and only the image gets finer (`export.py`)."""
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
    assert large.stat().st_size > small.stat().st_size  # the image, and only the image


# --- the re-rasterise ----------------------------------------------------------------------

def test_the_image_is_re_rasterised_finer_than_the_screens(viewer, tmp_path, monkeypatch):
    """The reason this module exists rather than a two-line `scene.render`.

    Caught at the seam it happens through, because the exported PNG cannot be asked how
    many samples went into it -- and along the horizontal axis only, since the synthetic
    frame has fewer scans than the viewport has pixels and `rasterise` will not invent
    rows it has no elements for.
    """
    shown: list[tuple] = []
    substituted = viewer.heatmap.substituted
    monkeypatch.setattr(
        viewer.heatmap, "substituted",
        lambda image, rect, levels: shown.append(image.shape) or substituted(image, rect, levels),
    )
    on_screen = viewer.heatmap.image_item.image.shape

    _export(viewer, tmp_path / "figure.png", "png", 3 * int(BASE_DPI))

    assert len(shown) == 1
    assert shown[0][1] == pytest.approx(3 * on_screen[1], rel=0.02)


def test_the_screen_image_is_put_back_afterwards(viewer, tmp_path):
    # Copied, and compared by value: `ImageItem.setImage` keeps a `view()` of what it is
    # given rather than the array itself, so identity says nothing about either end.
    before = np.array(viewer.heatmap.image_item.image, copy=True)
    levels = viewer.heatmap.levels()

    _export(viewer, tmp_path / "figure.png", "png", 300)

    assert np.array_equal(viewer.heatmap.image_item.image, before)
    assert viewer.heatmap.levels() == levels


def test_the_screen_image_is_put_back_even_if_the_render_raises(viewer):
    """A context manager and not two calls, for this: a window left showing a
    substitute after a failed export is a viewer whose readouts no longer describe what
    is on screen."""
    before = np.array(viewer.heatmap.image_item.image, copy=True)
    levels = viewer.heatmap.levels()
    finer = np.full((before.shape[0], before.shape[1] * 2), 7.0, dtype=np.float32)

    with pytest.raises(RuntimeError):
        with viewer.heatmap.substituted(finer, (0.0, 0.0, 1.0, 1.0), (0.0, 5.0)):
            assert viewer.heatmap.image_item.image.shape == finer.shape
            assert viewer.heatmap.levels() == (0.0, 5.0)
            raise RuntimeError("the painter fell over")

    assert np.array_equal(viewer.heatmap.image_item.image, before)
    assert viewer.heatmap.levels() == levels


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
