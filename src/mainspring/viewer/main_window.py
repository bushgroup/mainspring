"""`MainWindow`: the one object that knows about all the others.

It owns the file, the settings, the two workers, and the four widgets, and it is the
only place where they meet -- the heatmap does not know there is an info panel, the
side plots do not know there is a file. That is what keeps each of the later tasks to
one or two modules.

The flow it coordinates, once around:

    open  ->  LoadWorker decodes a frame  ->  build DisplayAxes  ->  choose the view
          ->  RenderMailbox  ->  RenderWorker  ->  image + profiles + readouts

Task 04 has no `RenderMailbox` yet, so the last two arrows collapse: `_on_frame_loaded`
rasterises directly on the GUI thread. That is deliberate rather than a shortcut -- a
full-frame raster is 10-21 ms (`notes/reader-layer.md`), well under a frame of
animation -- and it is exactly what task 05 replaces once gestures make that path hot.

**Choosing the view is where keep-ranges lives.** On opening a file the window either
takes the frame's full range or keeps the ranges already on screen, depending on the
setting, and that decision belongs here rather than in the heatmap because it is a
property of the session and not of the widget. Task 04 always takes the full range;
keep-ranges arrives with the settings it depends on (task 06).

Frame navigation, the toolbar toggles, and "sum all frames" with its progress and
cancel are also here, since each of them changes what is asked of the workers rather
than how a widget draws.
"""

from __future__ import annotations

import os
import time

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMainWindow, QProgressBar

from ..uimf import DisplayAxes, FrameParams, GlobalParams, SparseFrame, rasterise
from .heatmap import HeatmapView
from .settings import ViewerSettings
from .workers import LoadWorker

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """The application window, holding the heatmap, the load worker, and the settings."""

    frame_shown = Signal(object)
    """Emitted with the `RasterResult` after each (re)paint -- what a test waits on, and
    what the info panel will read from once it exists (task 06)."""

    def __init__(self, settings: "ViewerSettings | None" = None) -> None:
        super().__init__()
        self.settings = settings or ViewerSettings()
        self.setWindowTitle("mainspring")
        self.resize(1000, 700)

        self.heatmap = HeatmapView(colour_map=self.settings.colour_map)
        self.setCentralWidget(self.heatmap)
        self.heatmap.view_resized.connect(self._on_view_resized)

        self._busy = QProgressBar()
        self._busy.setRange(0, 0)  # indeterminate: a decode's length is not known upfront
        self._busy.setMaximumWidth(120)
        self._busy.hide()
        self.statusBar().addPermanentWidget(self._busy)
        self._build_menu()

        self._worker = LoadWorker(self.settings.cache_budget_mb * 1024 * 1024)
        self._worker.opened.connect(self._on_opened)
        self._worker.frame_loaded.connect(self._on_frame_loaded)
        self._worker.failed.connect(self._on_failed)

        self._global: GlobalParams | None = None
        self._current_frame: SparseFrame | None = None
        self._current_axes: DisplayAxes | None = None
        self._open_started = 0.0

    def _build_menu(self) -> None:
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._prompt_open)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_action)

    def _prompt_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open UIMF file", self.settings.last_directory, "UIMF files (*.uimf)"
        )
        if path:
            self.settings.last_directory = os.path.dirname(path)
            self.open_file(path)

    def open_file(self, path: str) -> None:
        """Open a UIMF file and show its first frame.

        Decoding happens on the load worker's thread, so the window and this call both
        return immediately; `_on_frame_loaded` does the rest once the signal arrives.
        """
        self._open_started = time.perf_counter()
        self.statusBar().showMessage(f"Opening {os.path.basename(path)}...")
        self._busy.show()
        self._worker.open(path)

    def show_frame(self, frame: int) -> None:
        """Switch to a frame, keeping the current view. Arrives with the lab record's task 06."""
        raise NotImplementedError("frame navigation arrives with the lab record's task 06")

    def sum_frames(self, frames: "list[int] | None" = None) -> None:
        """Sum several frames into one image, with progress and cancel. Arrives with task 06."""
        raise NotImplementedError("summing frames arrives with the lab record's task 06")

    # --- worker callbacks -------------------------------------------------------------

    def _on_opened(self, global_params: GlobalParams, frame_numbers: "list[int]") -> None:
        self._global = global_params
        if not frame_numbers:
            self._busy.hide()
            self.statusBar().showMessage("This file has no frames")
            return
        self._worker.request_frame(frame_numbers[0])

    def _on_frame_loaded(
        self, frame_number: int, sparse_frame: SparseFrame, frame_params: FrameParams
    ) -> None:
        assert self._global is not None  # a frame cannot load before opened() fires
        calibration = frame_params.calibration(self._global.bin_width_ns)
        axes = DisplayAxes.build(
            sparse_frame, calibration, frame_params.average_tof_length_ns,
            raw_units=self.settings.raw_units, swapped=self.settings.swap_axes,
        )
        self._current_frame = sparse_frame
        self._current_axes = axes
        self._paint(axes.full_range)
        self._busy.hide()
        elapsed_ms = (time.perf_counter() - self._open_started) * 1000.0
        self.statusBar().showMessage(
            f"Frame {frame_number}: {len(sparse_frame)} points"
            f" -- opened in {elapsed_ms:.0f} ms"
        )

    def _on_failed(self, message: str) -> None:
        self._busy.hide()
        self.statusBar().showMessage(f"Error: {message}")

    def _on_view_resized(self, width: int, height: int) -> None:
        if self._current_frame is None or self._current_axes is None:
            return
        self._paint(self._current_axes.full_range, width, height)

    def _paint(self, view_range: "tuple[tuple[float, float], tuple[float, float]]",
               width: "int | None" = None, height: "int | None" = None) -> None:
        if width is None or height is None:
            width, height = self.heatmap.pixel_size()
        x_range, y_range = view_range
        result = rasterise(
            self._current_frame, self._current_axes, x_range, y_range, width, height,
            aggregate=self.settings.aggregate,
        )
        self.heatmap.set_image(result)
        self.frame_shown.emit(result)

    def closeEvent(self, event) -> None:
        self._worker.stop()
        self._worker.wait(2000)
        super().closeEvent(event)
