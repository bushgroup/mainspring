"""`MainWindow`: the one object that knows about all the others.

It owns the file, the settings, the two workers, and the widgets, and it is the only
place where they meet -- the heatmap does not know there is an info panel, the side
plots do not know there is a file. That is what keeps each of the later tasks to one or
two modules.

The flow it coordinates, once around:

    open  ->  LoadWorker decodes a frame  ->  build DisplayAxes  ->  choose the view
          ->  RenderMailbox  ->  RenderWorker  ->  image + profiles + readouts

Every arrow after the third is travelled by every gesture, every frame change, and every
toolbar toggle alike, which is what task 05 built the render path for: a wheel tick, a
frame spinner step and a swap-axes toggle all reach the render worker by the same path,
so there is one place where "what is on screen" is decided and no second, direct route
that could disagree with it.

**Results are matched to a serial, not trusted in arrival order.** The mailbox drops
superseded *requests*, but a result already in flight when the frame changes would
otherwise repaint the new frame's window with the old frame's data. Every request
carries the serial of the state that produced it, and a result whose serial is not the
current one is dropped -- which is what made frame navigation and the axis toggles below
safe to add without touching the render path.

**Choosing the view is where keep-ranges lives.** Opening a file either takes the
frame's full range or keeps whatever is already on screen, depending on the setting;
navigating between frames of the *same* file always keeps the view, regardless of the
setting -- paging through a run should never cost the zoom the user just set. Swapping
axes or switching to raw units does neither: it re-expresses the same visible bins and
scans through the new axis tables (`_rebuild_axes`), which is the third way "keep the
view" comes up here.

**Three spinners name the frame, and only one of them is ever the source.** A clockwork
raw acquisition is one frame per repetition, so a method frame is a run of consecutive
frames and the toolbar can say which frame by number, or by method frame and repetition
(`FrameGrouping`, lab record, tasks 16 and 17). The frame number is the truth:
`_sync_grouping_controls` sets the other two from it with their signals blocked, and
nothing sets the frame spinner except `_show_frame`. Three controls each able to drive
the others would be a loop. All of them are hidden on a file that does not carry the
grouping, which is every file PNNL's writers produce.

**Following is two controls, because it is two decisions.** `Follow` is whether the
file is being watched at all; `Show` is what to do about what the watch finds. Watching
without moving is a real thing to want -- the frame axis and the repetition count grow
while the operator stays on the frame they were studying -- so it cannot be folded into
"jump to the newest frame", and jumping cannot be the only reason to watch. Everything
either one does arrives through `LoadWorker.live_update` and goes out again through
`show_frame` and `sum_frames`, so a followed acquisition reaches the screen by exactly
the path a keypress does (lab record, task 08).
"""

from __future__ import annotations

import bisect
import os
import time

import numpy as np
from PySide6.QtCore import Qt, QByteArray, Signal
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QSpinBox,
    QToolBar,
)

from ..uimf import (
    DisplayAxes,
    FrameGrouping,
    FrameParams,
    GlobalParams,
    LiveState,
    SparseFrame,
    is_local_path,
)
from . import fonts, labels, theme
from .controls import (
    DoubleSpinBox,
    SpinBox,
    add_labelled,
    add_menu_widget,
    describe,
    make_action,
)
from .export import ExportDialog, content_rect, export_display
from .heatmap import HeatmapView, pixel_of
from .info_panel import InfoPanel
from .settings import (
    COLOUR_MAPS,
    COLOUR_SCALES,
    TEXT_SCALES,
    ViewerSettings,
    load_settings,
    save_settings,
)
from .side_plots import SidePlots, peak_of
from .workers import LoadWorker, RenderMailbox, RenderRequest, RenderWorker, SumRequest

__all__ = ["APP_TITLE", "FOLLOW_MODES", "MainWindow"]

APP_TITLE = "mainspring"
"""The window title with no file open. With one open it is `"<file name> -- mainspring"`,
file first, the way Windows names a document window, so that two viewers on the taskbar
can be told apart by what they show rather than by what they are."""

# The sample's own frame-type convention (`notes/uimf-format.md`), extended past what
# our one sample uses: 0 and 1 both mean MS1 across the modern and legacy tables
# (`FrameParams.is_ms1`), 2 is MS/MS, 3 a calibration frame, 4 a prescan. An unrecognised
# code is shown as itself rather than dropped -- a future writer's code is still a
# choice worth filtering on, whether or not this table has a name for it yet.
_FRAME_TYPE_NAMES = {0: "MS1", 1: "MS1", 2: "MS2", 3: "Calibration", 4: "Prescan"}

COLOUR_SCALE_NAMES = {"linear": "Linear", "log": "Log", "sqrt": "Square root"}
"""What each of `settings.COLOUR_SCALES` is called in `View > Colour scale`. Here and not
there because these are display strings and `settings.py` is the module that must not
know about the menu; `"sqrt"` is the one that needs it, since the capitalised code word
is not what a reader of a menu is looking for."""

FOLLOW_FIXED = "Fixed frame"
FOLLOW_NEWEST = "Newest frame"
FOLLOW_METHOD_SUM = "Method frame sum"
FOLLOW_MODES = (FOLLOW_FIXED, FOLLOW_NEWEST, FOLLOW_METHOD_SUM)
"""What the `Show` box does with what the follow poll finds.

`Fixed frame` watches and does not move: the frame spinner's range, the type filter and
the repetition count grow under a frame the operator chose, which is what someone
studying one experiment while the run continues wants. `Newest frame` shows each frame
as it arrives, the one still being written included and labelled as such -- a frame is
about a second of acquisition and its scans land in batches, so it fills in front of
the operator rather than appearing whole. `Method frame sum` keeps a running total of
the finished repetitions of the method frame being acquired, which is the summed
heatmap today's files hold, arriving as it is earned. The third is offered only on a
file that carries the grouping (lab record, task 08)."""


def _frame_type_name(frame_type: int) -> str:
    return _FRAME_TYPE_NAMES.get(int(frame_type), f"Type {frame_type}")


class MainWindow(QMainWindow):
    """The application window, holding the heatmap, the info panel, the workers and the
    settings."""

    frame_shown = Signal(object)
    """Emitted with the `RasterResult` after each (re)paint -- what a test waits on, and
    what the self-check drives a gesture through."""

    def __init__(self, settings: "ViewerSettings | None" = None) -> None:
        super().__init__()
        self.settings = settings or load_settings()
        self.setWindowTitle(APP_TITLE)
        if self.settings.window_geometry:
            self.restoreGeometry(QByteArray(self.settings.window_geometry))
        else:
            self.resize(1000, 700)

        # State that a signal fired while building the toolbar below can already read --
        # a settings-restored toggle emits its own `toggled` the moment it is checked,
        # and that handler must see "no frame yet" rather than an attribute that does
        # not exist.
        self._global: GlobalParams | None = None
        self._current_frame: SparseFrame | None = None
        self._current_axes: DisplayAxes | None = None
        self._current_frame_number: int | None = None
        self._frame_params: FrameParams | None = None
        self._last_render: object | None = None
        self._serial = 0
        self._open_started = 0.0
        self._opening = False
        self._opened_from_command_line = False
        self._path: str | None = None
        self._frame_message = ""
        self._frame_numbers: list[int] = []
        self._frame_types: dict[int, int] = {}
        self._active_frame_numbers: list[int] = []
        self._grouping = FrameGrouping()
        self._sum_dialog: QProgressDialog | None = None
        self._live: LiveState | None = None
        self._live_sum_frames: tuple[int, ...] = ()

        self.heatmap = HeatmapView(colour_map=self.settings.colour_map)
        self.setCentralWidget(self.heatmap)
        self.side_plots = SidePlots(self.heatmap)
        # Before any of the window's own signals are wired: both plot widgets exist, so
        # this is the first moment the restored palette can be put on them, and doing it
        # here means every later change goes through the same one call (`theme.apply`).
        theme.apply(self, self.settings.theme)
        self.heatmap.view_resized.connect(self._on_view_resized)
        self.heatmap.view_changed.connect(self._on_view_changed)
        self.heatmap.cursor_moved.connect(self._on_cursor_moved)
        self.heatmap.cursor_left.connect(self._clear_readout)

        self.info_panel = InfoPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.info_panel)
        # After `addDockWidget` and not before: `resizeDocks` acts on the dock's position
        # in the main window's layout, which it does not have until it has been added.
        # The panel is no longer a fixed width (`info_panel.py`), so the width a user
        # drags it to is theirs to keep.
        self.resizeDocks(
            [self.info_panel], [self.settings.info_panel_width], Qt.Orientation.Horizontal
        )
        self.info_panel.setVisible(self.settings.show_info_panel)
        # The dock's own action rather than a hand-rolled one: Qt keeps it in step with
        # the dock's visibility in both directions, so the X button on the dock, the
        # toolbar button and the menu entry are one switch with three handles.
        self._info_action = self.info_panel.toggleViewAction()
        self._info_action.setText("Info")
        self._info_action.setShortcut("Ctrl+I")
        describe(
            self._info_action,
            "Show the panel of frame parameters and in-view totals.",
            shortcut="Ctrl+I",
        )
        self._info_action.toggled.connect(self._on_info_toggled)

        self._busy = QProgressBar()
        self._busy.setRange(0, 0)  # indeterminate: a decode's length is not known upfront
        self._busy.setMaximumWidth(120)
        self._busy.hide()
        describe(self._busy, "A frame is being decoded.")
        # Three zones, left to right: where the pointer is, what the window is doing,
        # and where each projection peaks. `QStatusBar.showMessage` is not used at all
        # (`_show_status`), so the left-hand widgets are never covered over.
        self._readout = QLabel("")
        describe(
            self._readout,
            "The axis values under the pointer, and the intensity of the pixel it is over.",
        )
        self._status = QLabel("")
        describe(self._status, "What the viewer last did, and what the frame on screen is.")
        self._peaks = QLabel("")
        describe(
            self._peaks,
            "Where each projection peaks and how high, with each projection being a sum"
            " over the range in view on the other axis.",
        )
        self.statusBar().addWidget(self._readout, 0)
        self.statusBar().addWidget(self._status, 1)
        self.statusBar().addPermanentWidget(self._peaks)
        self.statusBar().addPermanentWidget(self._busy)
        # After all three owners exist and before the menus are built: the application
        # font reaches a widget made afterwards as surely as one made already, so this
        # is simply the first moment `fonts.apply` has everything it needs.
        fonts.apply(self, self.settings.text_scale)
        self._build_menu()
        self._build_toolbar()

        self._worker = LoadWorker(self.settings.cache_budget_mb * 1024 * 1024)
        self._worker.opened.connect(self._on_opened)
        self._worker.frame_loaded.connect(self._on_frame_loaded)
        self._worker.summing_progress.connect(self._on_summing_progress)
        self._worker.summed.connect(self._on_summed)
        self._worker.live_update.connect(self._on_live_update)
        self._worker.follow_stopped.connect(self._on_follow_stopped)
        self._worker.failed.connect(self._on_failed)

        self._mailbox = RenderMailbox()
        self._render_worker = RenderWorker(self._mailbox)
        self._render_worker.rendered.connect(self._on_rendered)
        self._render_worker.failed.connect(self._on_failed)
        self._render_worker.start()

    def _build_menu(self) -> None:
        self.open_action = make_action(
            self,
            "&Open...",
            tip="Open a UIMF file.",
            shortcut="Ctrl+O",
            triggered=self._prompt_open,
        )
        # One entry per format rather than one "Export..." with a format chooser: the
        # File menu is where a user goes looking for the words PNG and PDF, and the
        # dialog behind both then has one thing in it (`export.ExportDialog`).
        self.export_png_action = make_action(
            self,
            "Export &PNG...",
            tip="Save the heatmap and its projections as a PNG image, without the colour bar.",
            triggered=lambda: self._prompt_export("png"),
        )
        self.export_pdf_action = make_action(
            self,
            "Export P&DF...",
            tip="Save the heatmap and its projections as a PDF, without the colour bar.",
            triggered=lambda: self._prompt_export("pdf"),
        )
        # Nothing to export until something has been rendered, and an entry that opens a
        # file dialog and then reports that there is no image is worse than a grey one.
        for action in (self.export_png_action, self.export_pdf_action):
            action.setEnabled(False)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_png_action)
        file_menu.addAction(self.export_pdf_action)

        # A window-level action rather than a key handler on the view box: the reset must
        # work wherever the focus happens to be, which is the complaint about having to
        # find it in a context menu (lab record, task 01).
        self.reset_action = make_action(
            self,
            "&Reset view",
            tip="Return the heatmap to the frame's full range.",
            shortcut="Home",
            triggered=self.heatmap.reset_range,
        )
        # A two-state entry rather than a Theme submenu of two: there are two palettes,
        # the dark one is the default, and a tick beside one word says which is on.
        self._light_action = make_action(
            self,
            "&Light mode",
            tip="Draw the plot area on white instead of black. The colour map does not change.",
            checkable=True,
            checked=(self.settings.theme == "light"),
            toggled=self._on_light_mode_toggled,
        )
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.reset_action)
        view_menu.addAction(self._info_action)
        view_menu.addAction(self._light_action)
        self._build_colour_map_menu(view_menu)
        self._build_colour_scale_menu(view_menu)
        self._build_text_size_menu(view_menu)
        self._build_data_menu()

    def _build_colour_map_menu(self, view_menu: "object") -> None:
        """A curated four-map choice, radio-style, in its own `View` submenu.

        pyqtgraph's `ColorBarItem` ships a right-click menu of its own listing every
        registered colormap; it is turned off at construction (`heatmap.py`) because a
        choice made through it would neither tick an entry here nor reach
        `ViewerSettings` -- this menu is the one place the colour map lives.
        """
        colour_map_menu = view_menu.addMenu("Colour map")
        group = QActionGroup(self)
        group.setExclusive(True)
        for name in COLOUR_MAPS:
            action = make_action(
                self,
                name.capitalize(),
                tip=f"Colour the heatmap with the {name.capitalize()} colormap.",
                checkable=True,
                checked=(name == self.settings.colour_map),
                toggled=lambda checked, n=name: self._on_colour_map_changed(n, checked),
            )
            group.addAction(action)
            colour_map_menu.addAction(action)

    def _build_colour_scale_menu(self, view_menu: "object") -> None:
        """How intensity maps onto colour, radio-style, beside the colour map.

        Built exactly like `_build_colour_map_menu` and placed next to it because the
        two are one question asked twice -- which colours, and how the numbers are
        spread across them. It was a toolbar combo until a user reading isotopically
        resolved spectra pointed out that everything describing how the *data* are drawn
        should be in a menu and the toolbar should be the things touched every minute
        (lab record, task 24).
        """
        colour_scale_menu = view_menu.addMenu("Colour scale")
        group = QActionGroup(self)
        group.setExclusive(True)
        for name in COLOUR_SCALES:
            label = COLOUR_SCALE_NAMES[name]
            action = make_action(
                self,
                label,
                tip=f"Map intensity onto colour on a {label.lower()} scale."
                    " The readouts still quote the untransformed intensity.",
                checkable=True,
                checked=(name == self.settings.colour_scale),
                toggled=lambda checked, n=name: self._on_colour_scale_changed(n, checked),
            )
            group.addAction(action)
            colour_scale_menu.addAction(action)

    def _build_text_size_menu(self, view_menu: "object") -> None:
        """One scale over every piece of text in the application, radio-style.

        In `View` rather than anywhere else because it is a statement about how the
        window is read, like the palette beside it. The four steps are
        `settings.TEXT_SCALES`; what each one reaches, and why an application font change
        is not enough on its own, is `viewer/fonts.py`.
        """
        text_size_menu = view_menu.addMenu("Text size")
        group = QActionGroup(self)
        group.setExclusive(True)
        for scale in TEXT_SCALES:
            label = f"{round(scale * 100)}%"
            action = make_action(
                self,
                label,
                tip=f"Draw every piece of text in the window at {label} of its"
                    " normal size.",
                checkable=True,
                checked=(scale == self.settings.text_scale),
                toggled=lambda checked, s=scale: self._on_text_scale_changed(s, checked),
            )
            group.addAction(action)
            text_size_menu.addAction(action)

    def _build_data_menu(self) -> None:
        """`Data settings`: what the numbers on screen are, rather than how they look.

        Three controls that were toolbar widgets until a user reading isotopically
        resolved spectra asked what `Bits` was (lab record, task 24). None of the three
        is touched in the ordinary course of looking at a frame -- the aggregate and the
        bit depth are set once for a session, and the type filter once for a file -- so
        each one was costing toolbar width that the frame navigation beside it earns
        every minute. The mnemonic is `D`, which neither `&File` nor `&View` has taken.

        `Aggregate` and `Type` are radio submenus and `Bits` is the same 1-32 spin box
        it always was, hosted in a `QWidgetAction`: a range is not a choice between
        fixed options, and a menu of thirty-two numbers would be a worse control than
        the box.
        """
        menu = self.menuBar().addMenu("&Data settings")

        aggregate_menu = menu.addMenu("Aggregate")
        aggregate_group = QActionGroup(self)
        aggregate_group.setExclusive(True)
        self._aggregate_actions: "dict[str, object]" = {}
        for name, what in (
            ("sum", "add the intensities inside one screen pixel, which conserves the"
                    " total"),
            ("max", "take the largest intensity inside one screen pixel, which keeps a"
                    " single-bin spike visible"),
        ):
            action = make_action(
                self,
                name.capitalize(),
                tip=f"Combine by {name}: {what}.",
                checkable=True,
                checked=(name == self.settings.aggregate),
                toggled=lambda checked, n=name: self._on_aggregate_changed(n, checked),
            )
            aggregate_group.addAction(action)
            aggregate_menu.addAction(action)
            self._aggregate_actions[name] = action

        # Rebuilt from the file's own frame types on every open and on every follow poll
        # (`_populate_type_filter`), so it is built empty here and filled there.
        self._type_menu = menu.addMenu("Type")
        self._type_group = QActionGroup(self)
        self._type_group.setExclusive(True)
        self._type_actions: "dict[str, object]" = {}
        self._populate_type_filter()

        self._bits_box = SpinBox()
        self._bits_box.setRange(1, 32)
        self._bits_box.setValue(self.settings.detector_bits)
        self._bits_box.valueChanged.connect(self._on_detector_bits_changed)
        self._bits_label = add_menu_widget(
            menu,
            "Bits: ",
            self._bits_box,
            tip="Detector bit depth, 1 to 32, that the per-push readout assumes.",
        )

    def _build_toolbar(self) -> None:
        """The axis toggles, frame navigation, the sums and follow.

        What is left after task 24 moved everything describing the *data* into
        `Data settings` and everything describing how it is *coloured* into `View`: the
        toolbar is now the controls a user touches while looking at a frame.

        Each control is fully configured -- range, items, initial value from the
        restored settings -- **before** its signal is connected, so that restoring a
        non-default setting cannot fire a handler while the rest of the window is still
        being built. `make_action` holds that order for the actions; the widgets below
        keep it by hand, because their value has to be set before `add_labelled` can
        hand them over.
        """
        toolbar = QToolBar("View", self)
        toolbar.setObjectName("view_toolbar")  # QMainWindow.saveState() keys it by this
        self.addToolBar(toolbar)
        # Qt makes this one itself and offers it in the window's right-click menu, which
        # is the only place it appears -- so it is the control most easily left mute.
        describe(toolbar.toggleViewAction(), "Show the toolbar.")

        self._swap_action = make_action(
            self,
            "Swap X/Y",
            tip="Put arrival time on the horizontal axis and m/z on the vertical.",
            checkable=True,
            checked=self.settings.swap_axes,
            toggled=self._on_swap_toggled,
        )
        toolbar.addAction(self._swap_action)

        self._raw_action = make_action(
            self,
            "Raw units",
            tip="Show TOF bin and scan number instead of calibrated m/z and arrival time.",
            checkable=True,
            checked=self.settings.raw_units,
            toggled=self._on_raw_units_toggled,
        )
        toolbar.addAction(self._raw_action)

        self._arrival_offset_box = DoubleSpinBox()
        self._arrival_offset_box.setRange(-100_000.0, 100_000.0)
        self._arrival_offset_box.setDecimals(3)
        self._arrival_offset_box.setSingleStep(1.0)
        self._arrival_offset_box.setValue(self.settings.arrival_offset_ms)
        self._arrival_offset_box.valueChanged.connect(self._on_arrival_offset_changed)
        self._arrival_offset_label = add_labelled(
            toolbar,
            " Arrival offset (ms): ",
            self._arrival_offset_box,
            tip="Shift the displayed arrival-time axis by this many milliseconds.",
        )

        self._keep_ranges_action = make_action(
            self,
            "Keep ranges",
            tip="Keep the current zoom when the next file is opened.",
            checkable=True,
            checked=self.settings.keep_ranges,
            toggled=self._on_keep_ranges_toggled,
        )
        toolbar.addAction(self._keep_ranges_action)

        self._keep_levels_action = make_action(
            self,
            "Keep levels",
            tip="Keep the current colour limits instead of rescaling to each frame.",
            checkable=True,
            checked=self.settings.keep_levels,
            toggled=self._on_keep_levels_toggled,
        )
        toolbar.addAction(self._keep_levels_action)

        toolbar.addAction(self._info_action)  # built with the dock, above

        self._frame_spin = SpinBox()
        self._frame_spin.setRange(0, 0)
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        self._frame_label = add_labelled(
            toolbar,
            " Frame: ",
            self._frame_spin,
            tip="Go to a frame by number, within the frames the type filter allows.",
        )

        # The method-frame group, hidden on every file that does not carry the grouping
        # -- which is every file PNNL's writers produce. Hidden and not merely disabled:
        # a control that can never do anything on this file is not a control, and the
        # toolbar is already long. `_show_grouping_controls` is the one switch, and what
        # it switches is the toolbar's own actions, not the widgets: a `QToolBar` lays
        # out by action visibility, so hiding the widget alone would leave its gap.
        self._grouping_actions: "list[object]" = []
        self._method_spin = SpinBox()
        self._method_spin.setRange(0, 0)
        self._method_spin.valueChanged.connect(self._on_method_spin_changed)
        self._method_label = self._add_hideable(
            toolbar,
            " Method frame: ",
            self._method_spin,
            tip="Go to a method frame. One method frame is every repetition the method"
                " asked for, stored here as consecutive frames.",
        )

        self._repetition_spin = SpinBox()
        self._repetition_spin.setRange(0, 0)
        self._repetition_spin.valueChanged.connect(self._on_repetition_spin_changed)
        self._repetition_label = self._add_hideable(
            toolbar,
            " Rep: ",
            self._repetition_spin,
            tip="Step through the repetitions of the method frame on screen, to see"
                " whether they drift.",
        )
        # How many the method asked for, beside a spinner whose maximum is how many the
        # file holds. The two differ exactly when a method frame was cut short, which is
        # worth seeing at a glance rather than working out.
        self._repetitions_readout = QLabel("")
        describe(
            self._repetitions_readout,
            "How many repetitions of this method frame the file holds, and how many the"
            " method asked for.",
        )
        self._grouping_actions.append(toolbar.addWidget(self._repetitions_readout))

        self.sum_method_frame_action = make_action(
            self,
            "Sum method frame",
            tip="Add every repetition of the method frame on screen into one heatmap.",
            triggered=lambda: self.sum_method_frame(),
        )
        toolbar.addAction(self.sum_method_frame_action)
        self._grouping_actions.append(self.sum_method_frame_action)

        self.sum_action = make_action(
            self,
            "Sum all",
            tip="Add every frame passing the type filter into one heatmap.",
            triggered=lambda: self.sum_frames(),
        )
        toolbar.addAction(self.sum_action)
        self._show_grouping_controls(False)

        # Follow last, because it is the control that acts on all the others: what it
        # finds reaches the screen through the frame spinner and the sum this toolbar
        # already carries. Not a persisted setting, unlike every other toggle here --
        # see `_on_follow_toggled`.
        self._follow_action = make_action(
            self,
            "Follow",
            tip="Watch this file for frames the instrument is still writing.",
            checkable=True,
            checked=False,
            toggled=self._on_follow_toggled,
        )
        toolbar.addAction(self._follow_action)

        self._follow_mode = QComboBox()
        self._follow_mode.addItems([FOLLOW_FIXED, FOLLOW_NEWEST])
        self._follow_mode.setEnabled(False)
        self._follow_mode.currentTextChanged.connect(self._on_follow_mode_changed)
        self._follow_label = add_labelled(
            toolbar,
            " Show: ",
            self._follow_mode,
            tip="Choose what following does with each new frame: stay where you are,"
                " show the newest frame, or keep a running sum of the method frame"
                " being acquired.",
        )

    def _add_hideable(self, toolbar: QToolBar, text: str, widget: object, *, tip: str) -> QLabel:
        """`add_labelled`, remembering the two toolbar actions that can hide the pair."""
        before = len(toolbar.actions())
        label = add_labelled(toolbar, text, widget, tip=tip)
        self._grouping_actions.extend(toolbar.actions()[before:])
        return label

    def _show_grouping_controls(self, shown: bool) -> None:
        """Show or hide the method-frame group as one thing."""
        for action in self._grouping_actions:
            action.setVisible(shown)

    def _prompt_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open UIMF file", self.settings.last_directory, "UIMF files (*.uimf)"
        )
        if path:
            self.settings.last_directory = os.path.dirname(path)
            self.open_file(path)

    def _prompt_export(self, fmt: str) -> None:
        """Ask where, then at what resolution, then write it.

        Path first and options second, the order Windows puts them in: a cancelled file
        dialog costs nothing, and the resolution dialog can then name the size the file
        it is about to write will be.
        """
        if self._current_frame is None or self._last_render is None:
            return
        filters = {"png": "PNG image (*.png)", "pdf": "PDF document (*.pdf)"}
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {fmt.upper()}", self._export_default_path(fmt), filters[fmt]
        )
        if not path:
            return
        dialog = ExportDialog(
            self, fmt=fmt, dpi=self.settings.export_dpi,
            rect=content_rect(self.heatmap, self.side_plots),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        # Not for a PDF: its dialog has no resolution row, so the number it answers with
        # is `BASE_DPI` and remembering it would quietly reset the PNG choice the user
        # made last time (lab record, task 24).
        if fmt != "pdf":
            self.settings.export_dpi = dialog.dpi()
        self._show_status(f"Exporting {os.path.basename(path)}...")
        # A 600 dpi render of a maximised window is still a second or two of a frozen
        # window, and it happens on this thread: the render worker's mailbox drops
        # whatever it is holding when a newer request arrives, which is right for a
        # gesture and would silently lose an export. A wait cursor is the honest way to
        # say so.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            width, height = export_display(
                self.heatmap, self.side_plots, path, fmt, dialog.dpi()
            )
        except Exception as exc:  # noqa: BLE001 -- shown in the status bar, as a failed render is
            self._on_failed(f"could not export {os.path.basename(path)}: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._show_status(
            f"Exported {os.path.basename(path)}, {width} x {height} pixels"
            f" at {dialog.dpi()} dpi"
        )

    def _export_default_path(self, suffix: str) -> str:
        """The file the save dialog opens on: the frame being looked at, beside the file
        it came from. A figure is nearly always named after both."""
        stem = os.path.splitext(os.path.basename(self._path))[0] if self._path else APP_TITLE
        frame = self._current_frame_number
        name = f"{stem}-frame{frame}.{suffix}" if frame else f"{stem}.{suffix}"
        return os.path.join(self.settings.last_directory, name)

    def open_file(self, path: str, *, from_command_line: bool = False) -> None:
        """Open a UIMF file and show its first frame.

        Decoding happens on the load worker's thread, so the window and this call both
        return immediately; `_on_opened` and `_on_frame_loaded` do the rest once the
        signals arrive.

        `from_command_line` marks an open that reached the window because a file
        association or a drag onto the `.exe` launched the process for this file, rather
        than a deliberate `File > Open`. `_on_failed` uses it to decide whether a status
        bar line is enough (the window is already in front of a user who chose the file)
        or whether a bad double-click needs a modal that names what happened (lab
        record, task 15).
        """
        self._open_started = time.perf_counter()
        # Following is about one acquisition, so it does not survive into the next file:
        # a second file opened from the dialog is nearly always a finished one, and a
        # poll left running on it would be lock traffic against nothing.
        self.stop_following()
        self._path = path
        self._opened_from_command_line = from_command_line
        self._show_status(f"Opening {os.path.basename(path)}...")
        self._busy.show()
        self._opening = True
        self._worker.open(path)

    @property
    def last_render(self) -> "object | None":
        """The newest `RenderResult` drawn: the image, both profiles and what they cost.

        The window's answer to "what is on screen right now", for the self-check and
        for a test that wants the profiles the side plots were given rather than the
        pixels they became.
        """
        return self._last_render

    def show_frame(self, frame: int) -> None:
        """Switch to a frame, keeping the current view.

        Unlike opening a file, this never resets the range -- keep-ranges is about what
        happens across a *file* open, and paging through one file's frames must not cost
        the zoom the user just set regardless of that setting. `_on_frame_loaded` is
        what tells the two cases apart, by `_opening`.
        """
        if self._global is None:
            return
        self._worker.request_frame(int(frame))

    def _sync_grouping_controls(self, frame_number: int) -> None:
        """Put the method-frame and repetition spinners on the frame now showing.

        The frame spinner, the method-frame spinner and the repetition spinner are three
        ways of saying the same thing, so exactly one of them is ever the source: the
        frame number. Both others are set from it here, with their signals blocked --
        the alternative is two controls that can each drive the other and a loop between
        them.

        Silent on a frame the grouping does not cover, which is the sum-all and
        sum-method-frame result (frame 0, not a frame of the file). Leaving the spinners
        where they were is right for both: after summing one method frame they still
        name it.
        """
        if not self._grouping.grouped:
            return
        method_frame = self._grouping.method_frame.get(int(frame_number))
        if method_frame is None:
            return
        members = self._grouping.frames.get(method_frame, ())
        asked = self._grouping.repetitions.get(method_frame, len(members))
        self._method_spin.blockSignals(True)
        self._method_spin.setValue(method_frame)
        self._method_spin.blockSignals(False)
        self._repetition_spin.blockSignals(True)
        self._repetition_spin.setRange(1, max(1, len(members)))
        self._repetition_spin.setValue(self._grouping.repetition.get(int(frame_number), 1))
        self._repetition_spin.blockSignals(False)
        if self._grouping.covers_whole(method_frame):
            text = f" (all {asked} repetitions in one frame) "
        elif self._grouping.is_short(method_frame):
            text = f" of {len(members)}, method asked {asked} "
        else:
            text = f" of {asked} "
        self._repetitions_readout.setText(text)

    def show_method_frame(self, method_frame: int, repetition: "int | None" = None) -> None:
        """Show one repetition of a method frame, by default the one already on screen.

        Keeping the repetition across a method-frame change is what makes the two
        spinners two independent axes rather than one path: stepping the method frame
        holds the repetition and shows the same point of each experiment, stepping the
        repetition holds the method frame and shows the drift within one. A method frame
        with fewer frames than the one before it clamps rather than refusing.
        """
        members = self._grouping.frames.get(int(method_frame), ())
        if not members:
            return
        wanted = self._repetition_spin.value() if repetition is None else int(repetition)
        index = min(max(1, wanted), len(members)) - 1
        self.show_frame(members[index])

    def sum_method_frame(self, method_frame: "int | None" = None) -> None:
        """Sum every repetition of one method frame, by default the one on screen.

        The whole reason the grouping is in the file: the summed heatmap of one ion
        mobility experiment is what a per-repetition acquisition has to be added back up
        into to be read the way today's files are. Straight through `sum_frames`, so it
        is the same generator, the same cache, the same progress dialog and the same
        cancel as Sum all.
        """
        if method_frame is None:
            method_frame = self._method_spin.value()
        members = self._grouping.frames.get(int(method_frame), ())
        if not members:
            return
        self.sum_frames(list(members), what=f"method frame {int(method_frame)}")

    def sum_frames(self, frames: "list[int] | None" = None, *, what: str = "") -> None:
        """Sum several frames into one image, with a progress dialog and cancel.

        Defaults to the frame-type filter's active set rather than literally every
        frame: "sum all" should respect whatever the filter has already narrowed the
        frame spinner to, not silently pull in a frame type the user just excluded.

        `what` names the set for the status line, so that a method-frame sum and a
        sum-all are told apart afterwards by the message they leave rather than only by
        the frame count. The work either way is `mainspring.uimf.frame.sum_frames` on
        the load worker's thread, read-bound at about 5 ms a frame -- so a hundred-frame
        method frame is under a second and every frame of a five-thousand-frame
        acquisition is twenty seconds, which is what the progress dialog and the cancel
        are for (lab record, task 17).
        """
        numbers = list(frames) if frames is not None else list(self._active_frame_numbers)
        if not numbers or self._global is None:
            return
        dialog = QProgressDialog(
            f"Summing {len(numbers)} frames...", "Cancel", 0, len(numbers), self
        )
        describe(dialog, "Summing the frames being added up. Cancel keeps the "
                 "frame already on screen.")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.canceled.connect(self._worker.cancel_sum)
        dialog.canceled.connect(dialog.close)
        self._sum_dialog = dialog
        self._worker.sum_all(numbers, what)

    # --- following a file the instrument is still writing ------------------------------

    def _on_follow_toggled(self, checked: bool) -> None:
        """Start or stop the load worker's poll, refusing what cannot be followed.

        **Not a persisted setting**, and the only toolbar toggle that is not. Every
        other one describes how the user likes to look at data; this one says something
        about one file, and a viewer that started polling every acquisition anyone ever
        opened -- years of finished runs on a group drive -- would be doing lock traffic
        against nothing on the strength of a tick set once.

        Two refusals, both with the tick put back rather than a dialog. Nothing open is
        self-evident. A file that is not on a local drive is not: following means
        reading a WAL database while another process writes it, and WAL coordinates the
        two through shared memory that only exists when both are on the same machine
        (`mainspring.uimf.is_local_path`). The file still opens and still reads; it is
        only the poll that is refused, and the status bar says which of the two it was.
        """
        if not checked:
            self._follow_mode.setEnabled(False)
            self._live = None
            self._live_sum_frames = ()
            self._worker.set_follow(False)
            if self._path is not None:
                self._show_status(
                    f"Stopped following {os.path.basename(self._path)}"
                )
            return
        refusal = None
        if self._path is None or self._global is None:
            refusal = "Open a file before following it"
        elif not is_local_path(self._path):
            refusal = ("This file is not on a local drive, and a file being written can"
                       " only be followed from the machine writing it")
        if refusal is not None:
            self._follow_action.blockSignals(True)
            self._follow_action.setChecked(False)
            self._follow_action.blockSignals(False)
            self._show_status(refusal)
            return
        self._follow_mode.setEnabled(True)
        self._live_sum_frames = ()
        self._worker.set_follow(True)
        self._show_status(f"Following {os.path.basename(self._path)}")

    def stop_following(self) -> None:
        """Turn Follow off, if it is on, by the same route the user would. Idempotent."""
        if self._follow_action.isChecked():
            self._follow_action.setChecked(False)  # its `toggled` does the rest

    @property
    def following(self) -> bool:
        """Whether the load worker is polling the open file."""
        return bool(self._follow_action.isChecked())

    def _populate_follow_modes(self) -> None:
        """Offer `Method frame sum` only on a file that says how its frames group.

        Removed rather than disabled, for the reason the method-frame spinners are
        hidden rather than greyed: an entry that can never be chosen on this file is not
        a choice. The current selection survives if it is still offered and falls back
        to `Fixed frame` if it is not.
        """
        wanted = list(FOLLOW_MODES) if self._grouping.grouped else [FOLLOW_FIXED, FOLLOW_NEWEST]
        if [self._follow_mode.itemText(i) for i in range(self._follow_mode.count())] == wanted:
            return
        chosen = self._follow_mode.currentText()
        self._follow_mode.blockSignals(True)
        self._follow_mode.clear()
        self._follow_mode.addItems(wanted)
        self._follow_mode.setCurrentText(chosen if chosen in wanted else FOLLOW_FIXED)
        self._follow_mode.blockSignals(False)

    def _on_follow_mode_changed(self, text: str) -> None:
        """Act on what the last poll already found, rather than waiting for the next.

        Switching to `Newest frame` a moment after a frame landed should show that
        frame, not the one after it a second later. The running sum is re-armed the same
        way, by forgetting what it last added up.
        """
        self._live_sum_frames = ()
        if self.following and self._live is not None:
            self._act_on_live_state(self._live)

    def _on_live_update(
        self,
        state: LiveState,
        frame_types: "dict[int, int] | None",
        grouping: "FrameGrouping | None",
    ) -> None:
        """One poll's findings: grow the frame axis, then do what `Show` asks.

        The two halves are deliberately separate. The frame axis grows on every update
        whatever the mode is, because "how many frames are there now, and how many
        repetitions has this method frame got" is what a watcher wants even while
        staying on one frame. Only the second half moves the view.

        `frame_types` and `grouping` arrive only when the frame list actually grew
        (`LoadWorker._poll`), so the common poll -- a frame still filling, nothing new
        in the list -- costs one query and no widget rebuilding.
        """
        self._live = state
        self._frame_numbers = list(state.frames)
        if frame_types is not None:
            self._frame_types = dict(frame_types)
            self._populate_type_filter()
        if grouping is not None:
            self._grouping = grouping
            self._show_grouping_controls(grouping.grouped)
            self._populate_follow_modes()
            if grouping.grouped:
                numbers = grouping.method_frame_numbers
                self._method_spin.blockSignals(True)
                self._method_spin.setRange(numbers[0], numbers[-1])
                self._method_spin.blockSignals(False)
        self._apply_type_filter()  # the range grows; the frame on screen stays put
        self._act_on_live_state(state)

    def _on_follow_stopped(self, message: str) -> None:
        """The poll raised and the worker gave up on it; put the toggle back.

        A file moved or unmounted mid-run fails every poll, so the alternative is an
        error a second until someone looks. The file stays open and everything already
        read stays on screen: only the watching stops.
        """
        if not self.following:
            return
        self._follow_action.blockSignals(True)
        self._follow_action.setChecked(False)
        self._follow_action.blockSignals(False)
        self._follow_mode.setEnabled(False)
        self._live = None
        self._live_sum_frames = ()
        self._show_status(f"Stopped following: {message}")

    def _act_on_live_state(self, state: LiveState) -> None:
        """The `Show` half of a poll: leave the view alone, chase it, or total it up."""
        if not state.frames:
            return
        mode = self._follow_mode.currentText()
        if mode == FOLLOW_NEWEST:
            # The newest frame, not the newest *finished* one: a frame is about a second
            # of acquisition and its scans arrive in batches, so following the finished
            # ones would show each experiment whole and one second stale instead of
            # filling. It is never cached while it is unfinished (`FrameCache.put`), so
            # every poll re-reads it and sees what has been added.
            self.show_frame(state.frames[-1])
        elif mode == FOLLOW_METHOD_SUM:
            self._sum_live_method_frame(state)

    def _sum_live_method_frame(self, state: LiveState) -> None:
        """Re-total the method frame being acquired, but only when it has gained one.

        Only the **finished** repetitions, unlike `Newest frame` above: a total that
        included a frame still being written would change every time it was recomputed
        and would settle on whatever the operator happened to stop at. And only when the
        set has changed, because a hundred-frame method frame is 0.2 s of reading (lab
        record, task 17) and re-adding the same frames once a second would keep the load
        worker busy enough to delay the frames it is waiting for.
        """
        method_frame = self._grouping.method_frame.get(state.frames[-1])
        if method_frame is None:
            return
        members = tuple(f for f in self._grouping.frames.get(method_frame, ())
                        if f not in state.provisional)
        if not members or members == self._live_sum_frames:
            return
        self._live_sum_frames = members
        # The spinners name the method frame being totalled, and the readout beside them
        # says how many repetitions of it are in yet. `_sync_grouping_controls` reads
        # that off a real frame of the file; the sum itself is frame 0 and cannot.
        self._sync_grouping_controls(members[-1])
        self._worker.sum_all(list(members), f"method frame {method_frame}", live=True)

    # --- worker callbacks -------------------------------------------------------------

    def _on_opened(
        self,
        global_params: GlobalParams,
        frame_numbers: "list[int]",
        frame_types: "dict[int, int]",
        grouping: FrameGrouping,
    ) -> None:
        self._global = global_params
        self._frame_numbers = list(frame_numbers)
        self._frame_types = dict(frame_types)
        self._grouping = grouping
        # Named as soon as the file has opened, frames or no frames: an empty file is
        # still the file on screen.
        self.setWindowTitle(f"{os.path.basename(self._path or '')} — {APP_TITLE}")
        self._populate_type_filter()
        self._apply_detector_bits()
        self._show_grouping_controls(grouping.grouped)
        self._populate_follow_modes()
        if grouping.grouped:
            numbers = grouping.method_frame_numbers
            self._method_spin.blockSignals(True)
            self._method_spin.setRange(numbers[0], numbers[-1])
            self._method_spin.blockSignals(False)
        self._active_frame_numbers = list(self._frame_numbers)
        self._frame_spin.blockSignals(True)
        if self._frame_numbers:
            self._frame_spin.setRange(min(self._frame_numbers), max(self._frame_numbers))
            self._frame_spin.setValue(self._frame_numbers[0])
        else:
            self._frame_spin.setRange(0, 0)
        self._frame_spin.blockSignals(False)
        if not frame_numbers:
            self._busy.hide()
            self._opening = False
            self._show_status("This file has no frames")
            return
        self._worker.request_frame(frame_numbers[0])

    def _populate_type_filter(self) -> None:
        """Rebuild `Data settings > Type`, and leave it alone when the types are the same.

        The early return is what makes this safe to call from the follow poll: a
        rebuilt menu ticks `All frames`, so an operator who had filtered to one frame
        type would silently be back on all of them the next time a frame arrived. A file
        whose frame types genuinely change under us does reset the filter, which is the
        honest answer to the set it was filtering having moved.

        The actions are parented to the submenu rather than to the window, so that
        removing one and deleting it is the end of it: an action parented to the window
        would outlive the rebuild and still answer `findChildren`, which is what
        `controls.unexplained` walks.
        """
        wanted = ["All frames"] + sorted({_frame_type_name(t) for t in self._frame_types.values()})
        if list(self._type_actions) == wanted:
            return
        for action in self._type_actions.values():
            self._type_group.removeAction(action)
            self._type_menu.removeAction(action)
            action.setParent(None)
        self._type_actions = {}
        for text in wanted:
            action = make_action(
                self._type_menu,
                text,
                tip="Show every frame in the file, whatever its type."
                    if text == "All frames"
                    else f"Show only the file's {text} frames, in the frame spinner and"
                         " in Sum all.",
                checkable=True,
                checked=(text == "All frames"),
                toggled=lambda checked, t=text: self._on_type_filter_changed(t, checked),
            )
            self._type_group.addAction(action)
            self._type_menu.addAction(action)
            self._type_actions[text] = action

    def _on_frame_loaded(
        self, frame_number: int, sparse_frame: SparseFrame, frame_params: FrameParams
    ) -> None:
        # `_opening` is true only for the very first frame after `open_file` -- every
        # later arrival of this same signal is a frame-navigation or filter-driven
        # request, which must keep the view regardless of keep-ranges.
        reset = (not self.settings.keep_ranges) if self._opening else False
        self._show_frame(frame_number, sparse_frame, frame_params, reset=reset)

    def _show_frame(
        self,
        frame_number: int,
        sparse_frame: SparseFrame,
        frame_params: FrameParams,
        reset: bool,
        message: "str | None" = None,
    ) -> None:
        assert self._global is not None  # a frame cannot load before opened() fires
        calibration = frame_params.calibration(self._global.bin_width_ns)
        axes = DisplayAxes.build(
            sparse_frame, calibration, frame_params.average_tof_length_ns,
            raw_units=self.settings.raw_units, swapped=self.settings.swap_axes,
            t0_offset_ms=self.settings.arrival_offset_ms,
        )
        self._current_frame = sparse_frame
        self._current_axes = axes
        self._current_frame_number = frame_number
        self._frame_params = frame_params
        self._serial += 1
        self.info_panel.set_file(self._global, frame_params)
        # A frame the instrument may still be adding scans to is drawn, and said to be
        # unfinished in both places a user looks: the status line under it and the panel
        # beside it. Drawing it is the point -- an experiment filling in front of the
        # operator is the live view worth having -- but a partial frame quoted as a
        # finished one is a number that will be wrong by the time it is written down.
        self.info_panel.set_frame_state(sparse_frame.provisional)
        unfinished = " -- still being written" if sparse_frame.provisional else ""
        base = message or f"Frame {frame_number}: {len(sparse_frame):,} points"
        self._frame_message = base + unfinished
        self._show_status(self._frame_message)
        if frame_number in self._frame_numbers:
            self._frame_spin.blockSignals(True)
            self._frame_spin.setValue(frame_number)
            self._frame_spin.blockSignals(False)
        self._sync_grouping_controls(frame_number)
        # Sets the reset target and the gesture limits, then asks for the first render;
        # every later render comes from a gesture through the same signal.
        self.heatmap.set_frame_extent(axes, reset=reset)

    def _on_failed(self, message: str) -> None:
        self._busy.hide()
        if self._opening:
            # The open itself failed: nothing is on screen for the title to name.
            failed_path = self._path
            command_line_open = self._opened_from_command_line
            self._path = None
            self._opened_from_command_line = False
            self.setWindowTitle(APP_TITLE)
            if command_line_open:
                # Explorer launched this process *because of* the file -- a double-click
                # on a corrupt file, a vanished network share, or something merely named
                # .uimf -- so a status-bar line alone reads as "the program is broken"
                # rather than as a bad file. A deliberate File > Open leaves the window
                # already in front of the user; that case still gets the status bar only.
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Icon.Warning)
                box.setWindowTitle(APP_TITLE)
                box.setText(f"Could not open {os.path.basename(failed_path or '')}")
                box.setInformativeText(message)
                box.exec()
        self._opening = False
        if self._sum_dialog is not None:
            self._sum_dialog.close()
            self._sum_dialog = None
        self._show_status(f"Error: {message}")

    def _on_view_resized(self, width: int, height: int) -> None:
        self._request_render(*self.heatmap.view_range(), width, height)

    def _on_view_changed(self, x_range: "tuple[float, float]",
                         y_range: "tuple[float, float]") -> None:
        self._request_render(x_range, y_range)

    def _request_render(
        self,
        x_range: "tuple[float, float]",
        y_range: "tuple[float, float]",
        width: "int | None" = None,
        height: "int | None" = None,
    ) -> None:
        if self._current_frame is None or self._current_axes is None:
            return
        if width is None or height is None:
            width, height = self.heatmap.pixel_size()
        self._mailbox.put(RenderRequest(
            frame=self._current_frame,
            axes=self._current_axes,
            x_range=x_range,
            y_range=y_range,
            width=width,
            height=height,
            aggregate=self.settings.aggregate,
            serial=self._serial,
        ))

    def _on_rendered(self, render: object) -> None:
        if render.serial != self._serial:
            return  # a frame or a file has moved on since this was asked for
        result = render.result
        self._last_render = render
        self.heatmap.set_image(result, colour_scale=self.settings.colour_scale)
        self.side_plots.set_profiles(render.x_profile, render.y_profile)
        self._show_peaks(render)
        self.heatmap.set_debug_text(
            f"{render.elapsed_ms:.1f} ms  {result.image.shape[1]}x{result.image.shape[0]}"
            f"  {result.points_in_view} pts"
        )
        if self._frame_params is not None:
            self.info_panel.set_view(
                result,
                self._frame_params.accumulations,
                self.detector_bits,
                from_file=self.detector_bits_from_file,
            )
        self.export_png_action.setEnabled(True)
        self.export_pdf_action.setEnabled(True)
        if self._opening:
            # An explicit flag and not the progress bar's visibility: a window that has
            # not been shown yet -- every pytest-qt test that does not call `show()` --
            # has no visible children to ask.
            self._opening = False
            self._busy.hide()
            elapsed_ms = (time.perf_counter() - self._open_started) * 1000.0
            self._show_status(
                f"{self._frame_message} -- opened in {elapsed_ms:.0f} ms"
            )
        self.frame_shown.emit(result)

    def _on_summing_progress(self, done: int, total: int) -> None:
        if self._sum_dialog is not None:
            self._sum_dialog.setValue(done)

    def _on_summed(
        self,
        sparse_frame: "SparseFrame | None",
        frame_params: "FrameParams | None",
        request: SumRequest,
    ) -> None:
        """Draw a finished sum, worded by the ask that produced it.

        The ask comes back with the answer rather than being remembered here, because
        the follow poll can have started a running total while the user's own `Sum all`
        was still going: two answers arrive, and which dialog to close and how to word
        the status line is a property of each, not of the window.
        """
        if not request.live and self._sum_dialog is not None:
            self._sum_dialog.close()
            self._sum_dialog = None
        if frame_params is None:
            return  # cancelled, or an empty frame list -- nothing to show
        if request.live and not self._follow_action.isChecked():
            return  # following was switched off while this was being added up
        named = f" of {request.what}" if request.what else ""
        if request.live:
            count = len(request.frames)
            message = (f"Running sum{named}: {count} repetition{'' if count == 1 else 's'}"
                       f" so far, {len(sparse_frame):,} points")
        else:
            message = (f"Sum of {len(request.frames)} frames{named}:"
                       f" {len(sparse_frame):,} points")
        self._show_frame(0, sparse_frame, frame_params, reset=False, message=message)

    # --- control callbacks --------------------------------------------------------------

    def _on_aggregate_changed(self, name: str, checked: bool) -> None:
        if not checked:
            return  # the exclusive group also reports the entry it is unticking
        self.settings.aggregate = name
        self._request_render(*self.heatmap.view_range())

    def _on_colour_scale_changed(self, name: str, checked: bool) -> None:
        # Same as `_on_colour_map_changed`: the exclusive group also fires this for the
        # entry it is unchecking, and only the newly-checked one is the change.
        if not checked:
            return
        self.settings.colour_scale = name
        if self._last_render is not None:
            self.heatmap.set_image(self._last_render.result, colour_scale=self.settings.colour_scale)

    def _on_swap_toggled(self, checked: bool) -> None:
        self.settings.swap_axes = checked
        self._rebuild_axes()

    def _on_raw_units_toggled(self, checked: bool) -> None:
        self.settings.raw_units = checked
        self._rebuild_axes()

    def _on_arrival_offset_changed(self, value: float) -> None:
        self.settings.arrival_offset_ms = float(value)
        self._rebuild_axes()

    def _on_keep_ranges_toggled(self, checked: bool) -> None:
        self.settings.keep_ranges = checked

    def _on_keep_levels_toggled(self, checked: bool) -> None:
        self.settings.keep_levels = checked
        if checked:
            self.heatmap.set_levels(*self.heatmap.levels())
        else:
            self.heatmap.release_levels()

    def _on_colour_map_changed(self, name: str, checked: bool) -> None:
        # The exclusive `QActionGroup` also toggles the previously-checked entry off,
        # which fires this same handler with `checked=False` -- only the newly-checked
        # one is the change to act on.
        if not checked:
            return
        self.settings.colour_map = name
        self.heatmap.set_colour_map(name)

    def _on_light_mode_toggled(self, checked: bool) -> None:
        """Repaint the canvas and nothing else.

        No re-render and no reload: `theme.apply` sets colours on the items already on
        screen, so the open file, the frame, the view ranges and the colour levels are
        all still exactly where the user left them (`theme.py`).
        """
        self.settings.theme = "light" if checked else "dark"
        theme.apply(self, self.settings.theme)

    def _on_text_scale_changed(self, scale: float, checked: bool) -> None:
        """Resize every piece of text, and nothing else.

        The counterpart of `_on_light_mode_toggled`, and the same promise: no re-render
        and no reload, so the open file, the frame, the view ranges and the colour levels
        are all still where the user left them (`fonts.py`).
        """
        if not checked:
            return
        self.settings.text_scale = float(scale)
        fonts.apply(self, self.settings.text_scale)

    def _on_info_toggled(self, checked: bool) -> None:
        self.settings.show_info_panel = checked

    def _on_detector_bits_changed(self, value: int) -> None:
        self.settings.detector_bits = int(value)
        self._refresh_view_readouts()

    @property
    def detector_bits(self) -> int:
        """The bit depth the per-push readout is dividing by, from wherever it came.

        The file's when the file stores one, the setting's otherwise. Bit depth has no
        name in PNNL's parameter set and so is a setting on every file PNNL's writers
        produce; `mainspring.uimf.writer` stores it, so a clockwork acquisition carries
        its digitizer's own depth and no longer depends on the user having set the right
        number (lab record, tasks 16 and 17).
        """
        stored = self._global.detector_bits if self._global is not None else None
        return int(stored) if stored else int(self.settings.detector_bits)

    @property
    def detector_bits_from_file(self) -> bool:
        """Whether `detector_bits` came out of the open file rather than the setting."""
        return bool(self._global is not None and self._global.detector_bits)

    def _apply_detector_bits(self) -> None:
        """Put the open file's stored bit depth in the menu's spin box, and lock it there.

        A spin box the user can turn while the number in use comes from somewhere else
        would be a control that lies. So on a file that stores a depth the box shows it
        and is read-only, and on a file that does not it goes back to showing the
        setting and is editable again. Signals blocked either way: a programmatic set
        must not write the file's value into the user's persisted setting, which they
        would then carry to the next file.
        """
        from_file = self.detector_bits_from_file
        self._bits_box.blockSignals(True)
        self._bits_box.setValue(self.detector_bits)
        self._bits_box.blockSignals(False)
        self._bits_box.setReadOnly(from_file)
        self._bits_box.setButtonSymbols(
            QSpinBox.ButtonSymbols.NoButtons if from_file else QSpinBox.ButtonSymbols.UpDownArrows
        )
        describe(
            self._bits_box,
            "Detector bit depth that the per-push readout assumes. This file stores its"
            " own, so it cannot be changed here."
            if from_file
            else "Detector bit depth, 1 to 32, that the per-push readout assumes.",
        )
        describe(self._bits_label, self._bits_box.toolTip())

    def _refresh_view_readouts(self) -> None:
        """Re-send the newest render's in-view numbers to the info panel.

        For the changes that alter what those numbers *mean* without a new render: the
        bit-depth setting, and opening a file whose stored depth replaces it.
        """
        if self._last_render is not None and self._frame_params is not None:
            self.info_panel.set_view(
                self._last_render.result,
                self._frame_params.accumulations,
                self.detector_bits,
                from_file=self.detector_bits_from_file,
            )

    def _on_type_filter_changed(self, text: str, checked: bool) -> None:
        if not checked:
            return  # the exclusive group also reports the entry it is unticking
        moved = self._apply_type_filter()
        if moved is not None:
            self.show_frame(moved)

    def _apply_type_filter(self) -> "int | None":
        """Recompute which frames the spinner may reach, and say which to move to.

        Split out of the filter's own handler because the frame list can change without
        the filter having been touched: a followed acquisition grows it once a second,
        and the spinner's range has to grow with it. Returns the frame to move to, or
        None to stay put, rather than moving itself -- so the follow poll can decide for
        itself whether a filter change is a reason to leave the frame on screen.
        """
        checked = self._type_group.checkedAction()
        text = checked.text() if checked is not None else ""
        if not text or text == "All frames":
            self._active_frame_numbers = list(self._frame_numbers)
        else:
            self._active_frame_numbers = [
                n for n in self._frame_numbers if _frame_type_name(self._frame_types.get(n, 0)) == text
            ]
        if not self._active_frame_numbers:
            return None
        self._frame_spin.blockSignals(True)
        self._frame_spin.setRange(min(self._active_frame_numbers), max(self._active_frame_numbers))
        self._frame_spin.blockSignals(False)
        if self._current_frame_number in self._active_frame_numbers:
            return None
        return _nearest(self._active_frame_numbers, self._current_frame_number or 0)

    def _on_frame_spin_changed(self, value: int) -> None:
        if not self._active_frame_numbers:
            return
        nearest = _nearest(self._active_frame_numbers, value)
        if nearest != value:
            self._frame_spin.blockSignals(True)
            self._frame_spin.setValue(nearest)
            self._frame_spin.blockSignals(False)
        if nearest != self._current_frame_number:
            self.show_frame(nearest)

    def _on_method_spin_changed(self, value: int) -> None:
        self.show_method_frame(value)

    def _on_repetition_spin_changed(self, value: int) -> None:
        self.show_method_frame(self._method_spin.value(), repetition=value)

    def _rebuild_axes(self) -> None:
        """Rebuild `DisplayAxes` for the swap or raw-units toggle, translating the
        visible region through the old and new axis tables rather than resetting it.

        A display coordinate's fractional source-element index is recovered by
        interpolating into the *old* edge table and re-expressed through the *new*
        one -- both are monotonic by construction, the calibration clamped so that it
        never decreases and a raw index table trivially increasing -- so the same bins
        and scans stay in view across the toggle, only relabelled or swapped.
        """
        if self._current_frame is None or self._current_axes is None or self._frame_params is None:
            return
        assert self._global is not None
        old_axes = self._current_axes
        (x0, x1), (y0, y1) = self.heatmap.view_range()
        bin_lo, bin_hi = (y0, y1) if old_axes.swapped else (x0, x1)
        scan_lo, scan_hi = (x0, x1) if old_axes.swapped else (y0, y1)
        bin_index_lo = _index_of(old_axes.bin_edges, bin_lo)
        bin_index_hi = _index_of(old_axes.bin_edges, bin_hi)
        scan_index_lo = _index_of(old_axes.scan_edges, scan_lo)
        scan_index_hi = _index_of(old_axes.scan_edges, scan_hi)

        calibration = self._frame_params.calibration(self._global.bin_width_ns)
        new_axes = DisplayAxes.build(
            self._current_frame, calibration, self._frame_params.average_tof_length_ns,
            raw_units=self.settings.raw_units, swapped=self.settings.swap_axes,
            t0_offset_ms=self.settings.arrival_offset_ms,
        )
        new_bin_lo = _value_at(new_axes.bin_edges, bin_index_lo)
        new_bin_hi = _value_at(new_axes.bin_edges, bin_index_hi)
        new_scan_lo = _value_at(new_axes.scan_edges, scan_index_lo)
        new_scan_hi = _value_at(new_axes.scan_edges, scan_index_hi)
        if new_axes.swapped:
            new_x_range = tuple(sorted((new_scan_lo, new_scan_hi)))
            new_y_range = tuple(sorted((new_bin_lo, new_bin_hi)))
        else:
            new_x_range = tuple(sorted((new_bin_lo, new_bin_hi)))
            new_y_range = tuple(sorted((new_scan_lo, new_scan_hi)))

        self._current_axes = new_axes
        self._serial += 1
        # The box's own `set_extent`, not `HeatmapView.set_frame_extent` -- that also
        # asks for an immediate render at whatever range is still on screen, in the old
        # axes' numbers, which would be a wasted request a moment before `setRange`
        # below asks for the translated one.
        self.heatmap.view_box.set_extent(new_axes, reset=False)
        self.heatmap.view_box.setRange(xRange=new_x_range, yRange=new_y_range, padding=0.0)

    # --- cursor readout ---------------------------------------------------------------

    def _on_cursor_moved(self, x: float, y: float) -> None:
        """The status-bar readout: where the pointer is, in every unit the frame has.

        Both the display units and the raw bin and scan are shown, because they answer
        different questions -- an m/z identifies a species, a bin identifies the sample
        of the digitizer trace it came from -- and neither can be recovered from the
        other by eye. The intensity is the one **drawn at that pixel**, read back out of
        the image on screen rather than recomputed, and it is labelled with the
        aggregation that produced it: a `sum` pixel is a total over however many bins
        and scans that pixel covers, and quoting it as if it were a stored intensity is
        the mistake this label exists to prevent.
        """
        axes = self._current_axes
        if axes is None or self._last_render is None:
            return
        result = self._last_render.result
        # The plain spelling: this is a line of numbers, and the italic `m` an axis
        # label is set in would read here as an error rather than as typesetting.
        parts = [
            f"{labels.plain(axes.x_label)} {x:,.4g}",
            f"{labels.plain(axes.y_label)} {y:,.4g}",
        ]
        bin_value, scan_value = (y, x) if axes.swapped else (x, y)
        bin_index = _element_of(axes.bin_edges, bin_value)
        scan_index = _element_of(axes.scan_edges, scan_value)
        if bin_index is not None and scan_index is not None:
            parts.append(f"bin {bin_index}  scan {scan_index}")
        cell = pixel_of(result, x, y)
        if cell is not None:
            row, column = cell
            parts.append(f"{result.aggregate} {float(result.image[row, column]):,.0f}")
        self._readout.setText("   |   ".join(parts))

    def _clear_readout(self) -> None:
        self._readout.setText("")

    # --- the status bar ---------------------------------------------------------------

    def _show_status(self, text: str) -> None:
        """Say what the window is doing, in the middle zone of the status bar.

        A label rather than `QStatusBar.showMessage`. Every one of the twelve places this
        replaced was untimed, and an untimed message covers the bar's own left-hand
        widgets for the rest of the session -- which is fine when nothing is there and
        impossible once the cursor readout is (lab record, task 24).
        """
        self._status.setText(text)

    def status_text(self) -> str:
        """What the status bar is saying. The read side of `_show_status`."""
        return self._status.text()

    def _show_peaks(self, render: "object | None") -> None:
        """Where each projection peaks, on the right of the status bar.

        Named by whatever the axis currently is rather than by a quantity, because the
        swap-axes toggle moves m/z from one projection to the other and a readout that
        said `m/z` in a fixed place would be wrong half the time. The height is a sum
        over the range in view on the other axis, which is what the tooltip says.
        """
        if render is None:
            self._peaks.setText("")
            return
        axes = render.result.axes
        parts = []
        for profile, name in (
            (render.x_profile, axes.x_label),
            (render.y_profile, axes.y_label),
        ):
            found = peak_of(*profile)
            if found is not None:
                position, height = found
                parts.append(
                    f"peak {labels.plain(name)} {position:,.6g} ({height:,.0f})"
                )
        self._peaks.setText("   |   ".join(parts))

    def closeEvent(self, event) -> None:
        self.settings.window_geometry = bytes(self.saveGeometry())
        # Read off the dock itself, in case a visibility change reached it by a route
        # the action's `toggled` did not report.
        self.settings.show_info_panel = not self.info_panel.isHidden()
        # Likewise the width: a dock is resized by dragging its edge, which emits
        # nothing worth connecting to, so it is read once here. A hidden dock's width is
        # whatever it was when it was hidden, so this is right either way.
        if self.info_panel.width() > 0:
            self.settings.info_panel_width = int(self.info_panel.width())
        save_settings(self.settings)
        self._mailbox.close()
        self._render_worker.wait(2000)
        self._worker.stop()
        self._worker.wait(2000)
        super().closeEvent(event)


def _nearest(numbers: "list[int]", value: int) -> int:
    """The entry of an ascending list closest to `value`; ties go to the lower.

    Bisected rather than scanned. The frame spinner runs this on the GUI thread once
    per step, including once per tick of a held arrow, and a clockwork raw acquisition
    puts thousands of frames in the list: 0.46 ms scanned against 0.001 ms bisected at
    5,000 frames, which was over half of what a spinner step cost (lab record, task 17).
    """
    index = bisect.bisect_left(numbers, value)
    if index == 0:
        return numbers[0]
    if index == len(numbers):
        return numbers[-1]
    below, above = numbers[index - 1], numbers[index]
    return below if value - below <= above - value else above


def _element_of(edges: "np.ndarray", value: float) -> "int | None":
    """The index of the source element containing `value`, or None if it is outside.

    `searchsorted` and not a division: the edge tables are monotonic but not uniform --
    m/z is quadratic in bin -- so this is the one place in the viewer where the axis
    really does have to be searched rather than divided through (`uimf/raster.py` says
    why the rasteriser gets to divide).
    """
    index = int(np.searchsorted(edges, value, side="right")) - 1
    if 0 <= index < len(edges) - 1:
        return index
    return None


def _index_of(edges: "np.ndarray", value: float) -> float:
    """The fractional source-element index a display coordinate names, by interpolating
    into a monotonic edge table -- the inverse of building that table in the first
    place, and how `_rebuild_axes` recovers "which bins and scans" from "which pixels"."""
    return float(np.interp(value, edges, np.arange(len(edges), dtype=np.float64)))


def _value_at(edges: "np.ndarray", index: float) -> float:
    """The inverse of `_index_of`: the display coordinate a fractional index maps to."""
    return float(np.interp(index, np.arange(len(edges), dtype=np.float64), edges))
