"""The PySide6 + pyqtgraph viewer. Imports Qt; nothing under `mainspring.uimf` may
import from here.

Where things live:

    app.py           the console entry point: QApplication, then MainWindow
    main_window.py   the window that owns the file, the workers and the widgets
    heatmap.py       the image and the one-gesture zoom, pan and reset
    side_plots.py    the in-view mass spectrum and arrival-time distribution
    info_panel.py    parameters, and the intensity-per-push readout
    workers.py       the load thread, the render thread, and the single-slot mailbox
    settings.py      ViewerSettings and its QSettings persistence

Importing this package imports Qt, so nothing here is imported at `mainspring` package
level. `notes/architecture.md` in the lab record says why the seam sits where it does
(lab record, task 02).
"""
