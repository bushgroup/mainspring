"""The viewer, end to end, offscreen: window constructs, loads a file, paints a frame.

pytest-qt runs these against `QT_QPA_PLATFORM=offscreen` (`conftest.py`), so no display
is needed -- a bare clone runs the whole suite on the synthetic fixture alone, same as
everywhere else in this repo. Loading is asynchronous (`LoadWorker`'s own thread), so
every test here waits on `MainWindow.frame_shown` rather than asserting immediately
after `open_file`.
"""

from __future__ import annotations

import numpy as np

from mainspring.viewer.main_window import MainWindow


def test_window_loads_the_synthetic_file_and_places_the_image(qtbot, synthetic_uimf):
    window = MainWindow()
    qtbot.addWidget(window)

    with qtbot.waitSignal(window.frame_shown, timeout=5000) as blocker:
        window.open_file(synthetic_uimf.path)
    result = blocker.args[0]

    assert result.axes.x_label == "m/z"
    assert result.axes.y_label == "Arrival time (ms)"
    # The image is placed by `setRect` over exactly the full calibrated range -- the
    # rasteriser was asked for that range, so this is also a check that MainWindow wired
    # frame, axes and worker together correctly rather than a tautology about rasterise().
    assert (result.x_range, result.y_range) == result.axes.full_range
    assert result.points_in_view == synthetic_uimf.points(1)
    assert result.tic_in_view == synthetic_uimf.tic(1)
    assert result.image.shape[0] > 0 and result.image.shape[1] > 0


def test_a_file_with_no_frames_fails_without_a_frame_paint(qtbot, tmp_path):
    import sqlite3

    path = tmp_path / "empty.uimf"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE Global_Parameters (Bins INTEGER, BinWidth DOUBLE,"
                 " TOFIntensityType TEXT, NumFrames INTEGER, TimeOffset INTEGER)")
    conn.execute("INSERT INTO Global_Parameters VALUES (4096, 1.0, 'ADC', 0, 0)")
    conn.execute("CREATE TABLE Frame_Parameters (FrameNum INTEGER PRIMARY KEY,"
                 " AverageTOFLength DOUBLE NOT NULL)")
    conn.commit()
    conn.close()

    window = MainWindow()
    qtbot.addWidget(window)

    window.open_file(str(path))
    qtbot.waitUntil(
        lambda: "no frames" in (window.statusBar().currentMessage() or "").lower(),
        timeout=5000,
    )
    # It opened, so it is the file on screen, frames or not.
    assert window.windowTitle() == "empty.uimf — mainspring"


def test_resizing_the_heatmap_re_rasterises_at_the_new_pixel_size(qtbot, synthetic_uimf):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    with qtbot.waitSignal(window.frame_shown, timeout=5000) as first:
        window.open_file(synthetic_uimf.path)
    before = first.args[0].image.shape

    with qtbot.waitSignal(window.frame_shown, timeout=5000) as second:
        window.resize(window.width() + 200, window.height() + 150)
    result = second.args[0]

    # Bins (4096) and scans (16) both exceed the widget's pixel size on neither axis in
    # the same way, so the column count -- capped by the viewport's width, not by the
    # frame -- is what should move when the window grows; the row count is capped by
    # the frame's 16 scans regardless of how tall the widget gets.
    width, height = window.heatmap.pixel_size()
    assert result.image.shape[1] == min(width, synthetic_uimf.bins)
    assert result.image.shape[0] == min(height, synthetic_uimf.scans)
    assert result.image.shape != before
    assert np.isclose(result.tic_in_view, synthetic_uimf.tic(1))


def test_the_title_names_the_open_file(qtbot, synthetic_uimf):
    import os

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "mainspring"

    with qtbot.waitSignal(window.frame_shown, timeout=5000):
        window.open_file(synthetic_uimf.path)

    assert window.windowTitle() == f"{os.path.basename(synthetic_uimf.path)} — mainspring"


def test_a_failed_open_leaves_the_title_bare(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)

    window.open_file(str(tmp_path / "does-not-exist.uimf"))
    qtbot.waitUntil(
        lambda: (window.statusBar().currentMessage() or "").startswith("Error"), timeout=5000
    )

    assert window.windowTitle() == "mainspring"
