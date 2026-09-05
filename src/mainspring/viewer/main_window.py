"""`MainWindow`: the one object that knows about all the others.

It owns the file, the settings, the two workers, and the four widgets, and it is the
only place where they meet -- the heatmap does not know there is an info panel, the
side plots do not know there is a file. That is what keeps each of the later tasks to
one or two modules.

The flow it coordinates, once around:

    open  ->  LoadWorker decodes a frame  ->  build DisplayAxes  ->  choose the view
          ->  RenderMailbox  ->  RenderWorker  ->  image + profiles + readouts

**Choosing the view is where keep-ranges lives.** On opening a file the window either
takes the frame's full range or keeps the ranges already on screen, depending on the
setting, and that decision belongs here rather than in the heatmap because it is a
property of the session and not of the widget.

Frame navigation, the toolbar toggles, and "sum all frames" with its progress and
cancel are also here, since each of them changes what is asked of the workers rather
than how a widget draws.
"""

from __future__ import annotations

__all__ = ["MainWindow"]


class MainWindow:
    """The application window. A `QMainWindow` once it exists.

    Arrives with the lab record's task 04, and grows through tasks 05 and 06.
    """

    def __init__(self, settings: object = None) -> None:
        raise NotImplementedError("the main window arrives with the lab record's task 04")

    def open_file(self, path: str) -> None:
        """Open a UIMF file and show its first frame. Arrives with the lab record's task 04."""
        raise NotImplementedError("the main window arrives with the lab record's task 04")

    def show_frame(self, frame: int) -> None:
        """Switch to a frame, keeping the current view. Arrives with the lab record's task 06."""
        raise NotImplementedError("frame navigation arrives with the lab record's task 06")

    def sum_frames(self, frames: "list[int] | None" = None) -> None:
        """Sum several frames into one image, with progress and cancel. Arrives with task 06."""
        raise NotImplementedError("summing frames arrives with the lab record's task 06")
