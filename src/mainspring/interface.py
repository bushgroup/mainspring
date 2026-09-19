"""The option names and words another program types to start this viewer on a run.

Qt-free and top level, which is the whole point. The acquisition software starts
mainspring on the file it is still writing (`viewer/app.py`), so these are an interface
between two programs rather than labels in a window. That caller's own layering forbids
the module which builds the command line from importing Qt, so the words cannot live in
`viewer/main_window.py` beside the `Show` box's labels: a caller that cannot import them
retypes them, and four string literals in two repositories are four chances to disagree.

So the words are here, the labels stay in the window, and `main_window.FOLLOW_MODE_WORDS`
maps one onto the other. A mode added to the window without a word here is a missing
entry in that one dict; a word renamed here is a command line that stops working
somewhere this repository cannot see, which is why renaming one is a version bump rather
than an edit (lab record, task 27).
"""

from __future__ import annotations

__all__ = ["OPTION_FOLLOW", "OPTION_SHOW", "SHOW_WORDS"]

OPTION_FOLLOW = "--follow"
"""Watch the file for what the instrument is still writing to it."""

OPTION_SHOW = "--show"
"""What following does with each new frame. Implies `OPTION_FOLLOW`."""

SHOW_WORDS = ("fixed", "newest", "method-sum", "rolling-sum")
"""The values `OPTION_SHOW` takes, in the order the `Show` box offers them.

Words a person could have typed rather than the labels this window happens to say today
-- `rolling-sum`, not `Sum newest frames`. `main_window.FOLLOW_MODE_WORDS` is keyed by
these and builds itself from them, and `tools/check_public.py` fails if the two ever
come apart.
"""
