"""What the acquisition software and this viewer say to each other about a run.

Two things, and both are between programs rather than between a program and a person:
the words another program types to start the viewer on a run (`OPTION_FOLLOW`,
`OPTION_SHOW`, `SHOW_WORDS`), and the pointer it publishes so that a viewer which is
*already open* can find the run in progress without anyone typing a path
(`write_live_pointer` and the four names around it, lab record, task 32).

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

import json
import os
from dataclasses import dataclass
from datetime import datetime

__all__ = [
    "LIVE_POINTER_ENV",
    "LIVE_POINTER_NAME",
    "LIVE_POINTER_VERSION",
    "LivePointer",
    "OPTION_FOLLOW",
    "OPTION_SHOW",
    "SHOW_WORDS",
    "clear_live_pointer",
    "live_pointer_path",
    "read_live_pointer",
    "write_live_pointer",
]

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


LIVE_POINTER_NAME = "live-run.json"
"""The file the acquisition software publishes the run in progress in.

A convention between two programs, so it is pinned as a string the way the summed
companion's suffix is: renaming the constant costs nothing, renaming the string leaves a
viewer watching a folder nobody writes to. `tools/check_public.py` compares the literal.
"""

LIVE_POINTER_ENV = "MAINSPRING_LIVE_POINTER"
"""An environment variable naming the pointer file outright, overriding
`live_pointer_path`. For a test that must not touch the real one, and for a bench where
two viewers are being compared side by side; not something an operator sets."""

LIVE_POINTER_VERSION = 1
"""What `write_live_pointer` stamps into `version`. A reader that finds a version it does
not know returns `None` rather than guessing at fields it has never seen, which is what
makes the next schema safe to invent."""


@dataclass(frozen=True)
class LivePointer:
    """A run in progress, as the program acquiring it published it.

    `path` is the whole of the behaviour: it is the file to follow. `writer` and
    `started` drive nothing and exist to be read by a person -- an operator asking why
    the viewer moved to this file, or why it did not, has one small text file to look at
    and it says which program left it there and when.
    """

    path: str
    writer: str = ""
    started: str = ""


def live_pointer_path() -> str:
    """Where the pointer lives when nothing overrides it.

    Per user rather than per machine, under `%LOCALAPPDATA%`, because the two programs
    are two windows one operator has open: the acquisition software writes it as the
    person running the instrument and the viewer reads it as the same person.
    `mainspring/` beside the numba cache the viewer already keeps there.
    """
    override = os.environ.get(LIVE_POINTER_ENV)
    if override:
        return override
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    return os.path.join(base, "mainspring", LIVE_POINTER_NAME)


def read_live_pointer(path: "str | None" = None) -> "LivePointer | None":
    """The run in progress, or `None`.

    **`None` for every kind of absence**, and there are more of them than there are
    kinds of presence: no file, a directory where the file should be, a file being
    rewritten as it is read, one that is not JSON, one whose version this does not know,
    one with no `path` in it, one whose `path` is blank or is not a string. None of
    those is a fault worth raising through: they are all "nothing is being acquired",
    and a viewer that raised on a half-written pointer would be a viewer that raised
    once every run, since the write is what makes the file exist.

    Nothing here says whether the file named is worth following. Whether it exists, is
    on this machine and was written recently enough to be a run rather than a crash is
    the caller's question, because the answer is a policy rather than a schema
    (`viewer/main_window.py`).
    """
    try:
        with open(path or live_pointer_path(), encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("version") != LIVE_POINTER_VERSION:
        return None
    named = record.get("path")
    if not isinstance(named, str) or not named.strip():
        return None
    writer = record.get("writer")
    started = record.get("started")
    return LivePointer(
        path=named,
        writer=writer if isinstance(writer, str) else "",
        started=started if isinstance(started, str) else "",
    )


def write_live_pointer(
    run: str,
    *,
    writer: str = "",
    started: "str | None" = None,
    path: "str | None" = None,
) -> str:
    """Publish `run` as the file being acquired, and return where it was published.

    mainspring never calls this -- it reads the pointer, it does not write one. It is
    here so that the program that does write one imports the schema rather than retyping
    it, which is the whole reason this module exists (lab record, task 32).

    **Written and then renamed into place**, so a reader never sees half a file: a
    pointer is read once a couple of seconds by a viewer that may be polling while the
    run starts, and `os.replace` is atomic on both Windows and POSIX. The temporary file
    is in the same directory, since a rename across filesystems is not.
    """
    target = path or live_pointer_path()
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    record = {
        "version": LIVE_POINTER_VERSION,
        "path": os.path.abspath(run),
        "writer": writer,
        "started": started if started is not None
        else datetime.now().isoformat(timespec="seconds"),
    }
    partial = f"{target}.partial"
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    os.replace(partial, target)
    return target


def clear_live_pointer(path: "str | None" = None) -> None:
    """Take the pointer away, a run having ended. Idempotent, and never raises.

    The other half of `write_live_pointer`, and the half that belongs in a `finally`:
    the pointer says a run is in progress, so a run that ends without removing it leaves
    a viewer being told about an acquisition that is over. Nothing here raises, because
    a failure to remove it is not a reason for a run to end badly -- and the viewer
    guards against a pointer nobody took away in any case.
    """
    try:
        os.remove(path or live_pointer_path())
    except OSError:
        pass
