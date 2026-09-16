"""Every control the user can touch is built here, and every one of them explains itself.

A tooltip is the only documentation a control carries to the person using it, and the
viewer shipped without any: `Keep ranges`, `Raw units` and `Bits:` said nothing from the
toolbar, and the one-gesture zoom contract lived in a docstring. Adding the text once
would have fixed today's toolbar; what this module adds is the reason the next control
arrives explained too.

Two halves, and the second is the one that matters:

`make_action`, `add_labelled` and `add_menu_widget` **require** the sentence -- a control
cannot be built through them without one, and `tip=None` is a waiver that has to be
typed, so it shows up in a diff and in review rather than happening by omission.

`unexplained` walks a live window and names what has no tooltip, **by widget type**, so
a control built inline or by pyqtgraph is caught as surely as one built here. It is the
one definition of the rule: `tools/check_public.py` and `tests/test_tooltips.py` both
call it, and neither carries its own idea of what counts.

The trap the walk exists to survive: a `QAction` with no tooltip of its own returns its
*text* from `toolTip()`, so an emptiness test passes for every mute action in the file.
`_action_tip` compares against the mnemonic-stripped text and treats a match as missing.

House style for the sentence: one sentence, saying what the control does, in the prose
voice the public documentation uses -- purpose first, numbers rather than adjectives, no
em dashes. The shortcut is appended by `make_action`, never typed.

`name_of` and `numbered` are how a walk over this window reports what it found, and are
public because this is not the only such walk: `theme.themed` names stray colours with
the same two, so a reader of either report reads one convention rather than two.
"""

from __future__ import annotations

from weakref import WeakSet

import pyqtgraph as pg
from PySide6.QtCore import QRect
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QStyle,
    QStyleOptionSpinBox,
    QToolBar,
    QWidget,
    QWidgetAction,
)

__all__ = [
    "DoubleSpinBox",
    "SpinBox",
    "add_labelled",
    "add_menu_widget",
    "describe",
    "is_waived",
    "make_action",
    "name_of",
    "numbered",
    "unexplained",
    "waive",
]

_WAIVED: "WeakSet[object]" = WeakSet()
"""Objects deliberately left without a tooltip. A `WeakSet` rather than a dynamic
property because a pyqtgraph `QGraphicsItem` is not a `QObject` and has no properties;
the cost is that a checker must build the window in its own process, which both callers
already do."""

_TIPPED_TYPES = (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
)
"""Widget types that must explain themselves wherever they appear."""

_TIPPED_ITEMS = (pg.ImageItem, pg.PlotItem, pg.AxisItem)
"""pyqtgraph items that must explain themselves. `ColorBarItem` is a `PlotItem`, so the
colour bar is covered without naming it; a hidden axis is skipped by `unexplained`."""

_LABEL_HOSTS = (QToolBar, QDockWidget, QStatusBar, QMenu)
"""A `QLabel` counts as a control when it sits in one of these: `Bits:` in the
`Data settings` menu, the info panel's field names and the status bar's readouts are all
things a user points at. A label anywhere else is decoration."""


def waive(obj: object) -> None:
    """Record that `obj` is deliberately left without a tooltip."""
    _WAIVED.add(obj)


def is_waived(obj: object) -> bool:
    return obj in _WAIVED


def _shortcut_suffix(shortcut: "QKeySequence | str | None") -> str:
    if shortcut is None:
        return ""
    keys = shortcut if isinstance(shortcut, QKeySequence) else QKeySequence(shortcut)
    text = keys.toString(QKeySequence.SequenceFormat.NativeText)
    return f"  ({text})" if text else ""


def describe(
    obj: object, tip: "str | None", *, shortcut: "QKeySequence | str | None" = None
) -> None:
    """Give `obj` its tooltip, and its status-bar line if it can carry one.

    The tooltip gets the shortcut appended, because that is where a user looks for it;
    the status bar gets the bare sentence, because a menu already shows the shortcut
    beside the entry. Works on a widget, an action, or a pyqtgraph `QGraphicsItem`,
    which has `setToolTip` and nothing else.
    """
    if tip is None:
        waive(obj)
        return
    obj.setToolTip(tip + _shortcut_suffix(shortcut))
    setter = getattr(obj, "setStatusTip", None)
    if setter is not None:
        setter(tip)


def make_action(
    parent: object,
    text: str,
    *,
    tip: "str | None",
    shortcut: "str | None" = None,
    checkable: bool = False,
    checked: bool = False,
    triggered: "object | None" = None,
    toggled: "object | None" = None,
) -> QAction:
    """A `QAction` that explains itself, configured before it is connected.

    The connection is made last, and only after `checked` has been applied, so that
    restoring a non-default setting cannot fire a handler while the rest of the window
    is still being built. That invariant is older than this module
    (`MainWindow._build_toolbar`) and moving the construction here is what makes it hold
    for every action rather than for the ones whose author remembered it.
    """
    action = QAction(text, parent)
    if shortcut is not None:
        action.setShortcut(shortcut)
    describe(action, tip, shortcut=shortcut)
    if checkable:
        action.setCheckable(True)
        action.setChecked(checked)
    if triggered is not None:
        action.triggered.connect(triggered)
    if toggled is not None:
        action.toggled.connect(toggled)
    return action


class _KeepsRoomForItsValue:
    """A spin box whose size hint reserves room for the value as well as the buttons.

    `QAbstractSpinBox.sizeHint` adds the widest value's text to a frame, and the style
    then carves the up and down buttons out of whatever width the widget ends up with --
    buttons whose width follows the font while that text allowance does not. At 100 per
    cent there is enough left for four digits; at 200 per cent the buttons take the whole
    control and the number is not drawn at all, which is a control that has stopped
    saying what it is set to (lab record, task 24).

    Measured rather than padded by a guess: the style is asked where the edit field would
    fall inside its own hint, and the shortfall against the widest value the range can
    show is added. Mixed in rather than written twice, because `QSpinBox` and
    `QDoubleSpinBox` are siblings and `textFromValue` is what makes the arithmetic the
    same for both -- each formats its own kind of number, decimals and all.
    """

    def sizeHint(self):
        hint = super().sizeHint()
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        option.rect = QRect(0, 0, hint.width(), hint.height())
        field = self.style().subControlRect(
            QStyle.ComplexControl.CC_SpinBox,
            option,
            QStyle.SubControl.SC_SpinBoxEditField,
            self,
        )
        # Negated, so the minus sign is allowed for whichever end of the range is longer.
        widest = -max(abs(self.minimum()), abs(self.maximum()))
        text = f"{self.prefix()}{self.textFromValue(widest)}{self.suffix()}"
        needed = self.fontMetrics().horizontalAdvance(text)
        hint.setWidth(hint.width() + max(0, needed - field.width()))
        return hint


class SpinBox(_KeepsRoomForItsValue, QSpinBox):
    """`QSpinBox`, keeping room for its value at any text size."""


class DoubleSpinBox(_KeepsRoomForItsValue, QDoubleSpinBox):
    """`QDoubleSpinBox`, keeping room for its value at any text size."""


def add_labelled(toolbar: QToolBar, label_text: str, widget: object, *, tip: str) -> QLabel:
    """Put `label_text` and `widget` on the toolbar as one thing, sharing one tooltip.

    The word is what a user points at -- a bare `Bits:` beside a spin box is the part
    that reads as a question -- so the label is not left as an untipped decoration that
    the walk would then have to make an exception for.
    """
    label = QLabel(label_text)
    describe(label, tip)
    describe(widget, tip)
    # Fixed horizontally, so a crowded toolbar reaches for its own overflow arrow rather
    # than squeezing a spin box down to its arrows. `View > Text size` is what made that
    # reachable at an ordinary window width: at 150 per cent the frame number was gone
    # while the control that showed it was still there (lab record, task 24).
    widget.setSizePolicy(
        QSizePolicy.Policy.Fixed, widget.sizePolicy().verticalPolicy()
    )
    toolbar.addWidget(label)
    toolbar.addWidget(widget)
    return label


_MENU_TEXT_INDENT = 24
"""Pixels from a menu's left edge to where an entry's text begins, past the column Qt
reserves for a tick. Not readable from a style at build time without a laid-out menu, so
it is a number here and the cost of it being a little wrong is a label a few pixels out
of line."""


def add_menu_widget(menu: QMenu, label_text: str, widget: object, *, tip: str) -> QLabel:
    """`add_labelled` for a menu: `label_text` and `widget` as one entry, one tooltip.

    A menu holds actions, so a control that is not a choice between fixed options -- a
    spin box over a range -- travels in a `QWidgetAction` whose default widget lays the
    two out side by side. All three are tipped: `unexplained` unwraps the action to the
    host, finds the spin box by type wherever it is, and now counts a label in a menu as
    a control the same way it counts one on the toolbar.

    The left margin is the width of the check column an ordinary entry's tick sits in,
    so the label starts where the entries above it start rather than against the frame.
    """
    label = QLabel(label_text)
    describe(label, tip)
    describe(widget, tip)
    host = QWidget(menu)
    describe(host, tip)
    layout = QHBoxLayout(host)
    layout.setContentsMargins(_MENU_TEXT_INDENT, 2, 8, 2)
    layout.addWidget(label)
    layout.addWidget(widget)
    action = QWidgetAction(menu)
    action.setDefaultWidget(host)
    menu.addAction(action)
    return label



def _strip_mnemonic(text: str) -> str:
    """Qt's own transform: `&&` is a literal ampersand, a lone `&` marks the next key."""
    return text.replace("&&", "\x00").replace("&", "").replace("\x00", "&")


def _action_tip(action: QAction) -> str:
    """The action's *explicit* tooltip, or `""`.

    `QAction.toolTip()` falls back to the action's text, so a mute action answers with
    its own label and passes any naive emptiness test. A tooltip that is only the label
    repeated explains nothing, so it counts as no tooltip either way.
    """
    tip = action.toolTip().strip()
    return "" if tip == _strip_mnemonic(action.text()).strip() else tip


def _in_dialog(widget: object) -> bool:
    """A transient dialog's furniture is not the window's to explain."""
    node = widget
    while node is not None:
        if isinstance(node, QDialog):
            return True
        node = node.parent()
    return False


def _is_internal(widget: object) -> bool:
    """Qt names the widgets it builds for itself `qt_*`: a dock's close button, the
    toolbar's overflow arrow. They are Qt's to label, in Qt's translations."""
    return widget.objectName().startswith("qt_")


def _hosted_label(label: QLabel) -> bool:
    node = label.parent()
    while node is not None:
        if isinstance(node, _LABEL_HOSTS):
            return True
        node = node.parent()
    return False


def _is_furniture(widget: object) -> bool:
    """A control inside another control is that control's own machinery.

    A combo box's drop-down list, a tree's header, a spin box's arrows: each is a
    `QAbstractItemView` or a `QAbstractButton` in its own right, and none of them is a
    thing the window put there. The control that contains them is walked instead, and
    is where the sentence belongs.
    """
    node = widget.parent()
    while node is not None:
        if isinstance(node, _TIPPED_TYPES):
            return True
        node = node.parent()
    return False


def _check_widget(widget: object, missing: "list[str]", seen: "set[int]") -> None:
    if widget is None or id(widget) in seen:
        return
    seen.add(id(widget))
    if is_waived(widget) or _is_internal(widget) or _in_dialog(widget):
        return
    if _is_furniture(widget):
        return
    # A toolbar button *is* its action, and Qt copies the action's tooltip onto it.
    if isinstance(widget, QAbstractButton) and widget.defaultAction() is not None:
        return
    if not widget.toolTip().strip():
        missing.append(name_of(widget))


def name_of(obj: object) -> str:
    """What to call `obj` in a checker's output: its class, and whatever names it.

    Nothing here is unique -- four axes share a class and two projections share a type --
    so a report that lists several is passed through `numbered` afterwards.
    """
    if isinstance(obj, pg.AxisItem):
        return f"AxisItem {obj.orientation!r}"  # four of them share a class name
    for attr in ("objectName", "text", "name"):
        getter = getattr(obj, attr, None)
        if callable(getter):
            try:
                value = getter()
            except TypeError:
                continue
            if isinstance(value, str) and value.strip():
                return f"{type(obj).__name__} {value.strip()!r}"
    return type(obj).__name__


def numbered(names: "list[str]") -> "list[str]":
    """Sorted, with repeats distinguished.

    Two mute controls of the same class, or two items of one type carrying the same
    stray colour, must both be reported: a report that named one of them would look
    like a single problem, and fixing it would look like fixing both.
    """
    counts: dict[str, int] = {}
    named: list[str] = []
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        named.append(name if counts[name] == 1 else f"{name} #{counts[name]}")
    return sorted(named)


def unexplained(window: object) -> "list[str]":
    """Names of everything on `window` that a user can point at and learn nothing from.

    Type-based, not registry-based: a control added tomorrow is walked because of what
    it is, so nothing has to be remembered. An empty list is the rule this repo holds
    itself to (`CLAUDE.md`), checked from `tools/check_public.py` and from the tests.
    """
    missing: list[str] = []
    seen: set[int] = set()

    for action in window.findChildren(QAction):
        if action.isSeparator() or action.menu() is not None or is_waived(action):
            continue
        # A `QWidgetAction` is the wrapper `QToolBar.addWidget` puts around a widget.
        # Its own tooltip is never shown; the widget's is, so that is what is asked for
        # -- and asking here, rather than trusting the widget walk, is what catches a
        # custom widget of a type this module has never heard of.
        if isinstance(action, QWidgetAction):
            _check_widget(action.defaultWidget(), missing, seen)
            continue
        if not _action_tip(action):
            missing.append(name_of(action))

    # `findChildren` takes one type at a time, and a widget can match several of them.
    for tipped_type in _TIPPED_TYPES:
        for widget in window.findChildren(tipped_type):
            _check_widget(widget, missing, seen)

    for label in window.findChildren(QLabel):
        if _hosted_label(label):
            _check_widget(label, missing, seen)

    for view in window.findChildren(QGraphicsView):
        scene = view.scene()
        if scene is None:
            continue
        for item in scene.items():
            if not isinstance(item, _TIPPED_ITEMS) or is_waived(item):
                continue
            if isinstance(item, pg.AxisItem) and not item.isVisible():
                continue  # the side plots hide theirs; there is nothing to point at
            if not item.toolTip().strip():
                missing.append(name_of(item))

    return numbered(missing)
