"""Draw `packaging/icon/social-preview.png`, the 1280x640 card GitHub shows when a link
to the repository is unfurled in a browser tab, a chat client or a search result.

Run:  uv run tools/make_social_preview.py
      uv run tools/make_social_preview.py --open    # and show it

GitHub has no API for this image and the `gh` CLI cannot set it: the file this writes
has to be uploaded by hand, under Settings > General > Social preview. It is drawn here
rather than kept as a binary nobody can regenerate, so a change of wording or of the
mark is an edit and a re-run.

The mark is `packaging/icon/mainspring.svg` itself, rendered through Qt (QSvgRenderer, a
PySide6 dependency already) at the size it is drawn, not scaled up from the .ico. The
card is the one the sibling repository `clockwork` uses -- same near-black field, same
centred stack, same viridis rule -- because the two are halves of one instrument's
software and should read as a family. What differs is what is mainspring's own: the
spiral instead of the gears, and the direction of the ramp. clockwork's mark runs dark
to light, so its rule ends on yellow; mainspring's runs yellow at the spiral's inner end
out to #440154 at the tail, so its rule begins there and descends.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SOURCE = os.path.join(ROOT, "packaging", "icon", "mainspring.svg")
TARGET = os.path.join(ROOT, "packaging", "icon", "social-preview.png")

WIDTH, HEIGHT = 1280, 640

# The ramp's two ends and the green a third of the way along from the light one, which
# is the stop that stays legible as small text on black. The deep purple is the mark's
# outer end: it is the glow behind the spiral and never a foreground colour, since at
# three pixels on a near-black card it would simply be absent.
VIRIDIS_DARK = "#440154"
VIRIDIS_GREEN = "#4ac16d"
VIRIDIS_YELLOW = "#fde725"

WORDMARK = "mainspring"
TAGLINE = "Interactive viewer and Python reader for UIMF files"
DOES = "heat map   ·   mass spectrum   ·   mobilogram   ·   live follow"

# A centred stack: the mark, the wordmark, a rule, the tagline, what it does. Centred
# rather than set beside the mark because the tagline is fifty characters and wraps
# against anything narrower than the full card. Four elements and not five: an unfurled
# card is often drawn around 500 px wide, where a fifth line of small print is a grey
# smear, and the repository's name and licence are printed beside the image anyway.
MARK_BOX = (WIDTH / 2 - 112, 46, 224, 224)
RULE_WIDTH = 300
RULE_HEIGHT = 3

# The wordmark's box, and the gaps that set everything below it. The box is 160 px for a
# 112 px font because `mainspring` has two descenders and Qt clips drawText to the
# rectangle it is given: at clockwork's 120 px -- which suffices for a word with none --
# the tails of the p and the g are sliced off at the box edge. Everything under the
# wordmark is then measured from the ink rather than given a fixed y, because a rule
# placed at a y guessed from the font size lands *on* those same two descenders.
WORDMARK_BOX = (0, 294, WIDTH, 160)
RULE_GAP = 12       # between the wordmark's lowest ink and the rule
TAGLINE_GAP = 19    # between the rule and the tagline's box
TAGLINE_HEIGHT = 56
MONO_GAP = 12       # between the tagline's box and the mono line's
MONO_HEIGHT = 36


def _font(families: list[str], size: int, weight: int, spacing: float = 0.0):
    """The first of `families` this machine actually has, at `size` pixels.

    Qt substitutes silently for a missing family, which on a card whose whole job is to
    be read would go unnoticed until it was published, so the families are tried in turn
    and a machine with none of them is an error rather than a substitution.
    """
    from PySide6.QtGui import QFont, QFontDatabase

    available = set(QFontDatabase.families())
    chosen = next((name for name in families if name in available), None)
    if chosen is None:
        raise SystemExit(f"none of these fonts is installed: {', '.join(families)}")
    font = QFont(chosen)
    font.setPixelSize(size)
    font.setWeight(QFont.Weight(weight))
    if spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return font


def draw() -> None:
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import (QColor, QFontMetricsF, QImage, QLinearGradient, QPainter,
                               QPen, QRadialGradient)
    from PySide6.QtSvg import QSvgRenderer

    image = QImage(WIDTH, HEIGHT, QImage.Format.Format_ARGB32)
    image.fill(QColor("#08090c"))  # so the file is byte-identical from run to run
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    # The card: near-black, lifted very slightly towards the ramp's dark end at the top
    # left so the mark does not sit on a flat field.
    backdrop = QLinearGradient(0, 0, WIDTH, HEIGHT)
    backdrop.setColorAt(0.0, QColor("#12151c"))
    backdrop.setColorAt(1.0, QColor("#08090c"))
    painter.fillRect(0, 0, WIDTH, HEIGHT, backdrop)

    glow = QRadialGradient(MARK_BOX[0] + MARK_BOX[2] / 2,
                           MARK_BOX[1] + MARK_BOX[3] / 2, 420)
    warm = QColor(VIRIDIS_DARK)
    warm.setAlpha(90)
    glow.setColorAt(0.0, warm)
    glow.setColorAt(1.0, QColor(0, 0, 0, 0))
    painter.fillRect(0, 0, WIDTH, HEIGHT, glow)

    renderer = QSvgRenderer(SOURCE)
    if not renderer.isValid():
        raise SystemExit(f"not a readable SVG: {SOURCE}")
    renderer.render(painter, QRectF(*MARK_BOX))

    centred = int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

    word_font = _font(["Segoe UI Semibold", "Segoe UI", "Arial"], 112, 600, -1.0)
    word_box = QRectF(*WORDMARK_BOX)
    painter.setPen(QPen(QColor("#f2f4f7")))
    painter.setFont(word_font)
    painter.drawText(word_box, centred, WORDMARK)

    # A rule in the ramp itself, under the wordmark rather than through it. Where the
    # bottom of the wordmark is has to be measured, not assumed: Qt centres the line in
    # the box by the font's height, so the baseline moves with the box, and the lowest
    # ink is a descender's tail below that baseline. `tightBoundingRect` gives the ink,
    # and the rule clears it by RULE_GAP. The gradient runs the way the spiral does,
    # outward from the yellow inner end -- the mirror of clockwork's, whose mark climbs
    # towards yellow and whose rule therefore ends there.
    metrics = QFontMetricsF(word_font)
    baseline = word_box.top() + (word_box.height() - metrics.height()) / 2 + metrics.ascent()
    rule_y = baseline + metrics.tightBoundingRect(WORDMARK).bottom() + RULE_GAP

    rule = QLinearGradient((WIDTH - RULE_WIDTH) / 2, 0, (WIDTH + RULE_WIDTH) / 2, 0)
    rule.setColorAt(0.0, QColor(VIRIDIS_YELLOW))
    rule.setColorAt(1.0, QColor(VIRIDIS_GREEN))
    painter.fillRect(QRectF((WIDTH - RULE_WIDTH) / 2, rule_y, RULE_WIDTH, RULE_HEIGHT),
                     rule)

    tagline_top = rule_y + RULE_HEIGHT + TAGLINE_GAP
    painter.setPen(QPen(QColor("#c3ccd7")))
    painter.setFont(_font(["Segoe UI", "Arial"], 38, 400))
    painter.drawText(QRectF(0, tagline_top, WIDTH, TAGLINE_HEIGHT), centred, TAGLINE)

    painter.setPen(QPen(QColor(VIRIDIS_GREEN)))
    painter.setFont(_font(["Cascadia Mono", "Consolas", "Courier New"], 24, 400, 0.6))
    painter.drawText(QRectF(0, tagline_top + TAGLINE_HEIGHT + MONO_GAP, WIDTH,
                            MONO_HEIGHT), centred, DOES)

    painter.end()
    if not image.save(TARGET, "PNG"):
        raise SystemExit(f"could not write {TARGET}")
    print(f"{os.path.relpath(TARGET, ROOT)}  {WIDTH}x{HEIGHT}  "
          f"{os.path.getsize(TARGET) / 1024:.0f} KiB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true",
                    help="open the written file in the default viewer")
    args = ap.parse_args()

    # Qt needs a QGuiApplication before a font database exists. The platform plugin is
    # left to Qt rather than forced to `offscreen`: that plugin carries no font database
    # at all, and a card drawn under it comes out with every glyph a tofu box and no
    # error to say so. The check below is what refuses to write one.
    from PySide6.QtGui import QFontDatabase, QGuiApplication

    app = QGuiApplication.instance() or QGuiApplication([])
    if not QFontDatabase.families():
        raise SystemExit(
            "Qt loaded no fonts under the "
            f"{os.environ.get('QT_QPA_PLATFORM', 'default')} platform plugin, so every "
            "glyph would be drawn as an empty box; unset QT_QPA_PLATFORM and re-run")
    draw()
    del app

    if args.open:
        os.startfile(TARGET)  # noqa: S606 - Windows, and the path is this module's own
    return 0


if __name__ == "__main__":
    sys.exit(main())
