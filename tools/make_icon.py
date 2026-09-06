"""Rasterise the icon artwork in packaging/icon/ into the multi-resolution
src/mainspring/viewer/resources/mainspring.ico that the .exe, the installer and the
viewer's own window all use.

Run:  uv run tools/make_icon.py            # rewrite the .ico
      uv run tools/make_icon.py --check    # exit nonzero if the .ico is out of date

Two source drawings, not one. `mainspring.svg` is the artwork as designed: an
Archimedean spiral of 2.25 turns, viridis from a yellow inner end to a deep purple
outer one, unwinding into a straight tail. Its coils are separated by about a ninth of
the width, which survives 48 px and closes into a solid disc below that, so the frames
at 32 px and under come from `mainspring-small.svg` instead -- the same spiral drawn
with 1.5 turns for wider gaps, and with the ramp's dark end lifted from #440154 to
#3e4989 so the tail does not disappear against a dark taskbar. Both are plain SVG and
either can be redrawn without touching this script.

Qt does the rendering (QSvgRenderer, already a dependency through PySide6) and this
module writes the ICO container itself, because Qt's ICO writer emits one frame per
file and an icon that Windows can pick a size from needs all of them in one. Frames at
or below 64 px are stored as 32-bit BGRA DIBs, the form every Windows version reads;
the 128 and 256 px frames are stored as PNG, as they have been since Vista, which
keeps the file to a few tens of kilobytes.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SOURCE_DIR = os.path.join(ROOT, "packaging", "icon")
TARGET = os.path.join(ROOT, "src", "mainspring", "viewer", "resources", "mainspring.ico")

# (size, source drawing). 48 px is the last size at which the designed 2.25-turn spiral
# still reads; 32 and below take the wider-gapped drawing. 24 px is here because Windows
# asks for it at 125% and 150% display scaling in Explorer's list views.
FRAMES = (
    (256, "mainspring.svg"),
    (128, "mainspring.svg"),
    (64, "mainspring.svg"),
    (48, "mainspring.svg"),
    (32, "mainspring-small.svg"),
    (24, "mainspring-small.svg"),
    (16, "mainspring-small.svg"),
)


def _render(path: str, size: int) -> "QImage":  # noqa: F821 - Qt import is deferred
    """One frame, rendered from the SVG at its own size rather than downsampled from a
    larger one, so each frame gets antialiasing computed for the pixels it actually has.
    """
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        raise SystemExit(f"not a readable SVG: {path}")
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return image


def _dib(image: "QImage") -> bytes:  # noqa: F821
    """A frame in the BMP form an ICO expects: BITMAPINFOHEADER, then bottom-up BGRA
    rows, then an AND mask.

    The header's height is doubled because the format counts the mask's rows as well as
    the image's. The mask itself is left all zeros -- "opaque everywhere" -- since the
    32-bit frames carry their own alpha and Windows honours it; the bytes are still
    written because the length in the header promises them.
    """
    width, height = image.width(), image.height()
    pixels = bytearray()
    for y in range(height - 1, -1, -1):  # bottom-up
        for x in range(width):
            pixel = image.pixel(x, y)
            alpha, red = (pixel >> 24) & 0xFF, (pixel >> 16) & 0xFF
            green, blue = (pixel >> 8) & 0xFF, pixel & 0xFF
            pixels += bytes((blue, green, red, alpha))
    mask_stride = ((width + 31) // 32) * 4  # 1 bit per pixel, rows padded to 4 bytes
    header = struct.pack(
        "<IiiHHIIiiII",
        40,  # header size
        width,
        height * 2,  # image rows + mask rows
        1,  # planes
        32,  # bits per pixel
        0,  # BI_RGB, uncompressed
        len(pixels) + mask_stride * height,
        0,
        0,
        0,
        0,
    )
    return header + bytes(pixels) + bytes(mask_stride * height)


def _png(image: "QImage") -> bytes:  # noqa: F821
    from PySide6.QtCore import QBuffer, QByteArray

    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(data)


def build() -> bytes:
    """The whole .ico as bytes."""
    frames: list[bytes] = []
    sizes: list[int] = []
    for size, source in FRAMES:
        image = _render(os.path.join(SOURCE_DIR, source), size)
        frames.append(_png(image) if size >= 128 else _dib(image))
        sizes.append(size)

    directory = struct.pack("<HHH", 0, 1, len(frames))  # reserved, type 1 = icon, count
    offset = len(directory) + 16 * len(frames)
    entries = b""
    for size, frame in zip(sizes, frames):
        entries += struct.pack(
            "<BBBBHHII",
            size if size < 256 else 0,  # 256 is stored as 0: the field is one byte
            size if size < 256 else 0,
            0,  # palette size, 0 for a direct-colour frame
            0,  # reserved
            1,  # planes
            32,  # bits per pixel
            len(frame),
            offset,
        )
        offset += len(frame)
    return directory + entries + b"".join(frames)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the .ico differs from the sources",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no display needed to rasterise
    from PySide6.QtGui import QGuiApplication

    QGuiApplication.instance() or QGuiApplication([sys.argv[0]])
    data = build()

    if args.check:
        current = open(TARGET, "rb").read() if os.path.exists(TARGET) else b""
        if current == data:
            print(f"{os.path.relpath(TARGET, ROOT)} is up to date")
            return 0
        print(
            f"{os.path.relpath(TARGET, ROOT)} is out of date -- run `uv run tools/make_icon.py`",
            file=sys.stderr,
        )
        return 1

    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    with open(TARGET, "wb") as handle:
        handle.write(data)
    print(
        f"wrote {os.path.relpath(TARGET, ROOT)}: "
        f"{len(FRAMES)} frames ({', '.join(str(s) for s, _ in FRAMES)}), {len(data) / 1024:.0f} KiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
