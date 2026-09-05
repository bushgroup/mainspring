"""The `Frame_Scans.Intensities` blob: run-length-zero over LZF, both ways.

A stored spectrum is a **run-length-zero (RLZ)** stream of fixed-width little-endian
numbers, **LZF**-compressed. The element type comes from the file's
`TOFIntensityType`: `ADC` is int32, `TDC` int16, `FOLDED` float32 (lab record,
task 01, verified on four files).

RLZ walks a bin counter `b` from 0: a value `v < 0` skips `-v` bins, a value `v > 0`
is the intensity at bin `b` and then `b += 1`, and a value `v == 0` advances one bin
without emitting a point. Writers do emit those explicit zeros -- thousands per file --
so `NonZeroCount` is an upper bound on the number of points and a safe preallocation
size, never an exact count.

LZF is Marc Lehmann's liblzf as PNNL's `CLZF2.cs` carries it: a control byte `c` under
32 introduces `c + 1` literal bytes, and `c >= 32` a back-reference of length
`(c >> 5) + 2` -- with one extra length byte when `c >> 5 == 7` -- at offset
`((c & 31) << 8 | next) + 1` behind the output cursor. Back-references may overlap the
bytes they are still producing, so a copy is byte-at-a-time.

**The encoder is here, implemented, and the decoder is not.** Only tests write UIMF
files, so the encoder is small pure Python and stays that way; the decoder is on the
viewer's critical path and arrives with its numba kernel and pure fallback in the lab
record's task 03. Writing the encoder first is what lets a synthetic file exist before
anything can read one.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "INTENSITY_DTYPES",
    "decode_intensities",
    "dtype_for",
    "encode_intensities",
    "lzf_compress",
    "lzf_decompress",
    "rlz_decode",
    "rlz_encode",
]

INTENSITY_DTYPES: dict[str, np.dtype] = {
    "ADC": np.dtype("<i4"),
    "TDC": np.dtype("<i2"),
    "FOLDED": np.dtype("<f4"),
}
"""`TOFIntensityType` to element type. All four files of task 01 are `ADC`."""


def dtype_for(tof_intensity_type: str | None) -> np.dtype:
    """The element type a file's `TOFIntensityType` names, defaulting to `ADC`/int32.

    An absent or empty value means the oldest convention, which is int32; an unknown
    one is an error rather than a guess, because guessing wrong decodes silently into
    nonsense.
    """
    name = (tof_intensity_type or "ADC").strip().upper()
    try:
        return INTENSITY_DTYPES[name]
    except KeyError:
        raise ValueError(
            f"unknown TOFIntensityType {tof_intensity_type!r}; "
            f"one of {sorted(INTENSITY_DTYPES)}"
        ) from None


# --- encoding (used by the synthetic writer; see tests/synthetic.py) ---------------

_HLOG = 14
_HSIZE = 1 << _HLOG
_MAX_LIT = 1 << 5
_MAX_OFF = 1 << 13
_MAX_REF = (1 << 8) + (1 << 3)


def rlz_encode(
    bin_index: np.ndarray,
    intensity: np.ndarray,
    dtype: np.dtype | str = "<i4",
) -> np.ndarray:
    """Run-length-zero a sparse spectrum into the stream a UIMF writer stores.

    `bin_index` is strictly increasing and `intensity` parallel to it; zero intensities
    are encoded literally, as writers do, rather than folded into the surrounding gap.
    Trailing zeros past the last point are not represented at all, which is why a
    decoder needs the frame's `Bins` to know the axis extent.

    Gaps wider than the element type can hold negatively are split into consecutive
    skips -- only int16 files can reach that, and only after a 32767-bin silence.
    """
    dtype = np.dtype(dtype)
    bin_index = np.asarray(bin_index, dtype=np.int64)
    intensity = np.asarray(intensity)
    if bin_index.shape != intensity.shape:
        raise ValueError("bin_index and intensity must have the same shape")
    if bin_index.size and np.any(np.diff(bin_index) <= 0):
        raise ValueError("bin_index must be strictly increasing")

    if dtype.kind == "i":
        max_skip = int(np.iinfo(dtype).max)
    else:
        max_skip = 1 << 24  # exactly representable in float32; far past any real gap

    out: list[float] = []
    cursor = 0
    for b, value in zip(bin_index.tolist(), intensity.tolist()):
        gap = b - cursor
        while gap > 0:
            step = min(gap, max_skip)
            out.append(-step)
            gap -= step
        out.append(value)
        cursor = b + 1
    return np.array(out, dtype=dtype)


def lzf_compress(data: bytes | bytearray | memoryview) -> bytes:
    """Compress with liblzf, following `CLZF2.cs` so that UIMF-Library can read it back.

    Faithful to the reference in output, not merely in format: same 14-bit hash, same
    rehash-every-position match loop. Any valid LZF stream would decode, but matching
    the writer keeps a byte comparison against a real blob meaningful.

    Incompressible input still comes back as a valid stream of literal runs, slightly
    larger than the input; the caller decides whether that is worth storing.
    """
    src = bytes(data)
    n = len(src)
    if n == 0:
        return b""

    out = bytearray()
    htab = [0] * _HSIZE
    iidx = 0
    lit = 0
    out.append(0)  # placeholder control byte for the run that starts here

    if n > 2:
        hval = (src[0] << 8) | src[1]
        while iidx < n - 2:
            hval = ((hval << 8) | src[iidx + 2]) & 0xFFFFFFFF
            hslot = ((hval >> (24 - _HLOG)) - hval * 5) & (_HSIZE - 1)
            ref = htab[hslot]
            htab[hslot] = iidx
            off = iidx - ref - 1
            if (
                off < _MAX_OFF
                and iidx + 4 < n
                and ref > 0
                and src[ref] == src[iidx]
                and src[ref + 1] == src[iidx + 1]
                and src[ref + 2] == src[iidx + 2]
            ):
                length = 2
                maxlen = min(n - iidx - length, _MAX_REF)
                # Close the literal run, or drop the control byte it never used.
                if lit:
                    out[len(out) - lit - 1] = lit - 1
                    lit = 0
                else:
                    out.pop()
                while True:
                    length += 1
                    if length >= maxlen or src[ref + length] != src[iidx + length]:
                        break
                length -= 2
                iidx += 1
                if length < 7:
                    out.append(((off >> 8) + (length << 5)) & 0xFF)
                else:
                    out.append(((off >> 8) + (7 << 5)) & 0xFF)
                    out.append((length - 7) & 0xFF)
                out.append(off & 0xFF)
                out.append(0)  # placeholder for the next literal run
                iidx += length + 1
                if iidx >= n - 2:
                    break
                # Rewind and hash every position the match covered.
                iidx -= length + 1
                while True:
                    hval = ((hval << 8) | src[iidx + 2]) & 0xFFFFFFFF
                    hslot = ((hval >> (24 - _HLOG)) - hval * 5) & (_HSIZE - 1)
                    htab[hslot] = iidx
                    iidx += 1
                    if length == 0:
                        break
                    length -= 1
            else:
                out.append(src[iidx])
                iidx += 1
                lit += 1
                if lit == _MAX_LIT:
                    out[len(out) - lit - 1] = lit - 1
                    lit = 0
                    out.append(0)

    while iidx < n:
        out.append(src[iidx])
        iidx += 1
        lit += 1
        if lit == _MAX_LIT:
            out[len(out) - lit - 1] = lit - 1
            lit = 0
            out.append(0)

    if lit:
        out[len(out) - lit - 1] = lit - 1
    else:
        out.pop()
    return bytes(out)


def encode_intensities(
    bin_index: np.ndarray,
    intensity: np.ndarray,
    dtype: np.dtype | str = "<i4",
) -> bytes:
    """A sparse spectrum as a stored `Frame_Scans.Intensities` blob: RLZ, then LZF."""
    stream = rlz_encode(bin_index, intensity, dtype)
    return lzf_compress(stream.tobytes())


# --- decoding (lab record, task 03) -----------------------------------------------


def lzf_decompress(data: bytes, expected_size: int = 0) -> bytes:
    """Expand an LZF stream. `expected_size` is a hint, not a promise about the length.

    Arrives with the numba kernel and its pure-Python fallback (lab record, task 03).
    """
    raise NotImplementedError("LZF decompression arrives with the lab record's task 03")


def rlz_decode(
    values: np.ndarray,
    bins: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Undo the run-length-zero stream into `(bin_index, intensity)`.

    Explicit zeros in the stream advance the bin counter and emit nothing, matching
    UIMF-Library. `bins` bounds the result; 0 means trust the stream.

    Arrives with the lab record's task 03.
    """
    raise NotImplementedError("RLZ decoding arrives with the lab record's task 03")


def decode_intensities(
    blob: bytes,
    dtype: np.dtype | str = "<i4",
    count_hint: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """One stored blob to `(bin_index, intensity)`, the inverse of `encode_intensities`.

    `count_hint` is the row's `NonZeroCount` -- an upper bound on the number of points,
    so a safe preallocation size (lab record, task 01).

    Arrives with the lab record's task 03.
    """
    raise NotImplementedError("blob decoding arrives with the lab record's task 03")
