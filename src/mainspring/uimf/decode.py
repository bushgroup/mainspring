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

**One zero is not a bin.** PNNL's old encoder, on a run of zeros long enough to reach
the skip's limit, wrote `-32768` and then a spurious `0`; UIMF-Library's decoder skips a
`0` whose predecessor in the same stream is `-32768` or the element type's minimum
(`RlzEncode.tt`, "an old bug in the run-length zero encoding"). That *marker* moves
nothing when skipped and every later point of its scan up one bin when not, so the
decoders here skip it too -- `skip_markers=True`, the default. The exception is a file
mainspring's writer stamped (`MainspringWriter`): the acquisition console that fills it
writes zeros literally and does not clamp, so there `-32768, 0` is a gap of 32768 bins
and a real zero, and `UimfFile` decodes it with `skip_markers=False`. mainspring's own
encoder never writes a `-32768` skip at all, so what it writes reads the same under
either rule (lab record, task 37).

LZF is Marc Lehmann's liblzf as PNNL's `CLZF2.cs` carries it: a control byte `c` under
32 introduces `c + 1` literal bytes, and `c >= 32` a back-reference of length
`(c >> 5) + 2` -- with one extra length byte when `c >> 5 == 7` -- at offset
`((c & 31) << 8 | next) + 1` behind the output cursor. Back-references may overlap the
bytes they are still producing, so a copy is byte-at-a-time.

**Both directions have two backends that must agree byte for byte.** Decoding has a
vectorised numpy path that is the reference and a numba kernel over a whole frame's blobs
at once, which is what the viewer runs: reading a frame is where the difference decides
whether a gesture feels immediate. Encoding has `rlz_encode` and `lzf_compress`, pure
Python and the reference, and numba kernels over a whole frame's scans at once
(`encode_frame_blobs`), which is what `UimfWriter.write_scans` runs. The encoder was
written for the test fixture alone; it became production when the acquisition side's
fold started writing every summed companion through it, and a beam-on fold spent 94% of
its ten minutes in the pure `lzf_compress` (lab record, task 33). Both kernel sets are
written frame-at-a-time rather than scan-at-a-time, and numba is imported only when
something asks for it.

The encode kernels are not a different compressor that happens to be valid. They
reproduce the pure encoder's output exactly -- same hash, same match rule, same quirk
that position 0 of a stream can never be matched -- so a file written with numba and a
file written without it are the same bytes, and a byte comparison against a stored blob
stays a meaningful test of either.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np

__all__ = [
    "BACKENDS",
    "count_markers",
    "INTENSITY_DTYPES",
    "decode_frame_blobs",
    "decode_intensities",
    "dtype_for",
    "encode_frame_blobs",
    "encode_intensities",
    "frozen_cache_folder",
    "install_frozen_cache_locator",
    "lzf_compress",
    "lzf_decompress",
    "numba_available",
    "rlz_decode",
    "rlz_encode",
    "warm_kernels",
]

INTENSITY_DTYPES: dict[str, np.dtype] = {
    "ADC": np.dtype("<i4"),
    "TDC": np.dtype("<i2"),
    "FOLDED": np.dtype("<f4"),
}
"""`TOFIntensityType` to element type. All four files of task 01 are `ADC`."""

_FAST_DTYPES = tuple(INTENSITY_DTYPES.values())
"""The element types the encode kernels are compiled for; anything else encodes pure."""

_CLAMP = -32768
"""The skip PNNL's old encoder clamped to, after which it wrote a spurious zero."""

_AMBIGUOUS_SKIP = 32768
"""The one skip length mainspring's encoder never writes: it would be `-32768`."""


def _marker_floor(dtype: np.dtype) -> int:
    """The second predecessor that makes a zero a marker: the element type's minimum.

    UIMF-Library tests `short.MinValue` and `<T>.MinValue`; its float instantiation
    tests `short.MinValue` twice, so for float32 the floor is `-32768` again.
    """
    return int(np.iinfo(dtype).min) if dtype.kind == "i" else _CLAMP


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


# --- encoding: the pure path, which is also the reference ---------------------------

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

    A skip of exactly 32768 is written `-32767, -1`, never `-32768`: UIMF-Library takes
    a `0` straight after `-32768` for its old encoder's spurious one and skips it, so a
    literal zero there would read one bin early to every PNNL tool (lab record, task 37).
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
            if step == _AMBIGUOUS_SKIP:
                step -= 1
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
    backend: str = "auto",
) -> bytes:
    """A sparse spectrum as a stored `Frame_Scans.Intensities` blob: RLZ, then LZF.

    `backend` is as for `decode_frame_blobs`; every backend returns the same bytes. One
    spectrum at a time is the convenient call; a whole frame is `encode_frame_blobs`.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; one of {BACKENDS}")
    if backend != "pure":
        bin_index = np.asarray(bin_index, dtype=np.int64)
        scan_start = np.array([0, bin_index.size], dtype=np.int64)
        return encode_frame_blobs(scan_start, bin_index, intensity, dtype, backend)[0]
    stream = rlz_encode(bin_index, intensity, dtype)
    return lzf_compress(stream.tobytes())


def _compilable(
    bin_index: np.ndarray, intensity: np.ndarray, dtype: np.dtype
) -> np.ndarray | None:
    """`intensity` as `dtype` when the kernels would give the pure encoder's answer for it,
    otherwise None.

    The pure encoder goes through Python numbers, so a value the element type cannot hold
    is an `OverflowError` there and an int that float32 must round is rounded once. A cast
    here would wrap the first silently and could round the second differently, so anything
    that does not survive the cast unchanged goes to the pure path and meets whatever it
    meets there today. So do a byte order other than the machine's, which numba cannot
    compile, an element type the format does not use, and anything that is not 1-D.
    """
    if dtype not in _FAST_DTYPES or not dtype.isnative:
        return None
    if bin_index.ndim != 1 or np.ndim(intensity) != 1:
        return None
    values = np.asarray(intensity)
    if values.dtype == dtype:
        return values
    if values.dtype.kind not in "biuf":
        return None
    cast = values.astype(dtype)
    with np.errstate(invalid="ignore"):
        if not np.array_equal(cast, values):
            return None
    return cast


def encode_frame_blobs(
    scan_start: np.ndarray,
    bin_index: np.ndarray,
    intensity: np.ndarray,
    dtype: np.dtype | str = "<i4",
    backend: str = "auto",
) -> list[bytes]:
    """Every scan of a CSR frame as its stored blob, `b""` for an empty scan.

    The inverse of `decode_frame_blobs`, laid out like `SparseFrame`: the points of scan
    `s` are `bin_index[scan_start[s]:scan_start[s + 1]]`, ascending within the scan, with
    the parallel `intensity`. With numba this is three passes over contiguous memory --
    size the RLZ streams, write them, compress them -- in one call that releases the GIL,
    and every blob is the pure encoder's to the byte: `tests/test_decode.py` holds the
    two against each other, and against every blob of a real summed file when the clone
    has one.

    Memory is the RLZ stream plus a compression buffer of about the same size, on top of
    the input, for the whole of what is passed. `UimfWriter.write_scans` passes a few
    million points at a time rather than a frame.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; one of {BACKENDS}")
    dtype = np.dtype(dtype)
    kernels = _kernels() if backend in ("auto", "numba") else None
    if backend == "numba" and kernels is None:
        raise RuntimeError("the numba backend was asked for and numba is not importable")

    scan_start = np.asarray(scan_start, dtype=np.int64)
    bin_index = np.asarray(bin_index, dtype=np.int64)
    scans = scan_start.size - 1
    values = None if kernels is None else _compilable(bin_index, intensity, dtype)
    if values is None:
        intensity = np.asarray(intensity)
        return [
            encode_intensities(bin_index[scan_start[s]:scan_start[s + 1]],
                               intensity[scan_start[s]:scan_start[s + 1]], dtype, "pure")
            for s in range(scans)
        ]
    if bin_index.shape != values.shape:
        raise ValueError("bin_index and intensity must have the same shape")
    if bin_index.size > 1:
        falling = np.diff(bin_index) <= 0
        # A step down between the last point of one scan and the first of the next is
        # the CSR layout, not a disorder.
        boundary = scan_start[1:-1]
        boundary = boundary[(boundary > 0) & (boundary < bin_index.size)] - 1
        falling[boundary] = False
        if falling.any():
            raise ValueError("bin_index must be strictly increasing")

    if dtype.kind == "i":
        max_skip = int(np.iinfo(dtype).max)
    else:
        max_skip = 1 << 24  # as rlz_encode: exactly representable in float32

    sizes = np.empty(scans, dtype=np.int64)
    kernels.rlz_encode_sizes(bin_index, scan_start, max_skip, sizes)
    elem_off = np.zeros(scans + 1, dtype=np.int64)
    np.cumsum(sizes, out=elem_off[1:])
    stream = np.empty(int(elem_off[-1]), dtype=dtype)
    kernels.rlz_encode_fill(bin_index, values, scan_start, max_skip, elem_off, stream)

    src = stream.view(np.uint8)
    src_off = elem_off * dtype.itemsize
    # LZF's worst case is a literal run of 32 bytes per 33 written, plus the one
    # placeholder control byte the compressor writes before it knows it will not need it.
    nbytes = np.diff(src_off)
    capacity = nbytes + (nbytes >> 5) + 2
    dst_off = np.zeros(scans + 1, dtype=np.int64)
    np.cumsum(capacity, out=dst_off[1:])
    dst = np.empty(int(dst_off[-1]), dtype=np.uint8)
    lengths = np.empty(scans, dtype=np.int64)
    htab = np.zeros(_HSIZE, dtype=np.int64)
    kernels.lzf_compress(src, src_off, dst, dst_off, lengths, htab)

    ends = (dst_off[:-1] + lengths).tolist()
    return [dst[a:b].tobytes() for a, b in zip(dst_off[:-1].tolist(), ends)]


# --- decoding: the pure path, which is also the reference ---------------------------


def lzf_decompress(data: bytes, expected_size: int = 0) -> bytes:
    """Expand an LZF stream. `expected_size` is a hint, not a promise about the length.

    The pure-Python path, and the reference the numba kernel is tested against: a
    `bytearray` and a byte loop, because LZF is sequential by construction -- a
    back-reference may overlap the bytes it is still producing, so a copy cannot be
    vectorised.

    A stream that runs off its own end, or points behind the start of its output, is a
    `ValueError` rather than a short result. A truncated spectrum decodes into plausible
    nonsense, and plausible nonsense is the one outcome worth crashing over.
    """
    del expected_size  # a hint this backend has no use for; see the docstring
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            run = ctrl + 1
            if i + run > n:
                raise ValueError("LZF literal run runs past the end of the stream")
            out += data[i:i + run]
            i += run
        else:
            length = ctrl >> 5
            if length == 7:
                if i >= n:
                    raise ValueError("LZF length-extension byte runs past the end")
                length += data[i]
                i += 1
            if i >= n:
                raise ValueError("LZF back-reference offset byte runs past the end")
            offset = ((ctrl & 0x1F) << 8 | data[i]) + 1
            i += 1
            length += 2
            ref = len(out) - offset
            if ref < 0:
                raise ValueError("LZF back-reference points before the start of the output")
            if offset >= length:
                out += out[ref:ref + length]
            else:  # overlapping: the copy must see the bytes it is writing
                for k in range(length):
                    out.append(out[ref + k])
    return bytes(out)


def _markers(values: np.ndarray) -> np.ndarray:
    """Which entries of one stream are the old encoder's spurious zero: a `0` whose
    predecessor is `-32768` or the element type's minimum. The first entry never is."""
    marker = np.zeros(values.size, dtype=bool)
    if values.size > 1:
        before = values[:-1]
        marker[1:] = (values[1:] == 0) & (
            (before == _CLAMP) | (before == _marker_floor(values.dtype)))
    return marker


def rlz_decode(
    values: np.ndarray, bins: int = 0, skip_markers: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Undo the run-length-zero stream into `(bin_index, intensity)`.

    Explicit zeros in the stream advance the bin counter and emit nothing, matching
    UIMF-Library -- except, with `skip_markers`, a zero straight after `-32768` or the
    element type's minimum, which advances nothing (the module docstring says why, and
    why a mainspring-stamped file is read with it off). `bins` bounds the result; 0
    means trust the stream.

    Vectorised rather than looped: the bin of an entry is the sum of its predecessors'
    strides, which is one `cumsum`, and the points are then a boolean mask. The stride
    is widened to int64 before it is negated, so that an int16 stream carrying -32768 --
    which our own encoder never writes, but a file might -- cannot wrap around into a
    forward skip.
    """
    values = np.asarray(values)
    if values.size == 0:
        return np.empty(0, dtype=np.int32), values
    stride = np.where(values < 0, -values.astype(np.int64), np.int64(1))
    if skip_markers:
        stride[_markers(values)] = 0
    position = np.cumsum(stride) - stride
    keep = values > 0
    if bins > 0:
        keep &= position < bins
    return position[keep].astype(np.int32), values[keep]


def decode_intensities(
    blob: bytes,
    dtype: np.dtype | str = "<i4",
    count_hint: int = 0,
    skip_markers: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One stored blob to `(bin_index, intensity)`, the inverse of `encode_intensities`.

    `skip_markers` is as for `decode_frame_blobs`.

    `count_hint` is the row's `NonZeroCount`, an upper bound on the number of points
    (lab record, task 01). It is accepted for symmetry with the frame-at-a-time path and
    used by neither: both decoders size their output from the stream itself, and a hint
    that happens to be wrong must not be able to truncate a spectrum.

    One blob at a time is the convenient call, not the fast one. A frame is thousands of
    small blobs and the per-call overhead dominates; the viewer uses `decode_frame_blobs`.
    """
    del count_hint  # see the docstring
    dtype = np.dtype(dtype)
    if not blob:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=dtype)
    raw = lzf_decompress(blob)
    if len(raw) % dtype.itemsize:
        raise ValueError(
            f"decompressed length {len(raw)} is not a multiple of the {dtype} element size"
        )
    return rlz_decode(np.frombuffer(raw, dtype=dtype), skip_markers=skip_markers)


# --- decoding: a whole frame in one pass, over the numba kernels --------------------

BACKENDS = ("auto", "numba", "pure")
"""What `decode_frame_blobs` may be told to use. `auto` is numba when it imports."""

_KERNELS: object | None = None
_KERNELS_LOOKED = False


def numba_available() -> bool:
    """Whether the compiled decode path is usable in this interpreter.

    False is not an error: the pure path gives the same answer an order of magnitude
    slower, which is the difference between a viewer and a script rather than between
    right and wrong. `uimf-info --bench` prints which one it ran.
    """
    return _kernels() is not None


def _kernels():
    """Compile the kernels on first use, or return None when numba is not installed.

    Lazy on purpose, and looked up at most once. `import numba` costs the better part of
    a second, and `import mainspring.uimf` is on the path of every pipeline that only
    wants a parameter out of a file.
    """
    global _KERNELS, _KERNELS_LOOKED
    if _KERNELS_LOOKED:
        return _KERNELS
    _KERNELS_LOOKED = True
    try:
        from numba import njit
    except Exception:  # noqa: BLE001 -- absent, broken and incompatible all mean "pure"
        _KERNELS = None
        return None
    from types import SimpleNamespace

    install_frozen_cache_locator()
    compiled = njit(cache=True, nogil=True)
    _KERNELS = SimpleNamespace(
        lzf_sizes=compiled(_k_lzf_sizes),
        lzf_expand=compiled(_k_lzf_expand),
        rlz_count=compiled(_k_rlz_count),
        rlz_fill=compiled(_k_rlz_fill),
        rlz_markers=compiled(_k_rlz_markers),
        rlz_encode_sizes=compiled(_k_rlz_encode_sizes),
        rlz_encode_fill=compiled(_k_rlz_encode_fill),
        lzf_compress=compiled(_k_lzf_compress),
    )
    return _KERNELS


_FROZEN_LOCATOR_INSTALLED = False


def frozen_cache_folder(root: str, py_file: str) -> str:
    """Where a frozen program keeps the compiled kernels of the module at `py_file`.

    `<root>/<program>_<package>`, for example `mainspring_uimf` for the viewer and
    `clockwork_uimf` for a frozen acquisition program that folds through this layer. The
    program's name keeps two frozen programs sharing one root from overwriting each
    other's index, and nothing here depends on the directory the program was started in.
    """
    program = os.path.splitext(os.path.basename(sys.executable))[0].lower()
    package = os.path.basename(os.path.dirname(py_file)) or "module"
    return os.path.join(root, f"{program}_{package}")


def install_frozen_cache_locator() -> bool:
    """In a frozen program, make numba cache kernels in one place whatever the launch folder.

    numba finds a cache folder through a list of locators, and in a PyInstaller build
    every one of them misreads the situation: the build carries bytecode and no `.py`,
    so the locator that honours `NUMBA_CACHE_DIR` declines (it wants the source file to
    exist), and the one that accepts a frozen program names its folder after
    `abspath(co_filename)`, where `co_filename` is PyInstaller's relative
    `mainspring/uimf/decode.py` (backslashed on Windows) -- so after the directory the program happened to be
    started in. An installed viewer started from twenty-five folders had compiled into
    twenty-five caches and never read the one it was seeded with (lab record, task 34).

    This puts a locator at the head of numba's list that answers only in a frozen program:
    `NUMBA_CACHE_DIR` if set, numba's own per-user folder otherwise, and
    `frozen_cache_folder` under it. The stamp stays numba's frozen one, the executable's
    modification time and size, so an upgrade still invalidates it once. Uses numba's
    internal caching classes (`CacheImpl._locator_classes`, the locator base and mixin);
    if they are not where numba 0.67 keeps them, this does nothing and numba behaves as
    it always has. Returns whether the locator is in place.
    """
    global _FROZEN_LOCATOR_INSTALLED
    if _FROZEN_LOCATOR_INSTALLED:
        return True
    if not getattr(sys, "frozen", False):
        return False
    try:
        from numba.core import caching, config
        from numba.misc.appdirs import AppDirs
    except Exception:  # noqa: BLE001 -- no numba, or a numba laid out differently
        return False

    class FrozenCacheLocator(caching._SourceFileBackedLocatorMixin, caching._CacheLocator):
        def __init__(self, py_func, py_file):
            self._py_file = py_file
            self._lineno = py_func.__code__.co_firstlineno
            root = config.CACHE_DIR or AppDirs(appname="numba", appauthor=False).user_cache_dir
            self._cache_path = frozen_cache_folder(root, py_file)

        def get_cache_path(self):
            return self._cache_path

        @classmethod
        def from_function(cls, py_func, py_file):
            if not getattr(sys, "frozen", False):
                return None
            self = cls(py_func, py_file)
            try:
                self.ensure_cache_path()
            except OSError:
                return None
            return self

    # `CacheImpl` in numba 0.67, `_CacheImpl` before it was made public.
    impl = getattr(caching, "CacheImpl", None) or getattr(caching, "_CacheImpl", None)
    try:
        impl._locator_classes.insert(0, FrozenCacheLocator)
    except Exception:  # noqa: BLE001
        return False
    _FROZEN_LOCATOR_INSTALLED = True
    return True


def warm_kernels() -> list[str]:
    """Compile every kernel for every element type the format uses; return what was warmed.

    What a build runs so the first file an installed program opens, sums or folds pays no
    compile: the five decode kernels (the marker count among them), the three encode kernels at the types
    `UimfWriter.write_scans` passes them (an int64 row pointer and bin index), and the sum
    behind `sum_frames` (two frames, since one alone never reaches it), each checked
    against the pure path. Where the compiled kernels land is numba's business, and in a
    frozen program `install_frozen_cache_locator`'s. Raises if numba is not importable.
    """
    if _kernels() is None:
        raise RuntimeError("numba is not importable in this environment; nothing to warm")
    from .frame import SparseFrame, sum_frames

    warmed = []
    for type_name in sorted(INTENSITY_DTYPES):
        dtype = dtype_for(type_name)
        bin_index = np.array([0, 3, 500, 4096], dtype=np.int64)
        intensity = np.array([1, 2, 3, 4], dtype=dtype)
        scan_start = np.array([0, 0, bin_index.size], dtype=np.int64)
        empty, blob = encode_frame_blobs(scan_start, bin_index, intensity, dtype,
                                         backend="numba")
        if empty != b"" or blob != encode_intensities(bin_index, intensity, dtype,
                                                      backend="pure"):
            raise AssertionError(f"{type_name}: the compiled encoder disagrees with the pure one")
        counts, _, values = decode_frame_blobs([blob, None, blob], dtype=dtype)
        if values.dtype != dtype or int(counts.sum()) != 2 * bin_index.size:
            raise AssertionError(f"{type_name}: the compiled decoder lost points")
        if count_markers([blob, None], dtype=dtype, backend="numba").tolist() != [0, 0]:
            raise AssertionError(f"{type_name}: the compiled marker count found one")
        frame = SparseFrame(frame=1, scans=2, bins=4097, scan_start=scan_start,
                            bin_index=bin_index.astype(np.int32), intensity=intensity)
        total = sum_frames([frame, frame])
        if total.intensity.dtype != dtype or total.intensity.tolist() != [2, 4, 6, 8]:
            raise AssertionError(f"{type_name}: the compiled sum disagrees")
        warmed.append(f"{type_name} ({dtype})")
    return warmed


# The eight kernels below -- five to decode, three to encode -- are module-level plain
# Python, so that numba can cache their compilation against this file and so that they
# read as the format's rules rather than as numba. They are not the pure fallback: these
# loops would be glacial in the interpreter, and the functions above are what runs
# without numba.


def _k_lzf_sizes(src, src_off, sizes):
    """Decompressed length of each blob, or -1 for a stream that runs off its own end."""
    for i in range(sizes.size):
        p = src_off[i]
        end = src_off[i + 1]
        produced = 0
        broken = False
        while p < end:
            ctrl = src[p]
            p += 1
            if ctrl < 32:
                run = ctrl + 1
                if p + run > end:
                    broken = True
                    break
                p += run
                produced += run
            else:
                length = ctrl >> 5
                if length == 7:
                    if p >= end:
                        broken = True
                        break
                    length += src[p]
                    p += 1
                if p >= end:
                    broken = True
                    break
                p += 1
                produced += length + 2
        sizes[i] = -1 if broken else produced


def _k_lzf_expand(src, src_off, dst, dst_off, status):
    """Expand every blob into its own slice of `dst`; `status[i]` is 0, or 1 for a
    back-reference that points behind the blob's output. Copies are byte-at-a-time
    because a reference may overlap the bytes it is producing."""
    for i in range(status.size):
        p = src_off[i]
        end = src_off[i + 1]
        base = dst_off[i]
        o = base
        status[i] = 0
        while p < end:
            ctrl = src[p]
            p += 1
            if ctrl < 32:
                for _ in range(ctrl + 1):
                    dst[o] = src[p]
                    o += 1
                    p += 1
            else:
                length = ctrl >> 5
                if length == 7:
                    length += src[p]
                    p += 1
                offset = ((ctrl & 31) << 8 | src[p]) + 1
                p += 1
                length += 2
                ref = o - offset
                if ref < base:
                    status[i] = 1
                    break
                for _ in range(length):
                    dst[o] = dst[ref]
                    o += 1
                    ref += 1


def _k_rlz_count(values, elem_off, bins, skip, floor, counts):
    """Points per blob: the entries above zero that land inside the bin axis.

    With `skip`, a zero whose predecessor *in the same blob* is `-32768` or `floor` (the
    element type's minimum) advances nothing; the look-back starts afresh per blob."""
    for i in range(counts.size):
        position = 0
        found = 0
        start = elem_off[i]
        for j in range(start, elem_off[i + 1]):
            value = values[j]
            if value < 0:
                position -= int(value)  # widen, then negate: int16 -32768 would wrap
            elif value > 0:
                if bins <= 0 or position < bins:
                    found += 1
                position += 1
            elif not (skip and j > start
                      and (values[j - 1] == -32768 or values[j - 1] == floor)):
                position += 1
        counts[i] = found


def _k_rlz_fill(values, elem_off, bins, skip, floor, out_start, out_bins, out_intensity):
    """The same walk again, writing blob `i`'s points from `out_start[i]`."""
    for i in range(out_start.size - 1):
        position = 0
        o = out_start[i]
        start = elem_off[i]
        for j in range(start, elem_off[i + 1]):
            value = values[j]
            if value < 0:
                position -= int(value)  # widen, then negate: int16 -32768 would wrap
            elif value > 0:
                if bins <= 0 or position < bins:
                    out_bins[o] = position
                    out_intensity[o] = value
                    o += 1
                position += 1
            elif not (skip and j > start
                      and (values[j - 1] == -32768 or values[j - 1] == floor)):
                position += 1


def _k_rlz_markers(values, elem_off, floor, markers):
    """Marker zeros per blob, the ones `skip` passes over in the two walks above."""
    for i in range(markers.size):
        found = 0
        for j in range(elem_off[i] + 1, elem_off[i + 1]):
            if values[j] == 0 and (values[j - 1] == -32768 or values[j - 1] == floor):
                found += 1
        markers[i] = found


def _k_rlz_encode_sizes(bin_index, scan_start, max_skip, sizes):
    """Stream length of each scan: one entry per point, one per `max_skip` of gap, and
    one more where the last piece of a gap is exactly 32768 and is written `-32767, -1`.
    No `max_skip` is 32768, so only the last piece can be."""
    for i in range(sizes.size):
        cursor = 0
        count = 0
        for j in range(scan_start[i], scan_start[i + 1]):
            gap = bin_index[j] - cursor
            if gap > 0:
                count += (gap + max_skip - 1) // max_skip
                if gap % max_skip == 32768:
                    count += 1
            count += 1
            cursor = bin_index[j] + 1
        sizes[i] = count


def _k_rlz_encode_fill(bin_index, intensity, scan_start, max_skip, elem_off, stream):
    """`rlz_encode`'s walk, scan `i`'s stream written from `elem_off[i]`."""
    for i in range(scan_start.size - 1):
        o = elem_off[i]
        cursor = 0
        for j in range(scan_start[i], scan_start[i + 1]):
            gap = bin_index[j] - cursor
            while gap > 0:
                step = min(gap, max_skip)
                if step == 32768:
                    step = 32767
                stream[o] = -step
                o += 1
                gap -= step
            stream[o] = intensity[j]
            o += 1
            cursor = bin_index[j] + 1


def _k_lzf_compress(src, src_off, dst, dst_off, lengths, htab):
    """`lzf_compress` on every stream, stream `i` written from `dst_off[i]`.

    Line for line the pure function, with one difference that changes no output: the
    hash table is shared across streams and holds absolute positions in `src`, so it is
    zeroed once per call rather than once per stream. An entry left by an earlier stream
    is behind this one's start, and the pure function's `ref > 0` test -- which is also
    what makes its own initial zeros mean "empty" -- is here `ref > 0` after subtracting
    the start, which rejects both.
    """
    for i in range(lengths.size):
        s0 = src_off[i]
        n = src_off[i + 1] - s0
        base = dst_off[i]
        if n == 0:
            lengths[i] = 0
            continue
        o = base
        dst[o] = 0  # placeholder control byte for the run that starts here
        o += 1
        lit = 0
        iidx = 0
        if n > 2:
            hval = (np.int64(src[s0]) << 8) | np.int64(src[s0 + 1])
            while iidx < n - 2:
                hval = ((hval << 8) | np.int64(src[s0 + iidx + 2])) & 0xFFFFFFFF
                hslot = ((hval >> 10) - hval * 5) & 16383
                ref = htab[hslot] - s0
                htab[hslot] = s0 + iidx
                off = iidx - ref - 1
                if (
                    off < 8192
                    and iidx + 4 < n
                    and ref > 0
                    and src[s0 + ref] == src[s0 + iidx]
                    and src[s0 + ref + 1] == src[s0 + iidx + 1]
                    and src[s0 + ref + 2] == src[s0 + iidx + 2]
                ):
                    length = 2
                    maxlen = min(n - iidx - length, 264)
                    if lit:
                        dst[o - lit - 1] = lit - 1
                        lit = 0
                    else:
                        o -= 1
                    while True:
                        length += 1
                        if length >= maxlen or src[s0 + ref + length] != src[s0 + iidx + length]:
                            break
                    length -= 2
                    iidx += 1
                    if length < 7:
                        dst[o] = (off >> 8) + (length << 5)
                        o += 1
                    else:
                        dst[o] = (off >> 8) + (7 << 5)
                        dst[o + 1] = length - 7
                        o += 2
                    dst[o] = off & 0xFF
                    dst[o + 1] = 0  # placeholder for the next literal run
                    o += 2
                    iidx += length + 1
                    if iidx >= n - 2:
                        break
                    iidx -= length + 1
                    while True:
                        hval = ((hval << 8) | np.int64(src[s0 + iidx + 2])) & 0xFFFFFFFF
                        hslot = ((hval >> 10) - hval * 5) & 16383
                        htab[hslot] = s0 + iidx
                        iidx += 1
                        if length == 0:
                            break
                        length -= 1
                else:
                    dst[o] = src[s0 + iidx]
                    o += 1
                    iidx += 1
                    lit += 1
                    if lit == 32:
                        dst[o - lit - 1] = lit - 1
                        lit = 0
                        dst[o] = 0
                        o += 1
        while iidx < n:
            dst[o] = src[s0 + iidx]
            o += 1
            iidx += 1
            lit += 1
            if lit == 32:
                dst[o - lit - 1] = lit - 1
                lit = 0
                dst[o] = 0
                o += 1
        if lit:
            dst[o - lit - 1] = lit - 1
        else:
            o -= 1
        lengths[i] = o - base


def decode_frame_blobs(
    blobs: Sequence[bytes | None],
    dtype: np.dtype | str = "<i4",
    bins: int = 0,
    backend: str = "auto",
    skip_markers: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A whole frame's blobs at once: `(counts, bin_index, intensity)`.

    `counts[i]` is how many points blob `i` produced, which is the CSR row pointer
    without a second walk; `bin_index` and `intensity` are every blob's points
    concatenated in blob order. `bins` drops points past the frame's bin axis, 0 to
    trust the stream.

    `skip_markers` reads a `0` straight after `-32768` (or the element type's minimum)
    as PNNL's old encoder's spurious zero and not a bin, as UIMF-Library does; it is
    the default because it is right on every file mainspring did not write. A file
    mainspring stamped carries that pair legitimately and wants `False`, which
    `UimfFile` passes on its own; a caller decoding such a file's blobs directly must
    pass it too, or a point after a 32768-bin gap and a real zero reads one bin early.

    A frame is thousands of small blobs, so the whole frame is one call and, with numba,
    four passes over contiguous memory: measure the sizes, expand, count the points,
    write them. Both backends return identical arrays -- `tests/test_decode.py` asserts
    that blob by blob on every file the clone has.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; one of {BACKENDS}")
    dtype = np.dtype(dtype)
    kernels = _kernels() if backend in ("auto", "numba") else None
    if backend == "numba" and kernels is None:
        raise RuntimeError("the numba backend was asked for and numba is not importable")

    payloads = [b"" if blob is None else bytes(blob) for blob in blobs]
    counts = np.zeros(len(payloads), dtype=np.int64)
    empty = (counts, np.empty(0, dtype=np.int32), np.empty(0, dtype=dtype))
    if not any(payloads):
        return empty
    skip = bool(skip_markers)

    if kernels is None:
        piece_bins: list[np.ndarray] = []
        piece_intensity: list[np.ndarray] = []
        for i, payload in enumerate(payloads):
            if not payload:
                continue
            raw = lzf_decompress(payload)
            if len(raw) % dtype.itemsize:
                raise ValueError(
                    f"blob {i}: decompressed length {len(raw)} is not a multiple of the"
                    f" {dtype} element size"
                )
            bin_index, intensity = rlz_decode(np.frombuffer(raw, dtype=dtype), bins, skip)
            counts[i] = bin_index.size
            piece_bins.append(bin_index)
            piece_intensity.append(intensity)
        if not piece_bins:
            return empty
        return (
            counts,
            np.concatenate(piece_bins).astype(np.int32, copy=False),
            np.concatenate(piece_intensity).astype(dtype, copy=False),
        )

    values, elem_off = _expand(kernels, payloads, dtype)
    floor = _marker_floor(dtype)
    kernels.rlz_count(values, elem_off, int(bins), skip, floor, counts)
    out_start = np.zeros(len(payloads) + 1, dtype=np.int64)
    np.cumsum(counts, out=out_start[1:])
    out_bins = np.empty(int(out_start[-1]), dtype=np.int32)
    out_intensity = np.empty(int(out_start[-1]), dtype=dtype)
    kernels.rlz_fill(values, elem_off, int(bins), skip, floor, out_start, out_bins,
                     out_intensity)
    return counts, out_bins, out_intensity


def count_markers(
    blobs: Sequence[bytes | None],
    dtype: np.dtype | str = "<i4",
    backend: str = "auto",
) -> np.ndarray:
    """How many marker zeros each blob carries: a `0` straight after `-32768` or the
    element type's minimum, the pair `skip_markers` decides the reading of.

    What `uimf-info --verify` reports beside the rule a file was read with: on a file
    mainspring did not write every one is PNNL's spurious zero, and skipping it is what
    puts each later point of its scan back on the bin the writer's `BPI_MZ` names.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; one of {BACKENDS}")
    dtype = np.dtype(dtype)
    kernels = _kernels() if backend in ("auto", "numba") else None
    if backend == "numba" and kernels is None:
        raise RuntimeError("the numba backend was asked for and numba is not importable")
    payloads = [b"" if blob is None else bytes(blob) for blob in blobs]
    markers = np.zeros(len(payloads), dtype=np.int64)
    if not any(payloads):
        return markers
    if kernels is None:
        for i, payload in enumerate(payloads):
            if payload:
                raw = lzf_decompress(payload)
                if len(raw) % dtype.itemsize:
                    raise ValueError(
                        f"blob {i}: decompressed length {len(raw)} is not a multiple of"
                        f" the {dtype} element size"
                    )
                markers[i] = int(_markers(np.frombuffer(raw, dtype=dtype)).sum())
        return markers
    values, elem_off = _expand(kernels, payloads, dtype)
    kernels.rlz_markers(values, elem_off, _marker_floor(dtype), markers)
    return markers


def _expand(kernels, payloads: list[bytes], dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Every blob LZF-expanded into one array of `dtype`, and each blob's element offsets."""
    src = np.frombuffer(b"".join(payloads), dtype=np.uint8)
    src_off = np.zeros(len(payloads) + 1, dtype=np.int64)
    np.cumsum([len(p) for p in payloads], out=src_off[1:])

    sizes = np.empty(len(payloads), dtype=np.int64)
    kernels.lzf_sizes(src, src_off, sizes)
    broken = np.flatnonzero(sizes < 0)
    if broken.size:
        raise ValueError(f"blob {int(broken[0])}: LZF stream runs past its own end")
    ragged = np.flatnonzero(sizes % dtype.itemsize)
    if ragged.size:
        i = int(ragged[0])
        raise ValueError(
            f"blob {i}: decompressed length {int(sizes[i])} is not a multiple of the"
            f" {dtype} element size"
        )

    dst_off = np.zeros(len(payloads) + 1, dtype=np.int64)
    np.cumsum(sizes, out=dst_off[1:])
    dst = np.empty(int(dst_off[-1]), dtype=np.uint8)
    status = np.empty(len(payloads), dtype=np.int64)
    kernels.lzf_expand(src, src_off, dst, dst_off, status)
    stray = np.flatnonzero(status != 0)
    if stray.size:
        raise ValueError(
            f"blob {int(stray[0])}: LZF back-reference points before the start of the output"
        )

    return dst.view(dtype), dst_off // dtype.itemsize
