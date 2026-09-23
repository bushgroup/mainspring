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

from typing import Sequence

import numpy as np

__all__ = [
    "BACKENDS",
    "INTENSITY_DTYPES",
    "decode_frame_blobs",
    "decode_intensities",
    "dtype_for",
    "encode_frame_blobs",
    "encode_intensities",
    "lzf_compress",
    "lzf_decompress",
    "numba_available",
    "rlz_decode",
    "rlz_encode",
]

INTENSITY_DTYPES: dict[str, np.dtype] = {
    "ADC": np.dtype("<i4"),
    "TDC": np.dtype("<i2"),
    "FOLDED": np.dtype("<f4"),
}
"""`TOFIntensityType` to element type. All four files of task 01 are `ADC`."""

_FAST_DTYPES = tuple(INTENSITY_DTYPES.values())
"""The element types the encode kernels are compiled for; anything else encodes pure."""


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


def rlz_decode(values: np.ndarray, bins: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Undo the run-length-zero stream into `(bin_index, intensity)`.

    Explicit zeros in the stream advance the bin counter and emit nothing, matching
    UIMF-Library. `bins` bounds the result; 0 means trust the stream.

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
    position = np.cumsum(stride) - stride
    keep = values > 0
    if bins > 0:
        keep &= position < bins
    return position[keep].astype(np.int32), values[keep]


def decode_intensities(
    blob: bytes,
    dtype: np.dtype | str = "<i4",
    count_hint: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """One stored blob to `(bin_index, intensity)`, the inverse of `encode_intensities`.

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
    return rlz_decode(np.frombuffer(raw, dtype=dtype))


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

    compiled = njit(cache=True, nogil=True)
    _KERNELS = SimpleNamespace(
        lzf_sizes=compiled(_k_lzf_sizes),
        lzf_expand=compiled(_k_lzf_expand),
        rlz_count=compiled(_k_rlz_count),
        rlz_fill=compiled(_k_rlz_fill),
        rlz_encode_sizes=compiled(_k_rlz_encode_sizes),
        rlz_encode_fill=compiled(_k_rlz_encode_fill),
        lzf_compress=compiled(_k_lzf_compress),
    )
    return _KERNELS


# The seven kernels below -- four to decode, three to encode -- are module-level plain
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


def _k_rlz_count(values, elem_off, bins, counts):
    """Points per blob: the entries above zero that land inside the bin axis."""
    for i in range(counts.size):
        position = 0
        found = 0
        for j in range(elem_off[i], elem_off[i + 1]):
            value = values[j]
            if value < 0:
                position += int(-value)
            else:
                if value > 0 and (bins <= 0 or position < bins):
                    found += 1
                position += 1
        counts[i] = found


def _k_rlz_fill(values, elem_off, bins, out_start, out_bins, out_intensity):
    """The same walk again, writing blob `i`'s points from `out_start[i]`."""
    for i in range(out_start.size - 1):
        position = 0
        o = out_start[i]
        for j in range(elem_off[i], elem_off[i + 1]):
            value = values[j]
            if value < 0:
                position += int(-value)
            else:
                if value > 0 and (bins <= 0 or position < bins):
                    out_bins[o] = position
                    out_intensity[o] = value
                    o += 1
                position += 1


def _k_rlz_encode_sizes(bin_index, scan_start, max_skip, sizes):
    """Stream length of each scan: one entry per point, and one per `max_skip` of gap."""
    for i in range(sizes.size):
        cursor = 0
        count = 0
        for j in range(scan_start[i], scan_start[i + 1]):
            gap = bin_index[j] - cursor
            if gap > 0:
                count += (gap + max_skip - 1) // max_skip
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A whole frame's blobs at once: `(counts, bin_index, intensity)`.

    `counts[i]` is how many points blob `i` produced, which is the CSR row pointer
    without a second walk; `bin_index` and `intensity` are every blob's points
    concatenated in blob order. `bins` drops points past the frame's bin axis, 0 to
    trust the stream.

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
            bin_index, intensity = rlz_decode(np.frombuffer(raw, dtype=dtype), bins)
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

    values = dst.view(dtype)
    elem_off = dst_off // dtype.itemsize
    kernels.rlz_count(values, elem_off, int(bins), counts)
    out_start = np.zeros(len(payloads) + 1, dtype=np.int64)
    np.cumsum(counts, out=out_start[1:])
    out_bins = np.empty(int(out_start[-1]), dtype=np.int32)
    out_intensity = np.empty(int(out_start[-1]), dtype=dtype)
    kernels.rlz_fill(values, elem_off, int(bins), out_start, out_bins, out_intensity)
    return counts, out_bins, out_intensity
