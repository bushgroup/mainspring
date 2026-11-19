# The UIMF format as mainspring reads and writes it

UIMF is the SQLite database in which PNNL's ion mobility acquisition software, and SLIM
instruments generally, record a run. This page describes the format as mainspring uses it: the
tables, how a stored spectrum is encoded, how bins become m/z and scans become arrival time, and
the ten parameters that only a file mainspring created carries. The reference for the format is
PNNL's [UIMF-Library](https://github.com/PNNL-Comp-Mass-Spec/UIMF-Library), and mainspring's
reader and writer follow it except where this page says otherwise. The code that implements each
section is named in it, so a statement here can be checked against the source of the release it
came with.

## The container

A UIMF file is a single SQLite database. mainspring reads five tables and writes all of them:

| Table | What it holds |
|---|---|
| `Global_Params` | One row per dataset-wide parameter: `ParamID`, `ParamName`, `ParamValue`, `ParamDataType`, `ParamDescription` |
| `Frame_Param_Keys` | The file's dictionary of per-frame parameters: `ParamID`, `ParamName`, `ParamDataType`, `ParamDescription` |
| `Frame_Params` | One row per frame and parameter: `FrameNum`, `ParamID`, `ParamValue` |
| `Frame_Scans` | One row per stored scan of a frame: `FrameNum`, `ScanNum`, `NonZeroCount`, `BPI`, `BPI_MZ`, `TIC`, and the compressed spectrum, `Intensities` |
| `Global_Parameters`, `Frame_Parameters` | The fixed-column forms of the first three, which files written before about 2015 hold instead |

Every value in the parameter tables is stored as text and typed by `ParamDataType`, which is one
of `System.Int32`, `System.Double`, `System.Single` or `System.String`. A file may carry the
modern tables, the fixed-column ones, or both. The reader takes the modern tables when they are
present and falls back to the fixed-column ones, as UIMF-Library does, and the writer
(`mainspring.uimf.UimfWriter`) writes both by default so that PNNL's own tools open what it
produces.

The list of frames is read from `Frame_Params` rather than from `NumFrames` in `Global_Params`,
because `NumFrames` is only what the writer had said by the time the file was read. On a file
still being acquired the two can differ, and the frame list is the one that is current.

**Journal mode.** A file that `UimfWriter` creates is in SQLite's write-ahead log (WAL) mode,
whereas the files PNNL's acquisition software creates use the default rollback journal. The
acquisition console that fills a run's scans opens the file without creating it, and SQLite's
journal mode belongs to the file rather than to a connection, so the console writes under the
mode the creator chose ([clockwork's console
protocol](https://github.com/bushgroup/clockwork/blob/main/docs/console-protocol.md#division-of-uimf-writing)).
WAL mode is what lets a reader follow a run without holding up the writer: a reader sees the file
as it stood when its query began, and the writer goes on appending beside it. A WAL file that
was not closed cleanly leaves a second file beside it, `<name>.uimf-wal`, holding the newest
commits, and the two must be copied together.

## Frames and scans

A frame is one ion mobility separation, or the sum of several, recorded as a grid of scans by
time-of-flight bins. `Scans` (a frame parameter) is the number of scans, i.e., time-of-flight
extractions along the arrival-time axis, and `Bins` (a global parameter) the number of
time-of-flight bins of width `BinWidth` nanoseconds in each. On SLIMPHONY a frame is 5,000 scans
by 114,688 bins of 1 ns. `Accumulations` is the number of separations summed into the frame, so a
stored intensity is the sum over that many pushes of the pusher for that scan and bin.

`Frame_Scans` holds a row only for a scan that recorded signal. The acquisition console writes a
row for each scan whose spectrum holds anything, and for scan 0 whatever it holds, so a scan with
nothing above the digitizer's zero-suppression threshold has no row at all. Neither the number
of rows nor the highest `ScanNum` therefore says how many scans the frame has. The arrival-time
axis always runs over `Scans`, from the frame parameter, and the first stored row need not be
scan 0. Files written by PNNL's software in 2011 store every scan, empty or not, and are read the
same way.

The four summary columns are written beside each spectrum. `TIC` is the sum of the scan's
intensities and `BPI` its largest, and mainspring's decoder reproduces both exactly on every file
it has been tested against, which is what `uimf-info --verify` checks. `NonZeroCount` counts the
entries the writer stored, explicit zeros included, so it is an upper bound on the number of
points rather than a count of them. `BPI_MZ` is the m/z of the base peak, computed by the writer,
and its convention varies between writers: some subtract `TimeOffset`, one stores the base peak's
bin index instead, and one never computed it. mainspring writes the m/z when the frame has a
usable calibration and the bin index when it does not, and neither its reader nor its viewer
uses the column.

## The stored spectrum

`Intensities` is a run-length-zero (RLZ) stream of fixed-width little-endian numbers, compressed
with LZF. The element type is set once for the file by `TOFIntensityType`: `ADC` is a 32-bit
integer, `TDC` a 16-bit integer and `FOLDED` a 32-bit float, and an absent value means `ADC`.

To decode the RLZ stream, walk a bin counter from 0. A negative value `v` skips `-v` bins, a
positive value is the intensity at the current bin and then advances the counter by one, and a
zero advances the counter by one without recording a point. Bins after the last entry are empty,
which is why the decoder needs `Bins` to know where the axis ends. The LZF layer is Marc
Lehmann's liblzf as UIMF-Library carries it (`CLZF2.cs`): a control byte below 32 introduces that
many literal bytes plus one, and any other control byte a back-reference into the output already
produced. `mainspring.uimf.decode` implements both directions, with a pure Python reference and a
compiled path that must agree with it byte for byte.

One zero is not a bin. PNNL's earlier encoder ended a long run of empty bins with a skip of
-32,768 and then wrote a zero it did not mean, and UIMF-Library skips a zero that directly
follows -32,768 or the element type's minimum. mainspring skips it too, except on a file it
created (one carrying `MainspringWriter`, below), where the console writes zeros literally and
-32,768 followed by 0 is a real gap and a real zero. mainspring's own encoder writes a gap of
exactly 32,768 bins as -32,767 and -1, so what it writes reads the same under either rule.

## m/z

Bin `b` of a frame is placed in m/z by UIMF-Library's calibration:

```
t  = b × BinWidth / 1000        flight time in µs, BinWidth in ns
mz = (K × (t − T0))²            K = CalibrationSlope, T0 = CalibrationIntercept
```

with `K` and `T0` read from the frame's own parameters, so a file whose calibration changes from
frame to frame is placed frame by frame. `TimeOffset`, a global parameter whose description says
every bin must be offset by it, is not applied, which matches UIMF-Library's own calibration.
The residual-error polynomial `MassCalibrationCoefficienta2`
through `f2` is zero in every file seen and is not applied by UIMF-Library or by mainspring. A
frame whose `CalibrationSlope` is zero has no m/z axis, and the viewer then draws bins instead.
Bins earlier than `T0`, where the parabola would turn back up, are drawn at m/z 0 so that the
axis never decreases. The code is `mainspring.uimf.calib`.

## Arrival time

Scan `s` arrives at

```
t_ms = s × AverageTOFLength × 10⁻⁶
```

where `AverageTOFLength` is the frame parameter giving the mean interval between pushes in
nanoseconds. One scan is one pusher period, so the arrival-time axis of a frame of 5,000 scans at
129 µs per scan runs to 645 ms. Arrival time counts from scan 0, and the viewer's arrival offset,
which moves the origin, is a display setting rather than part of the file.

## The parameters only mainspring writes

To record what PNNL's parameter set has no name for, `UimfWriter` adds ten parameters of its own,
six global and four per frame. Each name carries the prefix `Mainspring` and each ID starts above
1000. PNNL's `GlobalParamKeyType` currently ends at 22 and its `FrameParamKeyType` at 52, and
UIMF-Library skips a row whose ID it does not know rather than failing, so these cost a PNNL tool
nothing. The prefix matters because UIMF-Library also resolves an unknown name by matching it
against its own list, so an unprefixed `Repetition` would become a standard parameter on the day
PNNL adds one.

| ID | Name | Table | Type | Meaning |
|---|---|---|---|---|
| 1001 | `MainspringWriter` | `Global_Params` | String | The software that created the file, e.g., `mainspring 1.17.1`. A file carrying it records when each frame was finished, so a missing completion marker means an unfinished frame |
| 1002 | `MainspringDetectorBits` | `Global_Params` | Int32 | The digitizer's bit depth, i.e., the resolution of its converter rather than the width of the stored numbers |
| 1003 | `MainspringRunOutcome` | `Global_Params` | String | How the run ended: `completed`, `stopped`, `failed`, or `incomplete` if nothing has said |
| 1004 | `MainspringRunReason` | `Global_Params` | String | Why a failed run failed, in the words its operator was shown, and empty otherwise |
| 1005 | `MainspringRepetitionsPlanned` | `Global_Params` | Int32 | How many repetitions the run asked for, over every method frame |
| 1006 | `MainspringRepetitionsAcquired` | `Global_Params` | Int32 | How many of those it finished, each counted once |
| 1001 | `MainspringMethodFrame` | `Frame_Params` | Int32 | The method frame this frame belongs to, counted from 1 |
| 1002 | `MainspringRepetition` | `Frame_Params` | Int32 | Which repetition of that method frame this frame is, counted from 1 |
| 1003 | `MainspringRepetitions` | `Frame_Params` | Int32 | How many repetitions of that method frame were asked for |
| 1004 | `MainspringFrameComplete` | `Frame_Params` | Int32 | 1 once the program writing the file has finished the frame, and absent until then |

The global and frame IDs are separate series, as PNNL's are, so 1001 names one parameter in each
table. The names, IDs, types and descriptions are `GLOBAL_KEYS` and `FRAME_KEYS` in
`mainspring.uimf.writer`, and the self-check pins the names as text. A program that writes UIMF
through mainspring and has parameters of its own declares them from ID 2000 upward, and the
reader looks every parameter up by name. Its global parameters are written when the file is
created (`GlobalSpec.extra`) or at any point during the run (`UimfWriter.set_globals`), and its
frame parameters with the frame (`FrameSpec.extra`) or once the frame is acquired
(`finalise_frame`). The parameters the writer derives itself, the TOF axis among them, cannot be
rewritten this way, because the frames already in the file were encoded against them.

**Grouping.** A frame with `MainspringMethodFrame` and `MainspringRepetition` is one repetition
of a method frame. A frame with `MainspringMethodFrame` and no repetition covers the whole method
frame, which is how a summed frame says what it is a sum of. A file with neither is ungrouped,
which is every file PNNL's software writes. `MainspringRepetitions` is on every grouped frame, so a
method frame with fewer frames in the file than it states was cut short.

**Bit depth.** `MainspringDetectorBits` describes the converter. The acquisition console stores a
14-bit sample as a 16-bit code, so a file declaring 14 holds values up to 65,535 a push, and
the viewer judges such a file against that ceiling. A file without the parameter has no bit depth
the viewer can read, and the viewer's `Bits` setting supplies one.

## When a frame is finished

Nothing in PNNL's format says that a frame is finished. The console reports a frame done before
its own writer has drained, and a frame stores only the scans that recorded signal, so neither the
row count nor the last scan number can be compared against anything. `UimfWriter` therefore
writes a frame in two phases. `add_frame` writes everything known before the frame is acquired,
including `Scans`, `Accumulations`, the calibration, the start time and the grouping, and
`finalise_frame` writes what is known only afterwards: the duration and `MainspringFrameComplete`
= 1.

The reader's `UimfFile.is_provisional` reads that marker on any file that carries
`MainspringWriter`, so on such a file whether a frame is finished is exact. A frame whose marker
was lost, for example to a power cut between the last scan and the marker, reads as unfinished
for good, which is the honest answer because the frame may well be short. A file without
`MainspringWriter` carries no markers, and the reader falls back to treating the last frame of a
file written to within the last five seconds as unfinished.

`MainspringRunOutcome` answers the same question for the run. A program that asks for it gets
`incomplete` from the first commit and replaces it with `completed`, `stopped` or `failed` when
the run ends, so a run that never closed still says `incomplete`. The reader reports `unknown` for
a file that does not carry it.

## Raw and summed files

The lab's acquisition software, [clockwork](https://github.com/bushgroup/clockwork), writes two
files per run through `UimfWriter`. The raw file holds one frame per repetition, with
`Accumulations` = 1 and both grouping parameters, except that a method run in clockwork's
`single_frame` mode acquires a whole method frame as one frame. The summed file, named `<name>.summed.uimf` and
known to the reader as `SUMMED_SUFFIX`, holds one frame per method frame, the scan-by-scan and
bin-by-bin sum of that method frame's repetitions, with `Accumulations` equal to the number of
repetitions summed and `MainspringMethodFrame` without a repetition. The summed file has the
shape that PNNL's software writes, with the grouping added. The [README](../README.md#raw-and-summed-files)
says more about the pair and how mainspring treats it.
