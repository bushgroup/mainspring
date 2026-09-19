# mainspring user guide

mainspring shows one frame of a UIMF file as a heat map of m/z against arrival time, with the
mass spectrum of the visible region above it and the arrival-time distribution of the same
region beside it. This guide covers what the window shows, what each control does, and how to
read the two intensity numbers the info panel reports. Every control also carries a
one-sentence tooltip on hover, which says what it does; this document says why you would reach
for it.

## Starting the viewer

To run the viewer from a source checkout, follow the fresh-clone steps in the top-level
[`README.md`](../README.md) and then:

```
uv run mainspring
uv run mainspring FILE.uimf
```

The same `README.md` links the installer, which needs no Python installation, and covers
building the executable and compiling the installer yourself. That installer is per-user and
needs no administrator rights, because the viewer writes nothing outside your own profile.
Windows 11 is the platform the viewer is tested on. The installer also offers to open `.uimf`
files with mainspring; accept it and a double-click in
Explorer opens the viewer directly, unless something else was already set to open `.uimf` on
that machine, in which case mainspring is offered as a choice rather than replacing it
(Settings > Default apps).

The title bar names the open file and then the version, so a viewer left on a bench says
which build it is without anyone opening a menu.

## Opening a file

Use `Ctrl+O`, or `File > Open`, or pass a path on the command line, or double-click a `.uimf`
file in Explorer, or drop one onto the executable. The viewer remembers the directory you last
opened from and starts the dialog there.

Decoding runs on a background thread, so the window stays responsive while it works, and an
indeterminate progress bar sits in the status bar until the first image appears. The status bar
then reports the frame number, how many stored points it holds, and how long the open took. The
window title becomes the file's name.

A file opened from Explorer or a drag onto the executable that fails to decode gets a dialog
naming the file and the reason, since the window has nothing on screen yet to explain itself;
a file chosen from `File > Open` that fails reports the same reason in the status bar, since
you are already looking at the window.

File size and frame count are not what the open costs. A raw per-repetition acquisition of
5,000 frames over 83 MB opens and paints its first frame in under 0.2 seconds, because the
viewer reads one frame and rasterises it to the size of the window rather than loading the
file. Paging to another frame costs about 5 milliseconds however far into the file it is.

A file the instrument is still writing opens the same way, and the `Live` control then keeps
it up to date as the run continues. See [Following a file being
acquired](#following-a-file-being-acquired).

### Runs that are in two files

A run whose software did not close the file leaves two files behind, `260918_BK_003.uimf` and
`260918_BK_003.uimf-wal`, with the newest frames in the second one. A power cut on the instrument
PC leaves a run in this state, and so does a run that is still being acquired. The viewer reads
the pair and shows the whole acquisition, so this changes nothing about what is on screen. It
changes what you have to copy, and the status bar says so on the line that reports the open:
`Part of this run is in 260918_BK_003.uimf-wal and not yet in 260918_BK_003.uimf, so copy both
files together`.

Copying the `.uimf` on its own discards whatever was still in the log, and nothing reports the
loss. A 400 frame run cut off mid-write gave a file that opened without complaint and held 385 of
those frames. A run cut off earlier than that gives a file with no frames in it at all. Move the
two together, always.

Reading the pair does not merge them. The viewer opens files read-only and leaves both of them
byte for byte as it found them, so the `-wal` stays beside the `.uimf` for as long as nothing
writes to the run.

The one place the pair will not open is a folder you can read but cannot write to, such as a
share mounted read-only. Reading a log means writing an index file next to the database, so the
viewer reports `unable to open database file` and then names the log and the remedy, which is to
copy both files somewhere you can write to and open the copy.

## The window

![The mainspring window, showing a PNNL test file with the color scale set to log](images/window.png)

Five things fill the window above the status bar.

**The heat map** fills most of it. The horizontal axis is m/z, the vertical axis is arrival time
in milliseconds, and the color of each pixel is the intensity of the points that fall inside
it. Ticks point inward on all four sides, with values on the bottom and left.

**The mass spectrum** sits above the heat map and **the arrival-time distribution** to its
right, each separated from it by a small gap so that the baseline and its noise are readable
rather than running into the image. Both are projections of what is on screen rather than of
the whole frame: the spectrum sums the points inside the current arrival-time range, and the
distribution sums those inside the current m/z range. Narrowing one axis therefore sharpens the
other plot. Neither carries an axis of its own, because the heat map's axes already read for
the axis they share and their intensity axis rescales with every gesture. Their traces thicken
as the window grows, along with the ticks and the axis lines, so a maximised window on a large
panel reads at the weight a small one does.

**The color bar** is the strip in the far right column of the plot area. It reads in the units
`View > Color scale` selects, so on the `Linear` setting its numbers are stored intensities and
on `Log` or `Square root` they are the transformed values. Drag either handle to set the limits by
hand. Its gradient is one of four perceptually uniform color maps (`Viridis`, `Plasma`,
`Inferno`, `Magma`), chosen from `View > Color map`.

**The info panel** is docked on the right. `Ctrl+I` hides and shows it, and it can be dragged
out of the window and floated.

The screenshot above is PNNL's `9pep_mix` test file, which
[`tools/fetch_testdata.py`](../tools/fetch_testdata.py) downloads, with the color scale set to
`Log`. The diagonal bands are the multiplexed encoding that file was acquired with.

## Light mode

`View > Light mode` draws the plot area on white instead of black. It takes effect as you
tick it, and it costs nothing: the open file, the frame you are on, the ranges you have zoomed
to and any levels you have pinned are all where you left them. The viewer remembers the choice,
so the next session starts in the mode you last used. Black is the default.

The color map is not part of the mode. `Viridis` and its three companions read as themselves
whichever background they sit on, and a map that changed under you would make two figures of the
same frame hard to compare, so `View > Color map` stays yours to set. A frame with little in it
therefore draws as a dark rectangle on a white canvas, because dark purple is what the low end of
`Viridis` is.

The menu bar, the toolbar, the info panel and the dialogs follow Windows rather than this
setting. They are already light on a stock Windows install, which is what makes a light plot
area match the rest of the window.

## Moving around the frame

Every gesture is one step. There is no mode to switch into first and no zoom history to unwind,
because a wrong zoom costs one keystroke to undo.

| Gesture | What it does |
|---|---|
| Scroll wheel | Zoom about the pointer, both axes together |
| `Ctrl` + wheel | Zoom the horizontal axis alone |
| `Shift` + wheel | Zoom the vertical axis alone |
| Left-drag | Pan |
| Right-drag | Zoom to the box, applied when you release |
| `Shift` + left-drag | The same box, for trackpads and remote desktop |
| Double-click | Reset to the frame's full range |

The same two gestures work on either projection, where they act on the one axis that
projection shares with the heat map.

| Gesture on a projection | What it does |
|---|---|
| Right-drag | Zoom the shared axis to the band, applied when you release |
| `Shift` + left-drag | The same band, for trackpads and remote desktop |
| Double-click | Reset that axis alone, leaving the other where it is |

A projection is often where the peak you want is visible: the mass spectrum resolves an
isotope pattern that the heat map draws as a single column, so picking the pattern off the
curve is easier than boxing it on the image. Dragging along the curve sets the m/z range and
leaves the arrival-time range alone; dragging down the arrival-time distribution does the
reverse. Panning and the wheel do nothing on a projection, because a curve moved away from the
image it belongs to would no longer describe it.

| Shortcut | What it does |
|---|---|
| `Ctrl+O` | Open a file |
| `Home` | Reset the view to the frame's full range |
| `Ctrl+I` | Show or hide the info panel |

All three are on the menu bar as well: `Open` under `File`, and `Reset view`, `Info`,
`Light mode`, `Color map`, `Color scale` and `Text size` under `View`. `Aggregate`, `Type`
and `Bits` are under `Data settings`. There is no context menu on the heat map, because the right button is a
zoom gesture.

Zooming and panning are both clamped to the frame, so a gesture cannot leave it, and zooming in
stops when the visible range is two source elements across. On a SLIMPHONY frame that floor is
0.125 in m/z, which is about four TOF bins at m/z 530 and well inside an isotope pattern.

Each gesture is answered by a fresh render at the viewport's pixel size rather than by
magnifying the previous image. Requests are coalesced over 30 ms and only the newest is drawn,
so a burst of wheel ticks costs one image rather than one image per tick. The previous image
stays on screen and stretches while the next is being computed, which is what makes a fast drag
continuous.

The zoomed view below is the same frame and the same window, narrowed to m/z 480 to 820 and
arrival time 18 to 46 ms. Both projections have been recomputed over that region: the mass
spectrum now resolves individual peaks that the full-range view drew as one column, and `TIC in
view` has fallen from 126,118,062 to 51,225,603.

![The same window zoomed to a narrow m/z and arrival-time range](images/window-zoomed.png)

## The View menu

### Swap X/Y

Puts arrival time on the horizontal axis and m/z on the vertical. The two projections follow,
because each is named for the axis it projects onto rather than for a quantity: the plot above
the image is always the projection onto the horizontal axis and the plot to its right always
the projection onto the vertical one.

The visible region is carried across the swap rather than reset, so the same bins and scans
stay on screen.

### Raw units

Shows the TOF bin index and the scan number in place of calibrated m/z and arrival time. Use it
when the question is about the instrument rather than about the sample, since a bin identifies
the digitizer sample a count came from and an m/z does not.

This also sets what the cursor readout reports, since the readout names the axes it is on. The
visible region is carried across the toggle. A frame the writer never calibrated is shown in
raw units whatever this setting says, because a plausible-looking m/z axis over uncalibrated
data is worse than an honest bin axis.

### Color map

The gradient the heat map and the color bar are drawn with, one of `Viridis`, `Plasma`,
`Inferno` and `Magma`. All four are perceptually uniform, so a step in color is a step in
intensity and never an artefact of the map. The viewer offers no rainbow map for that reason.

### Color scale

How intensity maps onto color. `Linear` is proportional. `Log` and `Square root` compress the
dynamic range, which is what to reach for when one peak is bright enough to leave the rest of
the frame flat. All three are display transforms only. The cursor readout and the info panel
always quote the untransformed intensity, so switching the color scale changes the picture and
no number.

### Text size

Scales every piece of text in the window together, from 100 to 200 per cent in four steps: the
menus, the toolbar, the info panel, the status bar, the axis labels and the tick values. The
axes widen to keep their values readable, and the projections widen with them so that
everything stays lined up. Nothing else moves: the open file, the frame, the ranges you have
zoomed to and any pinned levels are all where you left them, and the viewer remembers the
choice.

Reach for it on a large monitor, or on an instrument PC you read from across the bench. The
viewer's type was sized for a 1000 by 700 window at 96 dpi and does not follow the window on
its own.

## The Data settings menu

Three controls that say what the numbers on screen are, rather than how they are drawn.
None of them is touched in the ordinary course of looking at a frame: the aggregate and the
bit depth are set once for a session and the type filter once for a file.

### Aggregate

A full-range view of a SLIMPHONY frame puts about 700 source elements inside each screen pixel,
114,688 TOF bins by 5,000 scans drawn 1,200 pixels by 700, and this chooses what the pixel then
shows. `Sum` adds them, which conserves total intensity and is the default. `Max` takes the
largest, which keeps a one-bin spike visible in a view where summing would bury it under a
broad neighbour.

The cursor readout in the status bar names the aggregate beside every intensity it quotes, for
the same reason: a summed pixel is a total over however many bins and scans it covers, and it
is not a stored intensity.

### Bits

The digitizer's bit depth, from 1 to 32, which the per-push readout uses to work out what
fraction of full scale a count is. PNNL's parameter set has no name for bit depth, so on a file
their acquisition software wrote this is a setting rather than something the viewer can read.
SLIMPHONY's current digitizer is 8-bit and clockwork will use 14-bit. Set it before you read
anything off the per-push line.

A file clockwork wrote stores the bit depth of the digitizer that produced it. On such a file
the control shows the stored value and cannot be changed, because the number in use is the
file's. The per-push readout says which of the two it is using.

### Type

Restricts the frame spinner and `Sum all` to frames of one type: `MS1`, `MS2`, `Calibration`,
`Prescan`, or `All frames`. A file whose writer used a code this list has no name for shows it
as `Type N` rather than hiding those frames.

## The Help menu

### User guide

Opens this guide, as it was when this version of mainspring was built, in a window of its own.
It needs no network, which matters on an instrument PC that has none. The window is not modal,
so the viewer stays usable while it is open, and it keeps your place if you close it and open
it again.

### User guide online

Opens the guide on the web, which is the newest one rather than this version's. Use it when you
want to know what a later release does.

### About mainspring

The version, the commit the build came from, and the license. The version is also in the title
bar, which is the quicker place to read it.

## The toolbar

### Keep ranges

Keeps the current m/z and arrival-time ranges when the next file is opened, instead of resetting
to the new frame's full extent. Set the window once and page through twenty acquisitions in it.
The setting survives restarts.

Moving between frames of one file always keeps the view, whether or not this is ticked. Keep
ranges is about what happens across a file open.

### Keep levels

Keeps the color bar's current limits instead of rescaling them to each new image. Tick it and
the limits on screen at that moment are pinned, across new frames and new zooms alike, until
you untick it. Two frames drawn under one set of limits can be compared by eye; two frames each
scaled to its own maximum cannot.

### Info

Shows and hides the info panel, on `Ctrl+I`. The panel's own close button is the same switch.

### Frame

Goes to a frame by number, within the frames the type filter allows. Typing or stepping to a
number outside that set snaps to the nearest frame inside it. The view is preserved.

### Method frame, Rep, and Sum method frame

These three appear only on a file that records which method frame each of its frames belongs to.
clockwork writes that record; PNNL's acquisition software has nowhere to put it, so on their
files the controls are absent and everything else behaves as described above.

A clockwork raw acquisition stores one frame per repetition. A method frame asking for 100
accumulations is therefore 100 consecutive frames in the file, and a session is many method
frames. `Method frame` and `Rep` are the two axes of that arrangement. Stepping `Method frame`
holds the repetition and shows the same point of each successive experiment; stepping `Rep`
holds the method frame and shows one experiment repetition by repetition, which is how you see
whether the repetitions drift. The `Frame` spinner still reaches any frame by its file number,
and all three stay in step.

Beside `Rep` the toolbar reports how many repetitions the method asked for. When a method frame
holds fewer frames than that, the readout gives both numbers, as in `of 87, method asked 100`.
A method frame reads short when a run was cancelled, when a power failure ended it, and while
it is still being acquired.

`Sum method frame` adds every repetition of the method frame on screen into one heat map. That
sum is the summed arrival-time distribution of one ion mobility experiment, which is what a
per-repetition file has to be added back up into to be read the way a summed file is. A method
frame of 100 frames takes about half a second.

### Sum all

Adds every frame the type filter allows into one heat map, drawn in place of the current frame.
A progress dialog counts the frames as they are read and cancels cleanly, leaving the frame
already on screen. Cancelling discards the partial total rather than showing it, because a sum
over an unknown number of frames is not a quantity anyone can use.

The status bar names the result as a sum and how many frames went into it, so a summed image is
never mistaken for a single frame. On a raw per-repetition file of 5,000 frames the whole sum
takes about 20 seconds, so use `Sum method frame` when one experiment is what you want.

### Live

Watches the open file for what the instrument writes to it, once a second, and keeps the frame
spinner, the type filter and the repetition count in step with what is there. `Show`, beside it,
decides what following does with each new frame. Both are covered in [Following a file being
acquired](#following-a-file-being-acquired).

### Show

`Fixed frame`, `Newest frame`, `Method frame sum`, or `Sum newest frames`. Available while
`Live` is on, and described with it below.

### Frames

How many of the newest finished frames `Sum newest frames` adds up, from 1 to 200. Five is the
default, which is about five seconds of acquisition on SLIMPHONY. A change takes effect on the
frames already in the file rather than at the next one, so the picture answers as you step the
number, and the number you set is remembered between sessions.

## Following a file being acquired

To watch a run as it happens, open the file the acquisition is writing and turn on `Live`. The
viewer then asks the file once a second what has been added to it. Nothing about the acquisition
changes: every read is a separate read-only connection that is opened, used and closed, which is
what keeps the viewer out of the writers' way.

Each poll costs about 10 milliseconds of query on a file of 5,000 frames, and a frame becomes
visible within 3 milliseconds of the software that wrote it saying it is finished. The second
between polls is the whole of the delay you see.

`Show` decides what happens to the view:

- **`Fixed frame`** leaves the view exactly where you put it. The frame spinner's range grows,
  the repetition count beside `Rep` grows, and the frame on screen stays the frame you chose.
  This is the default, and it is what you want while studying one frame of a run that is still
  going.
- **`Newest frame`** moves to each frame as it arrives, including the frame being written at
  this moment. A frame is about one second of acquisition and its scans reach the file in
  batches, so the frame fills in front of you rather than appearing whole.
- **`Method frame sum`** keeps a running total of the finished repetitions of the method frame
  being acquired, which is the summed heat map a finished file holds. It appears only on a file
  that records how its frames group, which today means a file clockwork wrote. The total is
  recomputed when a repetition finishes and not while one is being written, so it only ever
  grows.
- **`Sum newest frames`** totals the newest finished frames instead, as many of them as the
  `Frames` box beside it asks for. The window of frames moves as the run continues, so the
  integration time stays the same however long the run is, and it is offered on every file
  rather than only on one that records how its frames group. A run that has not yet written
  that many frames is summed as far as it goes, and the status bar says how many frames went
  in.

The two sums leave out the frame being written, for the same reason: a total that included it
would change every time it was recomputed and would settle on wherever you happened to stop.
Each is recomputed only when the set of frames it covers has changed, so a poll that finds
nothing new costs one query and no reading.

The zoom, the color levels and every other setting are untouched by any of this. Following
changes which frame is on screen, never how it is drawn.

### Starting the viewer on a run in progress

To open a file already following it, pass `--follow` on the command line, with `--show` to
choose what following does:

```
mainspring FILE.uimf --follow --show newest
```

`--show` takes `fixed`, `newest`, `method-sum` or `rolling-sum`, which are the four entries of
the `Show` box in the order they appear in it. The window comes up watching the file, in the
mode that was asked for, with nothing to find on the toolbar first. This is how another program
opens a run it is writing, and the lab's own acquisition software uses it for the run in
progress. An option this viewer does not offer is an error rather than a file by that name, and
the process exits 2 without opening a window.

A file that cannot be followed, one on a network drive, opens anyway. The status bar says why
it is not being followed and the viewer runs as it always does, because the file is what you
were trying to look at.

### Frames that are not finished yet

A frame the instrument may still be adding scans to is drawn like any other and labelled
`still being written`, in the status bar under the plot and beside `Frame:` in the info panel.
The viewer never keeps such a frame in its cache, so every poll reads what is actually in the
file rather than what was there a second ago.

A file clockwork wrote says frame by frame when a frame is finished, so the label is exact. No
other writer records it, and on their files the viewer falls back to treating the last frame of
a recently written file as unfinished. That fallback errs towards saying `still being written`
about a frame that is in fact complete, which costs a re-read and nothing else.

A frame whose completion was lost to a power failure reads as unfinished for good. That is
honest rather than a defect, because such a frame may well be short. Its data are intact and
every frame before it reads as complete.

### When the run ends and the file goes

A method that does not keep its raw file deletes that file when the run closes, once the summed
companion beside it exists to have replaced it. The raw file is the one that grows during a run
and so the one you were following, which means a successful run ends by making the file on screen
disappear.

The viewer stops following and says so by name, for example `Stopped following: 260918_BK_003.uimf
is no longer there`. Everything already read stays on screen, and beside that message a button
offers the companion that survived, `Open 260918_BK_003.summed.uimf`. Nothing repaints until you
click it. The companion holds one frame per method frame rather than one per repetition, so it is
the file your data ended up in and it is today's file shape.

A run cut short before its first fold leaves no companion. The message is the same and there is
nothing to offer alongside it.

### What cannot be followed

`Live` is refused on a file that is not on a drive attached to this machine, and the status bar
says so. Following means reading a database while another process writes it, and the two
processes coordinate through shared memory that only exists when both are on the same machine.
Opening and reading a file over a share is unaffected. It is only following one that is refused.

Following is also switched off whenever another file is opened, and it is not remembered between
sessions. It describes one acquisition rather than a way of working.

## The status bar

The status bar has two zones.

The left one is shared. Most of the time it is the cursor readout, which reports the two axis
values under the pointer, in the units the axes are in, and the intensity drawn at that pixel
labelled with the aggregate that produced it. `Raw units` decides which unit system appears,
since the readout names the axes it is on. The intensity is read back out of the image on
screen rather than recomputed, so it describes what you are looking at.

The same slot carries what the viewer has just done: the open message, the frame message, and
anything else it has to say. An ordinary message holds the slot for about five seconds and then
gives it back to the readout. A failure or a refusal holds it until something replaces it, so a
`Live` that was refused, or an open that failed, stays on screen rather than disappearing while
you are looking elsewhere. A button appears beside the message when a run you were following
ends by discarding its raw file, offering the summed companion that survived.

On the right is the peak of each projection, and the total. Where the mass spectrum peaks and
how high, the same for the arrival-time distribution, and then the sum of every stored point in
view. Each peak is named by the axis it is on, so the two swap over when `Swap X/Y` does, and
each height is a sum over the range in view on the other axis, like every value in a
projection. All three move as you zoom. The total is the same number the info panel calls
[TIC in view](#tic-in-view).

## The info panel

The upper half of the panel is the file's own parameters, global first and then the open
frame's, exactly as the file stores them. It is not a fixed list. Whatever keys the file's
parameter tables carry are what the tree shows, so a writer's optics voltages and its own
private keys appear alongside the handful the viewer itself parses.

Drag the panel's left edge to make it wider, and the value column widens with it. The viewer
remembers the width. A value still too long for the column is shown whole on hover, so a
calibration coefficient cut off at an ellipsis is one pointer away from being readable.

The lower half is one statement about the frame and four readouts, all of them describing the
image on screen rather than the whole frame, and all of them recomputed on every view change.

### Frame

`complete`, or `still being written` for a frame the instrument may be adding scans to. A
summed heat map shows `-`, because the question belongs to the frames that went into it and the
status bar names those. See [Frames that are not finished
yet](#frames-that-are-not-finished-yet).

### Max intensity in view

The largest single stored intensity inside the visible region. This is a stored value even when
`Aggregate` is set to `Sum`, so it is a detector reading rather than a pixel total, and it is
the number the per-push line divides by `Accumulations`.

### Per push

A UIMF intensity is the ADC sum over the frame's `Accumulations` time-of-flight pulses, and one
pulse is one push. The largest single-push value in view is therefore the maximum above divided
by `Accumulations`, reported here in ADC counts and as a percentage of full scale at the bit
depth the `Bits` control is set to. This is the number that says whether the detector is
saturating.

The readout states the `Accumulations`, the bit depth, and where the bit depth came from, so
that a count is never quoted without the two numbers that produced it. A file clockwork wrote
carries the digitizer's own bit depth and the readout says `from file`. Every other file leaves
the depth to the `Bits` control and the readout says `from setting`. Note that in the second
case the percentage is only as right as the setting. Quote a per-push value with the
`Accumulations` and the bit depth, or do not quote it.

### TIC in view

The total stored intensity inside the visible region, summed over every point in it. At full
range this equals the sum of the file's own `TIC` column for the frame.

### Points in view

How many stored, non-zero points fall inside the visible region. A UIMF frame is stored as the
points that exist rather than as a dense array, and this is a count of those, not of screen
pixels.

## Exporting a figure

`File > Export PNG` and `File > Export PDF` write what is on screen to a file. The heat map,
the mass spectrum and the arrival-time distribution all go in, at the zoom and under the
color settings they are drawn with. The color bar is left out. The figure takes the
background of the mode it was exported in, so tick `Light mode` first for a figure going into
a paper or onto a white slide.

Both entries ask where to write the file. `Export PNG` then asks at what resolution, with four
choices: 96, 150, 300 and 600 dpi. At 96 dpi the figure is the window's own pixels, one for
one, and at 300 dpi each axis carries 3.125 times as many; the dialog names the pixel size and
the size in inches before you commit to it. A figure over 50 megapixels is refused, which a
maximised window on a 4K display reaches at 600 dpi.

`Export PDF` asks nothing beyond where to write it, and names the page size. A PDF page is
vector at the figure's own size in inches, so there is no resolution to choose.

What the resolution buys is the type and the lines. The axes, the ticks, the labels and both
projections are drawn again at the higher density, so a 300 dpi figure carries crisp type at
print size rather than a magnified screenshot of 96 dpi type. The heat map itself is the image
already on screen, enlarged with square pixels, so a figure is exactly the picture you looked
at and chose to export: the same colors in the same places, the same color limits, the same
peaks resolved and the same ones not.

That is deliberate, and it is a change from 1.0.0, which re-rasterised the frame at the
export's own resolution. A heat map pixel is an aggregate over the bins and scans inside it,
so a finer image is a different picture: the color map shifts, blobs separate into peaks and
faint features appear that were not on screen. To see more detail, zoom in and export that.

A PDF carries the axes, the ticks, the labels and both projections as vector drawings, so they
stay sharp at any magnification, and it embeds the heat map as an image. Its page is the
figure's own size in inches.

## What the viewer remembers

Every toolbar toggle, the color map, the color scale, light mode, the text size, the
aggregate, the detector bit depth, how many frames `Sum newest frames` adds up, whether the
info panel is showing and how wide it is, the export resolution, the window's size and
position, and the directory you last opened from are all saved when the viewer closes and
restored when it starts. On Windows they live under
`HKEY_CURRENT_USER\Software\University of Washington\mainspring`. Deleting that key returns
every setting to its default.

The pinned color levels and the current view range are not among them. Keep levels and keep
ranges persist as switches, and what they hold is whatever is on screen in the session where
you turn them on.

`Live` is not remembered either, and it is the one toolbar control that is not. It says
something about one file rather than about how you like to look at data, and a viewer that
started polling every finished acquisition anyone opened would be doing work against nothing.

The saved bit depth is the `Bits` setting, and opening a file that stores its own does not
overwrite it. Close such a file and the control returns to the number you set.
