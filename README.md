# Roundnet Rally Editor

A local, macOS-focused desktop editor that turns a full roundnet/Spikeball
recording into reviewable rally clips and highlight videos. Detection proposes
cuts; you retain control over every decision. Video, analysis, corrections, and
learned profiles stay on your machine. No account or upload is required.

The application includes person/body-pose cues and local supervised calibration,
but it is not a pretrained roundnet action-recognition system. Real-game accuracy
has not yet been established on a labeled benchmark.

## Included workflows

- Import MP4, MOV, M4V, MKV, AVI, MTS/M2TS, and WebM media supported by your local
  FFmpeg installation; inspect duration, dimensions, frame rate, and codec.
- Select a playing-area rectangle, mark the net, and draw up to four serving-zone
  polygons on a representative source frame.
- Combine motion, spatial spread, audio transients, player tracks, serving
  formation, pose, and net-retrieval cues. Inspect signals on the timeline.
- Review uncertain clips first, loop one clip, or preview all kept clips in
  sequence. Adjust, add, split, merge, reject, restore, and mark clips reviewed.
- Save editable projects, recover autosaved work, and undo/redo rally edits and
  annotations. Original media is never modified.
- Set two teams, four players, and the starting server and receiver before editing.
  Classify each clip with one outcome button to update the score and known player
  statistics together. Add touch details only when you need deeper statistics.
- Review player cards, team totals, the original
  RPR breakdown, and who aced whom; export CSV, JSON, or a PNG stat card.
- Export original-aspect, landscape, portrait, or square MP4 videos, optionally
  with assisted crop keyframes, a scoreboard, captions, PNG branding, and a
  previewable four-player end card with adjustable duration.
- Exchange original-source cuts through JSON or a restricted CMX3600 EDL.
- Save reviewed correction labels, train a local profile from several recordings,
  inspect held-out evaluation, and explicitly activate or disable the profile.

## Requirements and installation

- macOS is the desktop target. Core detection/export code is largely portable.
- Python 3.10 or newer.
- FFmpeg and FFprobe on `PATH`.
- For native macOS body-pose detection, Apple's Swift compiler must be available
  (normally supplied by Xcode Command Line Tools). The first analysis compiles a
  small local helper; later runs reuse its temporary cache. The system Vision
  model needs no separately downloaded model file. If unavailable, the app falls
  back to OpenCV's bundled HOG person detector and reports the fallback.

```bash
brew install python ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

PySide6 provides the desktop interface, OpenCV provides frame/person analysis,
and NumPy provides signal processing and local profile training. Optional Apple
VideoToolbox encoding has a software H.264 fallback. No GPU is required.

## A recommended first edit

1. Open a game video. Keep the original in a stable location. Enter both teams,
   all four players, the starting server and receiver, the initial score, and
   the points-to-win target in **Match setup** before editing clips.
2. Select the playing area generously enough to include the players' reach,
   excluding adjacent courts, spectators, trees, and traffic where possible.
3. Open **Net / Serve Zones**. Choose a useful frame, mark the net, and draw the
   regions from which players serve. These are optional supporting cues.
4. Run **Detect Rallies**. Analysis runs in the background with cancellation.
5. Choose **Next uncertain** and inspect the proposed cuts. Mark valid clips
   reviewed, adjust their boundaries, and add missed rallies. Use **Reject /
   Restore** for ball collection or other false detections. An unchecked valid
   rally is merely omitted from export; it is not a negative training example.
6. Review the full recording for missed rallies before checking **I reviewed the
   whole video for missed rallies**. Leave it unchecked after partial review.
7. Classify each valid clip in **Point / Highlight** as Ace, Double Fault, Service
   Break, Sideout, Defensive Break, Defensive Hold, Error, or Redo. For Error,
   choose the player responsible. Add touch details for percentages and RPR,
   star highlights, and add crop centers as needed.
8. Export a video, save correction labels, and save a named project if you want
   a portable editable copy of your work.

The **Needs review**, **Starred highlights**, and **Rejected / unchecked** filters
do not alter export selection. Rejected detections remain in the project and can
be restored. **Play All Kept** skips gaps in the source; **Loop selected clip**
helps inspect boundaries without changing them.

## Project saving and recovery

**File → Save Project** writes a `.roundnet.json` project containing the source
reference, rally identities and edits, review status, clip classifications,
tags, crop keyframes, player rosters and optional touch logs,
playing-area/court setup, settings, and available diagnostic signals. It does not
copy the original video or embed an external learned-profile file.

Changes are also autosaved to `.roundnet/recovery/` inside the application
directory. The editor remembers the latest recovery file and can restore it on
restart. Saves are atomic, so an interrupted save does not replace the previous
good snapshot with a partial file. Source size and modification time help detect
replaced media before applying old cuts. Keep named projects and source media
backed up: recovery is a convenience, not a separate backup service.

Moving a project and its source together preserves their relative reference.
For a missing source, opening a named project allows relinking. **Edit → Undo /
Redo** keeps up to 100 in-session edit snapshots; the undo stack itself is not
saved across restarts.

## Keyboard controls

Shortcuts apply when a text or number editor does not have focus.

| Key | Action |
| --- | --- |
| Space | Play/pause |
| Left / Right | Seek 0.25 seconds |
| Shift+Left / Shift+Right | Seek 5 seconds |
| Comma / Period | Seek one nominal source frame |
| `[` / `]` | Previous/next rally |
| A | Add a rally around the playhead |
| W, then X | Mark a new rally's start and end |
| I / O | Set selected rally's start/end to playhead |
| S / M | Split / merge neighboring rallies |
| P | Preview selected rally |
| N | Select next uncertain, unreviewed rally |
| F / R | Star highlight / mark reviewed |
| 1 / 2 | Credit the point to Team A / Team B |
| Delete / Backspace | Reject or restore selected rally |
| Command+S / Command+Z | Save project / undo |

## Detection and false-positive reduction

At a reduced analysis frame rate, grayscale changes measure global and
playing-area motion. Spatial spread rewards distributed activity over a single
compact mover. FFmpeg extracts a small mono audio stream for impact/transient
evidence. Robust normalization reduces sensitivity to camera exposure and
recording volume; semantic scores such as spread are kept on their own bounded
scale rather than amplified to match each video's maximum.

The person-analysis layer uses macOS Vision body keypoints when available, or
OpenCV HOG bounding boxes otherwise. Short optical-flow tracks bridge person
detection frames. With the marked net and zones, the app can consider player
distribution, readiness, movement near the net, bending/retrieval evidence, and
pose-supported serve-like motion. These cues are not ball tracking, identity
recognition, or proof of a legal serve. HOG fallback cannot supply body keypoints.

The combined score is smoothed and segmented with hysteresis, sustained activity
and inactivity, minimum/maximum durations, pre/post-roll, and nearby-clip merging.
Compact candidates lacking a serve or distributed-play evidence can be left
unchecked as likely retrieval. They remain available for review. Serve and pose
signals are supporting evidence, not hard gates: a hidden server or missed person
should not alone veto a rally.

Useful controls in **Advanced Detection Settings** include motion/audio/serve
sensitivity, spread and serve weights, retrieval filtering, player tracking,
person-detection interval, court-context influence, rally threshold,
minimum/maximum duration, end-of-rally quiet time, pre/post-roll, merge gap, and
analysis FPS. Start with defaults and change one setting at a time while watching
the debug traces. Stationary footage with visible players is the best starting
point; higher analysis rates cost more processing time.

An optional local COCO17 YOLOv8/11-style pose ONNX model can be configured through
`pose_model_path`. The supported model contract is fixed 640-pixel input and raw
`[1,56,N]` predictions without NMS. No third-party model is bundled or downloaded.

## Points, highlights, and presentation

The editor does not automatically referee or identify players. After match setup,
choose one outcome for each clip. The starting pair determines the next server
and receiver under equal serving: the opening server serves once, later servers
serve twice, and win-by-two overtime uses one serve per turn. The point winner
follows from the clip outcome and that serving assignment. Changing an earlier
classification recalculates later assignments and scores. An unclassified clip
makes later assignments provisional until you classify it. The outcome buttons
guide you back to the earliest unresolved clip before scoring later serves.

Ace, Service Break, and Defensive Hold award the serving team a point. Double
Fault, Sideout, and Defensive Break award the receiving team a point. Error
requires the player who made the unforced error and awards the opposing team a
point. Redo keeps the clip but awards no point or statistics; the same server
and receiver are assigned to the next clip. Unchecked but valid points still count
toward the match; rejected detections do not. The configured initial score is
included in displayed totals. An end card with unresolved outcomes labels its
score as recorded and marks player totals that depend on those clips unknown.

### Player statistics

In **Point / Highlight**, click the outcome for the selected clip; the editor
records its known score and player facts together and advances to the next
unclassified clip. **Undo** restores a saved classification. The optional touch
dialog lets you tap players in touch order and record serve, receive, set, hit,
and defense details. Use **Undo touch** within that dialog to reverse a touch.

**Match statistics** shows known counts from classified clips, team totals, and
a navigable point log. Ace credits its inferred server and receiver; Error
credits its named player. Serve %, put-away %, set quality, defensive gets, and
other touch-derived values stay unknown until the needed touches are logged.
Export player CSV, full event JSON, or a shareable PNG card. Classifications
and optional touch logs are saved in projects and recovery and support undo/redo.

The **Original RPR** tab calculates Hitting, Serving, Defense, and Efficiency
using the published original model. It requires explicit full-match confirmation,
complete touch logs, known results, and serve/hit attempts for each rated player.
It is not a claim to reproduce later proprietary rating revisions. Player identity
and touch quality are entered manually, not detected from video. See
[statistics definitions and formulas](docs/statistics.md) for denominators,
incomplete-data handling, and attribution.

Stars mark highlights independently from whether a clip is enabled. Choose
starred-only export for a highlights cut. The scoreboard shows the score before
each included point, accounting for valid points omitted from the cut. Optional
captions use the clip's player/outcome/note. A supplied PNG is placed at the lower
right as branding. The export dialog previews the final-score player card and lets
you set how many seconds it stays on screen.

**Set assisted crop keyframes** lets you choose source timestamps and click the
desired framing center. Export smoothly moves between those centers and clamps
the crop to the source image. This is manual-assisted framing, not automatic ball
following. Without keyframes, aspect-ratio crops are centered. Preview the crop
carefully: portrait framing can exclude players on opposite sides of the net.

Exports read the original source, trim kept intervals, and concatenate them with
FFmpeg. Accurate non-keyframe boundaries require a high-quality re-encode; this
is not smart-copy export. Choosing a crop changes output dimensions, while the
original-aspect option retains source dimensions apart from even-pixel encoding
requirements. Original audio is included when present.

## Learning from your corrections

This is supervised calibration, not reinforcement learning. Correct source-time
intervals are direct target labels; there is no agent acting on delayed rewards.

**Save Correction Labels** writes two local sidecars without copying the video:

- Schema-v2 JSON records the source fingerprint, analysis settings/ROI, original
  predictions, reviewed intervals, explicit rejections, and review completeness.
- A compressed `.npz` stores aligned timestamps, features, and `1 / 0 / -1`
  rally/non-rally/ignore labels. Boundary margins and configured pre/post-roll are
  excluded from positive examples to avoid teaching idle padding as play.

Unreviewed background stays unknown. Unchecked valid clips stay ignored. Explicit
rejections are negative examples. Background can become negative only after you
confirm whole-video review; a completely reviewed video with no rallies is useful
negative data. Legacy full-review claims without explicit confirmation are not
silently trusted. Whole-video review should include checking for missed rallies,
not simply approving every suggested clip.

Use **Learning → Train / Evaluate Local Profile** to select correction files and
train a regularized local classifier. Training requires at least three distinct
original recordings, compatible saved features, and at least 20 reviewed samples
of each class. Different edits of the same original count as one recording. A
fold also needs both classes in its training set. These are minimum safeguards,
not a guarantee that the resulting model will generalize.

Each recording is held out in turn. The report compares learned and original
heuristic sample metrics. When fully reviewed, non-omitted reference intervals
are available, it also reports rally precision/recall, false positives per hour,
and start/end boundary errors. Partial labels do not produce whole-video event
accuracy claims. Collect varied cameras, lighting, players, and hard retrieval
examples; three very similar recordings provide only preliminary evidence.

Profiles are plain JSON, not executable pickles. Saving a trained profile does
not activate it: inspect the report and choose **Use Profile**. Subsequent
detection uses it; **Learning → Use Default Detector** restores default scoring.
Incompatible feature sets or invalid profiles produce a warning and fall back
to the default detector.

## Comparing with an external manual edit

**File → Export Edit Timeline (EDL / JSON)** preserves original source IN/OUT
times. **Import Corrected Timeline** reads source-time corrections into the
current video's rally list; review the imported cuts before confirming complete
labels. The import is a timeline replacement and can be undone.

Prefer this app's exact-seconds JSON for variable-frame-rate footage. The EDL path
supports single-source, straight-cut CMX3600 video events with non-drop-frame
timecodes, the original frame rate, and zero-based source timecode. Multi-reel
edits, transitions, drop-frame timecode, and nonzero source timecode offsets are
not supported. EDL export describes video cuts; relink original audio in the
external editor. The importer does not recreate external effects or annotations.

A rendered montage alone has lost its original timestamp mapping. Automatic
alignment against a manually edited MP4 and FCPXML import are not implemented;
use source-time JSON/EDL instead of treating montage timestamps as source labels.

## Headless analysis and tests

```bash
python -m detection.detector /path/to/game.mov
python -m detection.detector --help
python -m pytest -q
# Optional native desktop workflow tests (requires an active macOS session):
ROUNDNET_GUI_TESTS=1 python -m pytest tests/test_gui_workflow.py -q
```

The detector can write JSON with `--output-json` and diagnostic series with
`--include-signals`. Tests cover segmentation, timing, court/player evidence,
correction provenance, held-out training, source-time interchange, project
persistence/history, touch-order statistics and original RPR, crop/presentation plans, and real FFmpeg runs using generated
media. Native desktop smoke checks require a graphical macOS session. Synthetic
test results must not be read as measured real-roundnet accuracy.

## Remaining limitations

- Real-game precision, recall, and boundary quality still need a representative
  labeled roundnet dataset. No model has been trained on your games until you
  explicitly create a profile from your corrections.
- Camera movement, tiny/occluded players, busy adjacent courts, wind, music, and
  atypical serving angles can cause errors. Person detections and pose cues need
  review, and gathering balls is not guaranteed to be rejected every time.
- The main motion ROI is rectangular; serving zones are polygonal. Net/zone setup
  is manual and should be revisited when the camera moves.
- The app handles one source recording per project. Batch jobs, multi-camera
  synchronization, automatic score/identity recognition, and automatic ball
  tracking are not implemented.
- The application currently runs from Python; a signed standalone `.app` and an
  installer are not included.

## Project layout

```text
main.py                 application entry point
config/                 detection/export settings
models/                 rally metadata, player statistics, projects, and edit history
detection/              signals, court/player cues, scoring, and segmentation
training/               correction labels, local calibration, and evaluation
video/                  media metadata, FFmpeg export, presentation, and EDL
ui/                     player, timeline, review, statistics, court/crop, and learning dialogs
docs/                   statistics workflow, metric definitions, and rating formulas
tests/                  automated regression and generated-media integration tests
```
