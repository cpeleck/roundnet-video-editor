# Detector benchmark

Open **Admin benchmarks** in the local Mac browser editor. The local owner has
the `admin` capability automatically; the app has no account database. Paired
phones remain editors and cannot capture benchmark data, run evaluations, read
reports, or inspect/cancel benchmark jobs. The backend checks the local connection,
Host, and Origin; hiding the button is not the access control. Keep the normal
loopback development-server configuration when using admin tools.

## Collect ground truth

1. Start with **8–10 distinct original recordings**, covering different games,
   cameras, lighting, and difficult ball-retrieval examples. Include a recording
   near 25 minutes if you want a measured timing number for that duration.
2. Run **Find rallies** to collect aligned detector features. Review the entire
   source recording, adjust each real rally's start and end, add missed rallies,
   and explicitly mark false proposals **Not a rally** or delete them.
3. Open **Admin benchmarks**, choose that match, confirm that you checked the
   whole recording, and choose **Save ground truth snapshot**.
4. Repeat for the other recordings. Edit and save again when correcting labels.
   The dataset uses the newest saved snapshot of each original source fingerprint,
   even if the video was renamed, moved, or opened as multiple projects.

Snapshots preserve source-time labels and aligned features. Saving a snapshot
does not edit the match or mark its statistics complete. All non-rejected rallies
are true labels, including real points excluded from export. This differs from
the general-purpose correction-label ZIP, where an export omission stays unknown.
Only explicit whole-recording confirmation makes background negative. A valid
fully reviewed recording with zero true rallies can contribute negatives; each
training fold still needs examples of both classes across its remaining sources.

## Evaluate and compare

Select the snapshots and choose **Evaluate recordings**. At least three distinct
sources and 20 labeled feature samples per class are required by the existing
trainer. Three sources are a functional minimum; use 8–10 varied recordings for
the intended project benchmark.

The job leaves each original recording out in turn, fits scaling and the local
classifier on the others, and predicts the held-out recording. The report compares:

- **Learned · held out:** the profile fitted without that recording, using fixed
  start/end thresholds of 0.50/0.35.
- **Existing heuristic:** the saved original combined score and its saved
  segmentation/retrieval settings, recomputed before comparing to labels.
- **Motion only:** the saved normalized playing-area motion (global motion if
  unavailable), with fixed start/end thresholds of 0.55/0.38 and the same saved
  temporal segmentation/padding rules. It has no audio, serve, pose, spatial-spread,
  or retrieval filter. Its thresholds are not tuned against held-out labels.

Avoid adjusting detector settings after seeing held-out results and then treating
the same sources as an untouched test set. Settings are saved with each analysis
and used by evaluation. Use additional recordings to verify subsequent changes.

A predicted enabled rally matches a labeled rally only if **both start and end
are within ±1 second**, inclusively. Matching maximizes the number of valid
one-to-one pairs; one long clip or duplicate proposals cannot claim multiple hits.

| Number | Definition |
| --- | --- |
| Precision | Matched predictions / all enabled predicted rallies. |
| Recall | Matched labeled rallies / all labeled rallies. |
| Median boundary error | Median of all absolute start and end errors from matched pairs, in seconds. |

Precision and recall pool TP/FP/FN across the dataset. The boundary median pools
individual errors rather than averaging fold medians. It describes only matched
rallies, so read it alongside precision and recall. A dash means an undefined
denominator or no matched boundaries. Boundaries are the reviewed clip/export
boundaries, including intended padding; they are not necessarily ball-contact times.

The same page shows each held-out recording's numbers, ground-truth and predicted
interval tracks, and exact source times. Completed reports remain available after
restart and can be downloaded as JSON. Training runs in a bounded worker with
progress and cancellation; it never replaces project cuts. A final profile is
saved beside the report for inspection; the benchmark does not activate it.

## Detection time and correction burden

New detection runs record full analysis wall time, excluding process startup and
queue wait. The report retains system, machine architecture, Python version,
player-analysis backend, and the analysis settings for interpreting timing.

- **Measured on 24–26 min sources:** median actual detection time for sources in
  that duration range, with the number measured. Individual durations/times appear
  in the report. Without such a source this remains unavailable.
- **Estimated for 25 minutes:** scale each timed source's elapsed time to 1,500
  seconds of footage, then take the median. This is explicitly an estimate, and
  it describes the full detector, not learned-profile inference from cached features.
- **Typical manual corrections:** median net interval corrections per recording,
  with added/restored, removed/rejected, and trimmed counts per source. Both edges
  changed on one clip count as one trimmed interval; splits and merges appear as
  changes to intervals. These are not editing-click counts or manual review time.
  Outcomes, scorekeeping, review clicks, and export exclusions do not count.

Older analyses have no measured timing. To preserve already-reviewed cuts, save
a project copy before rerunning detection. Compare durations and settings on the
same machine when reporting performance. Never turn missing measurements into
zero seconds or present synthetic test results as real-game accuracy.

Data lives under `.roundnet/web/benchmarks/`: immutable label/feature revisions
in `dataset/`, and completed `report.json` / `profile.json` files in `reports/`.
They remain local and do not contain video frames or audio. Back up source videos,
projects, and this dataset together when keeping reproducible results.
