# Manual server and receiver verification — October 13, 2026

Verified by operating the browser app through its visible controls with a real match recording. No built-in test suite was used for this verification. Editing took place in an isolated backend with copies of saved projects; the four original project files were compared byte-for-byte afterward and were unchanged.

Players A/B were Team A and C/D were Team B. The opening pair was B serving to C. Nine clips were marked using the playback slider, Mark start, and Mark end, then scored with alternating Ace and Double Fault outcomes as test inputs (not observations of the recording).

Displayed pairs on creation and when reselecting every clip:

| Clip | Server | Receiver |
| --- | --- | --- |
| 1 | B | C |
| 2 | D | A |
| 3 | D | B |
| 4 | A | C |
| 5 | A | D |
| 6 | C | B |
| 7 | C | A |
| 8 | B | C |
| 9 | B | D |

## Problems reproduced and corrected

- Before an outcome was entered, the clip-end panel displayed the unchanged pair as the next pair and labeled the unchanged score as the score after the clip. It now retains the entering score and prompts for an outcome before showing the next pair.
- A result receipt from clip 9 remained visible when clip 1 was selected, naming D as the aced receiver while clip 1 correctly showed C. Receipts now identify their clip and clear on selection and other edits. Saved receipts are cleared when projects reopen, avoiding stale names after reconciliation.
- The selected and following pairs now explicitly say This clip and After this clip.

## Manual checks after editing

- Reclassified the opening point as Redo: clip 2 became B to C. Undo restored the point, Redo reapplied it, and Undo restored it again.
- Deleted the opening clip: the remaining opening clip became B to C. Undo restored all nine clips.
- Marked the opening clip Not a rally: clip 2 became B to C. Restoring it made clip 2 D to A again.
- Reopened the test match after restarting the backend: assignments persisted and the old receipt was absent.
- Added an unresolved opening clip: the panel prompted for its outcome without presenting a repeated next pair. Scoring it restored the full sequence.
- Played the opening clip to its automatic stopping boundary: the selected pair remained B to C; After this clip showed D to A and the score updated to 1–0.

The frontend type check and production build, Python syntax compilation, and git diff whitespace check also passed. This is scoped to serving display and the exercised editing paths; it is not a full release validation.
