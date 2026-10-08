# Point-by-point player statistics

## Workflow

When opening a new video, enter both teams, all four players, the starting
server and receiver, the initial score, and the points-to-win target in Match
setup before editing.
Slots A1, A2, B1, and B2 are stable identities: renaming a player updates the
displayed name without moving credits. This is one four-player match, not a
season database. The starting pair determines server and receiver for each
scored point under equal serving. The opening server serves once, subsequent
servers serve twice, and win-by-two overtime uses one serve per turn. After
A1 serves to B1, B2 serves to A2, then A1; A2 serves to B1, then B2;
B1 serves to A1, then A2. The same partner rotation applies to any starting pair.

Select a clip in Point / Highlight and click one outcome. The score and facts
known from that outcome update together; the editor advances to the next
unclassified clip. Editing an earlier clip recalculates later serving assignments
and scores. An unclassified clip makes subsequent assignments provisional. Undo
restores the prior classification. Classify the earliest unresolved clip first;
the outcome buttons pause on later clips until their server and receiver are
known. Clearing an earlier classification makes later serve-based credits
provisional until that gap is filled again.

| Outcome | Score and known player facts |
| --- | --- |
| Ace | Serving team scores; inferred server receives an ace and inferred receiver is aced. The serve was unreturnable with no reasonable chance to set. |
| Double Fault | Receiving team scores after two consecutive service errors by the server. |
| Service Break | Serving team scores through service pressure. |
| Sideout | Receiving team scores from a clean return without a rally. |
| Defensive Break | Receiving team scores after a defensive stop and conversion. |
| Defensive Hold | Serving team scores after a defensive touch and conversion. |
| Error | Choose the player responsible for an unforced error; the opposing team scores and that player receives an error. |
| Redo | Keep the clip with no score or statistics; the same server and receiver are assigned to the next clip. |

These are manual classifications. The app does not detect the outcome from video.
An unchecked valid clip still contributes to the match; a rejected detection
does not. Redo also contributes no point or player statistics.

In the browser editor, Defensive Hold, Defensive Break, and Error open a required
possession log. Start with the inferred receiver, then add the first-touch player
for each alternating possession. The first-touch player is the suggested hitter;
change that suggestion for hits on one or two. Select the final put-away player
and whether an opponent touched that hit. Error instead requires its player and
receive, set, or hit type. Save commits the outcome and player evidence together;
Cancel keeps the previous saved point, and Undo restores both in one step.

Quick logs default to one successful serve and no costly weak touches. Correct
first-serve faults, rims, lets, or costly receives/sets in the inline editor.
Ordinary possessions with the same first-touch player and hitter assume a partner
set; touch grades remain ungraded. These assumptions are saved separately from
player selections. Ace, Double Fault, Service Break, and Sideout default to no
rally: an ace serve, two faults, a failed receive, or a receiving-player put-away,
respectively. Redo has no point statistics. Edit quick log reopens the saved
selections. Full touch details allow correcting unusual player roles and adding
receive/set grades; converting a quick log to a full log requires grading or
completing its remaining evidence before full-log RPR eligibility.

The full touch log remains available in More details. In the legacy desktop
editor it is optional for classification. Log every serve attempt and touch,
including multiple possessions and both partners' contributions. Optional source
timestamps allow jumping back to a touch. A let is not a serve attempt.

These are editable suggestions, not enforced three-touch patterns. A player may
hit on the first or second contact. Use Receive for the serve, Defense for a hit.
When one physical contact both receives/defends and hits back onto the net, log
both roles for the same player (and the same timestamp, if used). These events
represent statistical roles, not an enforced physical-contact count.

Hit and Defense default to Auto. The next valid opposing net hit confirms a hit
was returned; a successful own-team net hit confirms a defensive get. At a fully
logged point's end, an unreturned winning hit becomes a put-away and an unconverted
defensive touch becomes touch-not-returned. A teammate's later error does not
remove an earlier successful get. No touch is separate from an unsuccessful touch.

Choose Strong, Weak, Error, or Ungraded for receives and sets. A weak touch is not
automatically a costly tough touch: explicitly check that flag only when it
indirectly contributed to losing the point. Set/hit errors and ace serves end the
sequence. An ace identifies both its server and the selected receiver.

Save drafts while reviewing. Check All attempts and touches only after verifying
the entire point and its winner. Contradictory sequences are rejected by the editor.
If an existing touch log becomes invalid after another edit, the report flags it
and excludes its player events rather than quietly producing wrong totals.

## Statistics and denominators

A classification contributes only facts it proves. Ace and aced counts, double
faults, named unforced errors, the winning team, and the serving/receiving teams
can be counted from the outcome and inferred assignment. A category alone cannot
establish all serve attempts, put-aways, set quality, defensive gets, or the
inputs to RPR. Those values stay unknown until enough touch detail is logged;
the editor does not estimate them from the category. A missing denominator shows
Unknown in the statistics dialog and a dash on the export card, never 0%.

- Serve %: in-play serves (including aces) / all attempts; lets excluded.
- Ace %: aces / all serve attempts. Aces : aced also tracks the targeted opponent.
- Put-away %: put-aways / known hit results, including hit errors. A winning hit
  counts as a put-away even if a defender touches it or a later opposing set fails.
- Gets: actual defensive touches followed by a successful return to the net.
- Defensive conversion %: gets / (gets + touches not returned); no-touch misses
  are displayed separately, not counted as defensive touches.
- Strong : weak sets: counts of graded sets. Set errors and ungraded sets are
  separate. Strong-set % includes errors in its known-result denominator.
- Receive %: strong or weak receives / graded receives plus aces suffered.
  Total receives also includes targeted in-play serves without a quality grade.
- Errors: explicitly recorded receive, set, and hit errors. Serve faults have
  their own counter. Error % uses graded receives, graded sets, and resolved hits.
- Break: the serving team wins the point, regardless of whether the outcome was
  Ace, Service Break, Defensive Hold, or an opposing player's Error. Broken:
  the receiving team loses it. The inferred server receives the individual break
  credit, and the inferred receiver receives the broken credit. A receiving win
  credits its receiver with a sideout. Player break/sideout opportunities use
  the same assignments. Break % divides breaks by known serves; sideout % uses
  known receptions.
- Double fault: the Double Fault classification records two consecutive service
  errors and a point to the receiving team. The optional touch log records
  individual attempts and their results.

Unknown results are not failures. Rate denominators include only known outcomes;
the All counts tab exposes attempts and unknowns. RPR requires stricter
whole-match coverage. Summary team points count all classified winners;
player-event coverage is reported separately.

Unchecked valid rallies count: omitting a point from a highlight reel must not
change match statistics. Rejected detections and Redo are excluded. Older
projects can retain quick tags and free-text player credits. A legacy Ace quick
tag with a roster player's name creates an ace serve event and awards the point
together; other free-text credits are preserved without being guessed into touch
statistics.

## Original Roundnet Player Rating

Implemented from Max Model's *Introducing Roundnet Player Rating*, not claimed
to match subsequent proprietary revisions in Roundnet Stats Tracker:

https://static1.squarespace.com/static/59e6b73dccc5c588c62abdb2/t/5ef399f33b82da542e9a8f6b/1593022964651/Introducing%2BRoundnet%2BPlayer%2BRating.pdf

The original equations were recovered from the primary document's indexed text
on 2026-10-05; its direct download returned 404. No formula is inferred from the
example scores in the supplied breakdown.

```
H = 20 * (1 - returned_hits / total_hits)
S = 5.5 * aces + 15 * serves_in / serve_attempts
D = (unreturned_defensive_touches + 0.4 * H * defensive_gets)
    * min(1, 44 / total_match_points)
E = 20 - 5 * (set_errors + hit_errors) - 2 * (tough_touches + aced)
RPR = H + S + D + E
```

Hitting includes missed hit attempts in total_hits; their additional penalty is
in E. E is labeled Efficiency in the UI (the paper calls it Cleanliness).
Receive errors remain visible raw statistics; they are not silently converted
into an ace or a missed set. Tough touches and aces suffered use explicit tags.
The approximate 20/30/30/20 category contributions are already in these equations;
they are not multiplied in a second time. Values are not clamped: RPR can exceed
100, and Efficiency can be negative. Defense scales down for matches over 44 points.

RPR recalculates automatically after each point when all recorded winners have
complete, valid RPR evidence and initial scores are 0–0. Empty future clips do
not block the running calculation. The card labels it "RPR so far" until every
outcome is known and you confirm the whole match in Match statistics. Each player's
rating additionally
requires at least one serve attempt and one hit attempt; undefined divisions
show a dash. Changes to scoring or touch evidence recalculate RPR and invalidate the
final-match label; starring a highlight does not. Quick logs can supply RPR evidence using the visible defaults. Ungraded
receive/set quality does not block quick-log RPR; those quality percentages and
physical-contact error rates remain unknown. Full logs retain their stricter
completeness requirements. The report separates RPR coverage from fully graded
touch coverage. On opening a saved project, quick logs with outdated inferred server/receiver
roles regenerate their derived events when the stored player selections still
form a valid point. Full touch observations and incompatible selections remain
unchanged and are flagged for review. This is a statistical
performance model, not a USAR skill level or a prediction of future ability.

## Reviewing, saving, and sharing

Match statistics has player cards, all counts/rates, the four RPR components,
team totals, an ace-opponent matrix, and a point audit log. Double-click a log row
to return to that rally. Clip outcomes shows the eight classifications alongside
older free-text tags. Export a PNG card, player CSV, or full JSON with the raw
and resolved events, coverage, and warnings. CSV rate values are fractions (0.75
means 75%); undefined values are blank. JSON includes all point detail. Exports
are local; there is no upload service.

Projects/autosave and undo preserve rosters, classifications, and touch logs.
Older projects open with their existing tags and empty touch logs when none were
saved. Splitting or merging an annotated point asks first and clears its
classification and touch logs; Undo restores them. Source replacement
clears old classifications and touch evidence. Boundary edits that strand a timestamp are flagged.
Video exports append the same current-score, four-player card as the shareable PNG
by default, with an explicit option to omit it.
It includes percentages and the four RPR components. Preview it in the export
dialog and set its duration from 1 to 30 seconds. If any outcome is unresolved,
the card labels its score as recorded and shows dependent player totals as
unknown; exported score overlays mark later clips provisional.

Reference app description (touch-order workflow, not code or exact-current-RPR
equivalence): https://apps.apple.com/us/app/roundnet-stats-tracker/id1458198404
