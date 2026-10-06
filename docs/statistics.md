# Point-by-point player statistics

## Workflow

Set four names and the starting server/receiver in Point / Highlight → Teams /
players / initial score. Slots A1,
A2, B1, B2 are stable identities: renaming a player updates their displayed name
without moving any credits. This is one four-player match, not a season database.

Select a rally and choose Log point. The dialog has its own video preview. Tap
the four player buttons in touch order; the usual serve, receive, set, hit, and
defense roles are suggested. Ace, Fault, Error, and Point Won finish the point
and select its winner. Undo touch reverses a quick action. The detailed controls
below the buttons handle other touch types, qualities, and unusual plays. Record
every serve attempt and touch. Multiple possessions and both partners'
contributions are supported.
Optional source timestamps allow jumping back to a touch. A let is not a serve
attempt. Record faults individually, including a rim when that is the reason.
The winner is explicitly selected; this is not automatic refereeing or rotation.

After a receive or defense, the editor suggests a set by the partner, then a hit.
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
- Break: the serving team wins the point. Broken: the receiving team loses it.
  Both teammates share these team figures; do not sum their rows as individual
  credits. Break % divides breaks by known serving points; sideout % uses known
  receiving points. These are not automatically attributed to a specific player.
- Double fault: a complete point ending in a serve fault after at least two
  non-let attempts. Single-fault rules can still be logged without inventing a
  second attempt. The app does not impose a particular rulebook.

Unknown results are not failures. Rate denominators include only known outcomes;
the All counts tab exposes attempts and unknowns. No denominator displays a dash,
not 0%. RPR requires stricter whole-match coverage. Summary team points count all
tagged winners; player-event coverage is reported separately.

Unchecked valid rallies count: omitting a point from a highlight reel must not
change match statistics. Rejected detections and Replay / no point are excluded.
An Ace quick tag with a roster player's name creates an ace serve event and
awards the point together. Other legacy free-text credits are preserved but are
not guessed into touch statistics; log those old rallies for detailed statistics.

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

RPR is shown only after all non-rejected, non-replay points are complete, none has
an unknown result or invalid sequence, initial scores are 0–0, and you explicitly
confirm the whole match in Match statistics. Each player's rating additionally
requires at least one serve attempt and one hit attempt; undefined divisions
show a dash. Changes to scoring or touch evidence invalidate the match-level
confirmation; starring a highlight does not. This is a statistical
performance model, not a USAR skill level or a prediction of future ability.

## Reviewing, saving, and sharing

Match statistics has player cards, all counts/rates, the four RPR components,
team totals, an ace-opponent matrix, and a point audit log. Double-click a log row
to return to that rally. Existing free-text credits and outcomes remain in Quick
tags, separate from the touch-derived counts. Export a PNG card, player CSV, or full JSON with the raw
and resolved events, coverage, and warnings. CSV rate values are fractions (0.75
means 75%); undefined values are blank. JSON includes all point detail. Exports
are local; there is no upload service.

Projects/autosave and undo preserve rosters and touch logs. Older projects open
with empty touch logs and default player slots. Splitting or merging an annotated
point asks first and clears its touch logs; Undo restores them. Source replacement
clears old touch evidence. Boundary edits that strand a timestamp are flagged.
Video exports append the same final-score, four-player card as the shareable PNG.
It includes percentages and the four RPR components. Preview it in the export
dialog and set its duration from 1 to 30 seconds.

Reference app description (touch-order workflow, not code or exact-current-RPR
equivalence): https://apps.apple.com/us/app/roundnet-stats-tracker/id1458198404
