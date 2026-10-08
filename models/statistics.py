"""Derived statistics, never mutable counters. Recompute from ordered point logs.

Original RPR equations: Max Model, Introducing Roundnet Player Rating.
See docs/statistics.md for attribution, conventions, and incomplete-data policy.
"""

from collections import Counter
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
from uuid import uuid4

from .match_flow import match_timeline, normalize_classification
from .point_stats import normalize_point_stats, normalize_roster

RPR_SOURCE = "https://static1.squarespace.com/static/59e6b73dccc5c588c62abdb2/t/5ef399f33b82da542e9a8f6b/1593022964651/Introducing%2BRoundnet%2BPlayer%2BRating.pdf"


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def resolve_events(point, winner):
    """Infer return/put-away/get only from later net contacts or a complete point."""
    events = deepcopy(point.get("events", []))
    for index, event in enumerate(events):
        if event["kind"] not in ("hit", "defense") or event["result"] != "auto":
            continue
        team = event["player_id"][0]
        next_hit = next((e for e in events[index+1:] if e["kind"] == "hit"), None)
        if event["kind"] == "hit":
            if next_hit and next_hit["player_id"][0] != team and next_hit["result"] != "error":
                result = "returned"
            elif point.get("complete") and winner == team:
                result = "put_away"
            else:
                result = "unknown"
        else:
            if next_hit and next_hit["player_id"][0] == team and next_hit["result"] != "error":
                result = "get"
            elif point.get("complete") and winner and winner != team:
                result = "touch_not_returned"
            else:
                result = "unknown"
        event["result"] = result
    return events


def point_issues(point, winner="", *, start=None, end=None):
    """Drafts may be saved, but inconsistent logs never silently enter totals."""
    point = normalize_point_stats(point)
    if not point:
        return []
    issues = []
    events = point["events"]
    server, receiver = point["server_id"], point["receiver_id"]
    complete = point["complete"]
    if server and receiver and server[0] == receiver[0]:
        issues.append("Server and receiver must be on opposite teams.")
    if complete and (not server or not receiver or winner not in ("A", "B")):
        issues.append("A complete point needs server, receiver, and winning team.")
    serves = [e for e in events if e["kind"] == "serve"]
    if complete and (not events or not serves):
        issues.append("Record every serve attempt before marking this point complete.")
    live = False
    possession = None
    net_kind = None
    terminal_team = None
    last_time = -1
    receive_seen = False
    possession_touches = 0
    for i, event in enumerate(events):
        kind, result, team = event["kind"], event["result"], event["player_id"][0]
        timestamp = event.get("time")
        if timestamp is not None:
            if timestamp < last_time:
                issues.append("Touch timestamps must follow event order.")
            if (start is not None and timestamp < start - .001) or (end is not None and timestamp > end + .001):
                issues.append("A touch timestamp falls outside this rally; retime it or remove its timestamp.")
            last_time = timestamp
        if terminal_team is not None:
            issues.append("A terminal ace/error/no-touch must be the final event.")
            break
        if event.get("tough") and complete and winner == team:
            issues.append("A costly tough touch must contribute to losing the point, not just be a weak touch.")
        if kind == "serve":
            if live or any(e["kind"] != "serve" for e in events[:i]):
                issues.append("Serve attempts must precede all rally touches.")
            if server and event["player_id"] != server:
                issues.append("Serve attempts must belong to the selected server.")
            if result in ("in", "ace"):
                live = True
                possession = "B" if team == "A" else "A"
                net_kind = "serve"
            if result == "ace":
                terminal_team = team
            continue
        if not live:
            issues.append("A receive, set, hit, or defense needs an in-play serve first.")
        if possession and team != possession:
            issues.append(f"Event {i+1} is credited to the wrong team for this possession.")
        if kind == "receive":
            if net_kind != "serve" or receive_seen or any(e["kind"] != "serve" for e in events[:i]):
                issues.append("A receive belongs directly after the live serve.")
            if receiver and event["player_id"] != receiver:
                issues.append("The receive event must belong to the selected receiver.")
            receive_seen = True
        if kind == "defense" and (net_kind != "hit" or possession_touches):
            issues.append("Defense is the first contact after an opponent's hit; use Set for a later contact.")
        possession_touches += 1
        if result == "error" or (kind == "defense" and result == "no_touch"):
            terminal_team = "B" if team == "A" else "A"
        if kind == "hit" and result != "error":
            possession = "B" if team == "A" else "A"
            net_kind = "hit"
            possession_touches = 0
    if complete and serves and not live:
        if serves[-1]["result"] in ("fault", "rim"):
            terminal_team = "B" if serves[-1]["player_id"][0] == "A" else "A"
        else:
            issues.append("A let alone cannot finish a point.")
    if terminal_team and winner and terminal_team != winner:
        issues.append("Winning team conflicts with the final ace/error/fault.")
    resolved = resolve_events(point, winner)
    for i, event in enumerate(resolved):
        if event["kind"] not in ("hit", "defense") or event["result"] == "error":
            continue
        team = event["player_id"][0]
        next_hit = next((e for e in resolved[i+1:] if e["kind"] == "hit"), None)
        returned = next_hit is not None and next_hit["result"] != "error"
        if event["kind"] == "hit":
            if event["result"] == "put_away" and (returned or winner and winner != team):
                issues.append("A put-away cannot be returned to the net or belong to the losing team.")
            if complete and event["result"] == "returned" and not returned:
                issues.append("Record the opposing net hit for a returned hit.")
        else:
            if event["result"] in ("touch_not_returned", "no_touch") and returned:
                issues.append("This defense was followed by a successful net return; use Get/Auto.")
            if complete and event["result"] == "get" and not returned:
                issues.append("Record the successful net return after the defensive get.")
        if complete and event["result"] == "unknown":
            issues.append("An automatic hit/defense result is unresolved; record the remaining touches.")
    if complete and live and terminal_team is None:
        finished_by_hit = any(e["kind"] == "hit" and e["result"] == "put_away" for e in resolved)
        if not finished_by_hit:
            issues.append("The point needs a terminal put-away, ace, error, or missed defense.")
    return list(dict.fromkeys(issues))


def classified_point_issues(point, classification, assignment):
    """Check detailed evidence against a clip outcome and inferred roles."""
    point = normalize_point_stats(point)
    classification = normalize_classification(classification)
    if not point or not classification:
        return []
    kind = classification["kind"]
    issues = []
    if assignment.get("provisional"):
        issues.append("Serving order is provisional until earlier clips are classified.")
    if point["server_id"] and point["server_id"] != assignment["server_id"]:
        issues.append("Touch log server conflicts with the clip's inferred server.")
    if point["receiver_id"] and point["receiver_id"] != assignment["receiver_id"]:
        issues.append("Touch log receiver conflicts with the clip's inferred receiver.")
    events = point["events"]
    if events:
        if kind == "ace" and any(e["kind"] != "serve" for e in events):
            issues.append("An Ace cannot include rally touches after the serve.")
        if kind == "ace" and any(e["kind"] == "serve" and e["result"] == "ace"
                                 and e["player_id"] != assignment["server_id"] for e in events):
            issues.append("Touch log Ace belongs to a different server.")
        if kind == "ace" and any(e["kind"] == "serve" and e["result"] == "in" for e in events):
            issues.append("An Ace cannot also contain a returned in-play serve.")
        if kind == "double_fault" and any(e["kind"] != "serve" or e["result"] in ("in", "ace") for e in events):
            issues.append("A Double Fault cannot include a live serve or rally touches.")
        if kind == "error" and any(e["result"] == "error" and e["player_id"] != classification["player_id"] for e in events):
            issues.append("Touch log Error belongs to a different player.")
        if kind == "error" and any(e["kind"] == "serve" and e["result"] == "ace" for e in events):
            issues.append("An Ace serve cannot end in a named Error.")
    if point["complete"] and events:
        if kind == "ace" and not any(e["kind"] == "serve" and e["result"] == "ace"
                                     and e["player_id"] == assignment["server_id"] for e in events):
            issues.append("Touch log does not support the Ace classification.")
        fault_attempts = [e for e in events if e["kind"] == "serve" and e["result"] != "let"]
        if kind == "double_fault" and (len(fault_attempts) != 2 or any(e["result"] not in ("fault", "rim") for e in fault_attempts)):
            issues.append("Touch log does not support the Double Fault classification.")
        if kind == "error" and not any(e["player_id"] == classification["player_id"]
                                       and e["result"] == "error" for e in events):
            issues.append("Touch log does not support the named Error classification.")
        if kind in ("defensive_break", "defensive_hold") and not any(e["kind"] == "defense" for e in events):
            issues.append("A complete defensive outcome needs its defensive touch.")
    return list(dict.fromkeys(issues))


COUNTERS = (
    "points", "points_won", "serve_attempts", "serves_in", "aces", "aced", "faults", "rims", "lets",
    "double_faults", "receives", "strong_receives", "weak_receives", "receive_errors", "sets",
    "strong_sets", "weak_sets", "set_errors", "hit_attempts", "put_aways", "hits_returned", "hit_errors",
    "defensive_touches", "defensive_gets", "defensive_not_returned", "no_touches", "tough_touches",
    "errors", "breaks", "broken", "break_opportunities", "sideouts", "sideout_opportunities", "unknown_results",
)


def original_rpr(stats, total_points, *, eligible):
    """Published original model, without clamping or reapplying its weights."""
    missing = stats["hit_attempts"] == 0 or stats["serve_attempts"] == 0
    if not eligible or missing:
        return {"hitting": None, "serving": None, "defense": None, "efficiency": None, "overall": None}
    hitting = 20 * (1 - stats["hits_returned"] / stats["hit_attempts"])
    serving = 5.5 * stats["aces"] + 15 * stats["serves_in"] / stats["serve_attempts"]
    defense = (stats["defensive_not_returned"] + .4 * hitting * stats["defensive_gets"]) * min(1, 44 / max(1, total_points))
    efficiency = 20 - 5 * (stats["set_errors"] + stats["hit_errors"]) - 2 * (stats["tough_touches"] + stats["aced"])
    return {"hitting": hitting, "serving": serving, "defense": defense, "efficiency": efficiency,
            "overall": hitting + serving + defense + efficiency}


def calculate_statistics(rallies, settings=None):
    settings = settings or {}
    rallies = list(rallies)
    timeline = {row["index"]: row for row in match_timeline(rallies, settings)}
    roster = normalize_roster(settings.get("players"))
    players = {p["player_id"]: {**p, **dict.fromkeys(COUNTERS, 0)} for p in roster}
    teams = {team: {"name": settings.get(f"team_{team.lower()}", f"Team {team}"),
                    "points_won": 0, "breaks": 0, "broken": 0, "break_opportunities": 0,
                    "sideouts": 0, "sideout_opportunities": 0} for team in ("A", "B")}
    coverage = {"points": 0, "complete": 0, "partial": 0, "untagged": 0, "invalid": 0,
                "replays": 0, "rejected": 0, "classified": 0, "unresolved": 0,
                "rpr_complete": 0, "touch_quality_complete": 0}
    point_rows, warnings, ace_pairs = [], [], Counter()
    outcome_tags, player_tags, classification_counts = Counter(), Counter(), Counter()
    for index, rally in sorted(enumerate(rallies), key=lambda item: (_field(item[1], "start_time", _field(item[1], "start", 0)), item[0])):
        assignment = timeline[index]
        classification = normalize_classification(_field(rally, "classification", {}))
        kind = classification.get("kind", "")
        if _field(rally, "rejected", False):
            coverage["rejected"] += 1
            continue
        if kind == "redo" or (not kind and _field(rally, "outcome", "") == "Replay / no point"):
            coverage["replays"] += 1
            if kind:
                classification_counts[kind] += 1
            continue
        coverage["points"] += 1
        if kind:
            coverage["classified"] += 1
            classification_counts[kind] += 1
        if not kind and _field(rally, "outcome", ""):
            outcome_tags[_field(rally, "outcome")] += 1
        if not kind and _field(rally, "player", ""):
            player_tags[_field(rally, "player")] += 1
        winner = assignment["winner"]
        coverage["unresolved"] += int(not winner)
        if winner in teams:
            teams[winner]["points_won"] += 1
        point = normalize_point_stats(_field(rally, "point_stats", {}))
        issues = point_issues(point, winner, start=_field(rally, "start_time"), end=_field(rally, "end_time"))
        if kind and point:
            issues.extend(classified_point_issues(point, classification, assignment))
        issues = list(dict.fromkeys(issues))
        row = {"rally_id": _field(rally, "rally_id", ""), "start": _field(rally, "start_time", 0),
               "end": _field(rally, "end_time", 0), "winner": winner, "enabled": _field(rally, "enabled", True),
               "outcome": _field(rally, "outcome", ""), "player_credit": _field(rally, "player", ""),
               "note": _field(rally, "note", ""), "classification": kind,
               "server_id": assignment["server_id"] if kind else point.get("server_id", ""),
               "receiver_id": assignment["receiver_id"] if kind else point.get("receiver_id", ""),
               "point_stats": point, "issues": issues, "events": []}
        point_rows.append(row)
        if kind:
            for player in players.values():
                player["points"] += 1
                player["points_won"] += int(winner == player["team"])
            if not assignment["provisional"]:
                server, receiver = assignment["server_id"], assignment["receiver_id"]
                serving, receiving = server[0], receiver[0]
                broke = winner == serving
                teams[serving]["break_opportunities"] += 1
                teams[receiving]["sideout_opportunities"] += 1
                teams[serving]["breaks"] += int(broke)
                teams[receiving]["broken"] += int(broke)
                teams[receiving]["sideouts"] += int(not broke)
                players[server]["break_opportunities"] += 1
                players[receiver]["sideout_opportunities"] += 1
                players[server]["breaks"] += int(broke)
                players[receiver]["broken"] += int(broke)
                players[receiver]["sideouts"] += int(not broke)
                if kind == "ace":
                    players[server]["aces"] += 1
                    players[receiver]["aced"] += 1
                    ace_pairs[server, receiver] += 1
                elif kind == "double_fault":
                    players[server]["double_faults"] += 1
            if kind == "error":
                players[classification["player_id"]]["errors"] += 1
        if issues:
            coverage["invalid"] += 1
            warnings.append(f"Point at {row['start']:.2f}s: {' '.join(issues)}")
            if kind == "double_fault" and not assignment["provisional"]:
                players[assignment["server_id"]]["faults"] += 2
            continue
        if not point or not point["events"]:
            coverage["untagged"] += 1
            if kind == "double_fault" and not assignment["provisional"]:
                players[assignment["server_id"]]["faults"] += 2
            continue
        coverage["complete" if point["complete"] else "partial"] += 1
        events = resolve_events(point, winner)
        row["events"] = events
        ungraded = any(e["result"] == "unknown" for e in events)
        rpr_unknown = any(e["result"] == "unknown" and
                          (not point.get("quick_log") or e["kind"] not in ("receive", "set")) for e in events)
        coverage["rpr_complete"] += int(point["complete"] and not rpr_unknown)
        coverage["touch_quality_complete"] += int(point["complete"] and not ungraded and not point.get("quick_log"))
        if not kind:
            for player in players.values():
                player["points"] += 1
                player["points_won"] += int(winner == player["team"])
        server, receiver = point["server_id"], point["receiver_id"]
        if not kind and server and winner:
            serving, receiving = server[0], "B" if server[0] == "A" else "A"
            broke = winner == serving
            teams[serving]["break_opportunities"] += 1
            teams[receiving]["sideout_opportunities"] += 1
            teams[serving]["breaks"] += int(broke)
            teams[receiving]["broken"] += int(broke)
            teams[receiving]["sideouts"] += int(not broke)
            players[server]["break_opportunities"] += 1
            players[server]["breaks"] += int(broke)
            if receiver:
                players[receiver]["sideout_opportunities"] += 1
                players[receiver]["broken"] += int(broke)
                players[receiver]["sideouts"] += int(not broke)
        serves = [e for e in events if e["kind"] == "serve"]
        attempts = [e for e in serves if e["result"] != "let"]
        if not kind and point["complete"] and len(attempts) >= 2 and attempts[-1]["result"] in ("fault", "rim"):
            players[attempts[-1]["player_id"]]["double_faults"] += 1
        for event in events:
            p = players[event["player_id"]]
            event_kind, result = event["kind"], event["result"]
            p["tough_touches"] += int(event.get("tough", False))
            if event_kind == "serve":
                p["serve_attempts"] += int(result != "let")
                p["serves_in"] += int(result in ("in", "ace"))
                p["aces"] += int(not kind and result == "ace")
                p["faults"] += int(result in ("fault", "rim"))
                p["rims"] += int(result == "rim")
                p["lets"] += int(result == "let")
                if receiver and result in ("in", "ace"):
                    players[receiver]["receives"] += 1
                    players[receiver]["aced"] += int(not kind and result == "ace")
                    if result == "ace" and not kind:
                        ace_pairs[event["player_id"], receiver] += 1
            elif event_kind == "receive":
                if not receiver:
                    p["receives"] += 1
                key = {"strong": "strong_receives", "weak": "weak_receives", "error": "receive_errors"}.get(result)
                if key:
                    p[key] += 1
                p["errors"] += int(result == "error" and not (kind == "error" and event["player_id"] == classification["player_id"]))
            elif event_kind == "set":
                p["sets"] += 1
                key = {"strong": "strong_sets", "weak": "weak_sets", "error": "set_errors"}.get(result)
                if key:
                    p[key] += 1
                p["errors"] += int(result == "error" and not (kind == "error" and event["player_id"] == classification["player_id"]))
            elif event_kind == "hit":
                p["hit_attempts"] += 1
                key = {"put_away": "put_aways", "returned": "hits_returned", "error": "hit_errors"}.get(result)
                if key:
                    p[key] += 1
                p["errors"] += int(result == "error" and not (kind == "error" and event["player_id"] == classification["player_id"]))
            elif event_kind == "defense":
                p["defensive_touches"] += int(result != "no_touch")
                key = {"get": "defensive_gets", "touch_not_returned": "defensive_not_returned", "no_touch": "no_touches"}.get(result)
                if key:
                    p[key] += 1
            p["unknown_results"] += int(result == "unknown")
        if kind == "double_fault":
            players[assignment["server_id"]]["faults"] += max(0, 2 - sum(e["kind"] == "serve" and e["result"] in ("fault", "rim") for e in events))
    # Calculate automatically from the recorded points. Empty future clips
    # do not block a running rating, but missing/invalid scored-point evidence
    # must never silently bias it. Confirmation only identifies a final rating.
    recorded_points = coverage["points"] - coverage["unresolved"]
    rated = bool(recorded_points > 0 and coverage["rpr_complete"] == recorded_points
                 and not coverage["invalid"] and not coverage["partial"]
                 and not any(row["provisional"] and row["winner"] for row in timeline.values())
                 and not settings.get("initial_score_a", 0) and not settings.get("initial_score_b", 0))
    final_rating = bool(rated and settings.get("stats_complete", False) and not coverage["unresolved"])
    incomplete_classified_detail = any(_field(row, "classification", "") for row in point_rows) and (coverage["complete"] != recorded_points or coverage["invalid"] or coverage["partial"])
    for p in players.values():
        hits_known = p["put_aways"] + p["hits_returned"] + p["hit_errors"]
        receives_known = p["strong_receives"] + p["weak_receives"] + p["receive_errors"] + p["aced"]
        p["serve_pct"] = ratio(p["serves_in"], p["serve_attempts"])
        p["ace_pct"] = ratio(p["aces"], p["serve_attempts"])
        p["put_away_pct"] = ratio(p["put_aways"], hits_known)
        p["receive_pct"] = ratio(p["strong_receives"] + p["weak_receives"], receives_known)
        p["strong_set_pct"] = ratio(p["strong_sets"], p["strong_sets"] + p["weak_sets"] + p["set_errors"])
        p["defensive_conversion_pct"] = ratio(p["defensive_gets"], p["defensive_gets"] + p["defensive_not_returned"])
        p["error_pct"] = ratio(p["errors"], p["strong_receives"] + p["weak_receives"] + p["receive_errors"]
                                + p["strong_sets"] + p["weak_sets"] + p["set_errors"] + hits_known)
        p["break_pct"] = ratio(p["breaks"], p["break_opportunities"])
        if incomplete_classified_detail:
            for metric in ("serve_pct", "ace_pct", "put_away_pct", "receive_pct", "strong_set_pct",
                           "defensive_conversion_pct", "error_pct"):
                p[metric] = None
        if any(row["point_stats"].get("quick_log") for row in point_rows):
            # Quick evidence determines RPR roles, not touch-quality grades or
            # the denominator of all physical contacts.
            for metric in ("receive_pct", "strong_set_pct", "error_pct"):
                p[metric] = None
        p["rpr"] = original_rpr(p, recorded_points, eligible=rated)
    for identity, team in teams.items():
        team["initial_score"] = settings.get(f"initial_score_{identity.lower()}", 0)
        team["score"] = team["initial_score"] + team["points_won"]
        team["break_pct"] = ratio(team["breaks"], team["break_opportunities"])
        team["sideout_pct"] = ratio(team["sideouts"], team["sideout_opportunities"])
    return {"schema_version": 1, "format": "roundnet_match_statistics", "players": list(players.values()),
            "teams": teams, "coverage": coverage, "points": point_rows, "warnings": warnings,
            "provisional": any(row["provisional"] for row in timeline.values()),
            "outcome_tags": dict(outcome_tags), "player_tags": dict(player_tags),
            "classification_counts": dict(classification_counts),
            "rpr_eligible": rated, "rpr_final": final_rating, "rpr_model": "original_max_model", "rpr_source": RPR_SOURCE,
            "aces_by_opponent": [{"server_id": a, "receiver_id": b, "aces": n} for (a, b), n in sorted(ace_pairs.items())]}


def export_statistics(path, report, *, protected_paths=()):
    destination = Path(path).expanduser().resolve()
    if destination in {Path(p).expanduser().resolve() for p in protected_paths if p}:
        raise ValueError("Save statistics separately from the source video and project")
    if destination.suffix.lower() == ".json":
        content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    elif destination.suffix.lower() == ".csv":
        buffer = io.StringIO(newline="")
        fields = ["player_id", "name", "team", *COUNTERS, "serve_pct", "ace_pct", "put_away_pct", "receive_pct",
                  "strong_set_pct", "defensive_conversion_pct", "error_pct", "break_pct", "rpr"]
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for p in report["players"]:
            row = {k: p.get(k) for k in fields}
            row["rpr"] = p["rpr"]["overall"]
            if row["name"].lstrip().startswith(("=", "+", "-", "@")):
                row["name"] = "'" + row["name"]
            writer.writerow(row)
        content = buffer.getvalue()
    else:
        raise ValueError("Choose a .json or .csv statistics file")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
