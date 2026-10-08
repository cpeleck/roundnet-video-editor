"""Guided possession logs: explicit player evidence with visible quick defaults."""

from copy import deepcopy

from .point_stats import PLAYER_IDS, new_event


def normalize_quick_log(value):
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("Unsupported quick-log version")
    possessions = value.get("possessions", [])
    if not isinstance(possessions, list) or len(possessions) > 200:
        raise ValueError("Invalid possession list")
    normalized = []
    for p in possessions:
        if not isinstance(p, dict) or p.get("first_touch_id") not in PLAYER_IDS or p.get("hitter_id") not in PLAYER_IDS:
            raise ValueError("Choose a first-touch player and hitter for each possession")
        row = {k: p[k] for k in ("first_touch_id", "hitter_id")}
        for flag in ("tough_receive", "tough_set"):
            if type(p.get(flag, False)) is not bool:
                raise ValueError("Costly-touch corrections must be true or false")
            row[flag] = p.get(flag, False)
        normalized.append(row)
    faults = value.get("serve_faults", [])
    if not isinstance(faults, list) or len(faults) > 50 or any(f not in ("fault", "rim", "let") for f in faults):
        raise ValueError("Invalid serve corrections")
    if sum(f != "let" for f in faults) > 1:
        raise ValueError("Only one failed attempt can precede the live serve")
    defender = value.get("terminal_defender_id")
    if defender is not None and defender not in PLAYER_IDS:
        raise ValueError("Choose the final defender or No touch")
    error = value.get("error")
    if error is not None:
        if not isinstance(error, dict) or error.get("player_id") not in PLAYER_IDS or error.get("kind") not in ("receive", "set", "hit"):
            raise ValueError("Choose the error player and receive, set, or hit")
        error = {k: error[k] for k in ("player_id", "kind")}
    return {"version": 1, "possessions": normalized, "serve_faults": deepcopy(faults),
            "terminal_defender_id": defender, "error": error,
            "assumptions": {"hitter_suggested_from_first_touch": True,
                            "ordinary_setter_is_partner": True,
                            "unflagged_touches_have_no_costly_penalty": True,
                            "unrecorded_serve_corrections": "none"}}


def quick_point_stats(value, classification, assignment):
    """Translate a quick log without inventing strong/weak touch grades."""
    log = normalize_quick_log(value)
    kind = classification["kind"]
    server, receiver = assignment["server_id"], assignment["receiver_id"]
    events = [new_event(server, "serve", f) for f in log["serve_faults"]]
    possessions = log["possessions"]
    if kind == "redo":
        if possessions or log["error"] or log["serve_faults"] or log["terminal_defender_id"]:
            raise ValueError("Redo has no point statistics")
        return {}
    if kind in ("ace", "double_fault", "service_break", "sideout"):
        if possessions or log["error"] or log["terminal_defender_id"]:
            raise ValueError("Use detailed touches to correct a no-rally outcome")
        if kind == "double_fault":
            if log["serve_faults"]:
                raise ValueError("Double Fault already records two faults")
            events = [new_event(server, "serve", "fault") for _ in range(2)]
        else:
            events.append(new_event(server, "serve", "ace" if kind == "ace" else "in"))
            if kind == "service_break":
                events.append(new_event(receiver, "receive", "error"))
            elif kind == "sideout":
                events.extend([new_event(receiver, "receive", "unknown"),
                               new_event(receiver, "hit", "put_away")])
    else:
        if not possessions:
            raise ValueError("Record the first touch in each possession")
        if kind.startswith("defensive") and len(possessions) < 2:
            raise ValueError("A defensive point needs a defensive possession")
        error = log["error"]
        if kind == "error" and (not error or error["player_id"] != classification["player_id"]):
            raise ValueError("Error details must identify the classified error player")
        if kind != "error" and error:
            raise ValueError("A defensive win ends with a put-away, not an Error")
        if error and log["terminal_defender_id"]:
            raise ValueError("An Error cannot also have a final defender")
        events.append(new_event(server, "serve", "in"))
        for i, p in enumerate(possessions):
            team = receiver[0] if i % 2 == 0 else server[0]
            first, hitter = p["first_touch_id"], p["hitter_id"]
            if first[0] != team or hitter[0] != team or (i == 0 and first != receiver):
                raise ValueError("Possessions must start with the inferred receiver and alternate teams")
            last = i == len(possessions) - 1
            terminal_error = error if last else None
            if terminal_error and terminal_error["player_id"][0] != team:
                raise ValueError("The error must belong to the final possession's team")
            if terminal_error and terminal_error["kind"] == "receive":
                if i != 0 or terminal_error["player_id"] != first:
                    raise ValueError("A receive error belongs to the initial receiver")
                events.append(new_event(first, "receive", "error"))
                break
            if i and p["tough_receive"]:
                raise ValueError("A costly receive is only available on the initial serve receive")
            events.append(new_event(first, "receive" if i == 0 else "defense",
                                    ("weak" if p["tough_receive"] else "unknown") if i == 0 else "auto",
                                    tough=p["tough_receive"]))
            partner = first[0] + ("2" if first[1] == "1" else "1")
            if terminal_error and terminal_error["kind"] == "set":
                events.append(new_event(terminal_error["player_id"], "set", "error"))
                break
            if first == hitter or p["tough_set"]:
                events.append(new_event(partner, "set", "weak" if p["tough_set"] else "unknown", tough=p["tough_set"]))
            if terminal_error:
                if hitter != terminal_error["player_id"]:
                    raise ValueError("The final hitter must be the hit-error player")
                events.append(new_event(hitter, "hit", "error"))
            else:
                events.append(new_event(hitter, "hit", "put_away" if last else "auto"))
        if log["terminal_defender_id"]:
            defender = log["terminal_defender_id"]
            if defender[0] == possessions[-1]["hitter_id"][0]:
                raise ValueError("The final defender must be on the opposing team")
            events.append(new_event(defender, "defense", "touch_not_returned"))
    return {"version": 1, "complete": True, "server_id": server, "receiver_id": receiver,
            "events": events, "quick_log": log}


def reconcile_quick_stats(point, classification, assignment, *, start=None, end=None):
    """Refresh generated roles while preserving explicit possession selections.

    Full touch observations and incompatible quick selections remain unchanged
    so the statistics report can identify the correction that needs review.
    """
    if not point.get('quick_log') or not point.get('complete') or assignment['provisional'] or (
        point.get('server_id') == assignment['server_id']
        and point.get('receiver_id') == assignment['receiver_id']
    ):
        return point
    from .statistics import point_issues, classified_point_issues
    try:
        refreshed = quick_point_stats(point['quick_log'], classification, assignment)
        issues = point_issues(refreshed, assignment['winner'], start=start, end=end)
        issues += classified_point_issues(refreshed, classification, assignment)
        return point if issues else refreshed
    except ValueError:
        return point
