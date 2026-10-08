"""Versioned, non-executable per-point annotations and stable four-player slots."""

from copy import deepcopy
import math
from uuid import uuid4


PLAYER_IDS = ("A1", "A2", "B1", "B2")
RESULTS = {
    "serve": ("in", "ace", "fault", "rim", "let"),
    "receive": ("strong", "weak", "error", "unknown"),
    "set": ("strong", "weak", "error", "unknown"),
    "hit": ("auto", "put_away", "returned", "error"),
    "defense": ("auto", "get", "touch_not_returned", "no_touch"),
}


def normalize_roster(players=None):
    if players is None:
        return [{"player_id": pid, "team": pid[0], "name": f"Player {pid}"} for pid in PLAYER_IDS]
    if not isinstance(players, list) or len(players) != 4:
        raise ValueError("A match needs four players: A1, A2, B1, B2")
    result, names, identities = [], set(), set()
    for p in players:
        if not isinstance(p, dict) or p.get("player_id") not in PLAYER_IDS:
            raise ValueError("Unknown player slot")
        pid, name = p["player_id"], p.get("name", "")
        if pid in identities or p.get("team") != pid[0]:
            raise ValueError("Player slots must be unique and stay on their assigned team")
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("Each player needs a name of 1–80 characters")
        name = " ".join(name.split())
        if name.casefold() in names:
            raise ValueError("Use distinct player names (add an initial if needed)")
        names.add(name.casefold())
        identities.add(pid)
        result.append({"player_id": pid, "team": pid[0], "name": name})
    return sorted(result, key=lambda p: PLAYER_IDS.index(p["player_id"]))


def new_event(player_id, kind, result, *, time=None, tough=False):
    event = {"event_id": uuid4().hex, "player_id": player_id, "kind": kind,
             "result": result, "time": time, "tough": tough}
    normalize_point_stats({"version": 1, "events": [event]})
    return event


def normalize_point_stats(value):
    """Validate storage shape, including drafts; semantic checks are separate."""
    if value == {}:
        return {}
    if not isinstance(value, dict) or type(value.get("version")) is not int or value.get("version") != 1:
        raise ValueError("Unsupported point-statistics version")
    complete = value.get("complete", False)
    if not isinstance(complete, bool):
        raise ValueError("Point statistics completeness must be true or false")
    normalized = {"version": 1, "complete": complete,
                  "server_id": value.get("server_id", ""), "receiver_id": value.get("receiver_id", ""),
                  "events": deepcopy(value.get("events", []))}
    if "quick_log" in value:
        from .quick_stats import normalize_quick_log
        normalized["quick_log"] = normalize_quick_log(value["quick_log"])
    for field in ("server_id", "receiver_id"):
        if normalized[field] not in ("", *PLAYER_IDS):
            raise ValueError(f"Unknown {field}")
    if not isinstance(normalized["events"], list) or len(normalized["events"]) > 1000:
        raise ValueError("A point must have at most 1000 touch events")
    identities = set()
    for e in normalized["events"]:
        if not isinstance(e, dict):
            raise ValueError("Invalid touch event")
        identity = e.get("event_id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Touch events need unique nonempty identities")
        identities.add(identity)
        if e.get("player_id") not in PLAYER_IDS or not isinstance(e.get("kind"), str) or e["kind"] not in RESULTS:
            raise ValueError("Unknown player or touch type")
        if e.get("result") not in RESULTS[e["kind"]]:
            raise ValueError(f"Invalid result for {e['kind']}")
        if not isinstance(e.get("tough", False), bool):
            raise ValueError("Tough-touch flag must be true or false")
        if e.get("tough") and (e["kind"] not in ("receive", "set") or e["result"] != "weak"):
            raise ValueError("Only weak receives/sets can be marked as a costly tough touch")
        timestamp = e.get("time")
        if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                                      or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("Touch timestamps must be finite non-negative source seconds")
    return normalized
