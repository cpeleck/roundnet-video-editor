"""Clip classification and deterministic four-player serving order.

The timeline is derived from saved clips each time it is requested.  A redo or
rejected detection does not consume a turn, so editing an earlier clip also
updates the suggested server and receiver for every later clip.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .point_stats import PLAYER_IDS


CLASSIFICATIONS = {
    "ace": {
        "label": "Ace",
        "description": "Unreturnable serve with no reasonable chance to set; serving team scores.",
    },
    "double_fault": {
        "label": "Double Fault",
        "description": "Two consecutive service errors; receiving team scores.",
    },
    "service_break": {
        "label": "Service Break",
        "description": "Serving team wins through service pressure.",
    },
    "sideout": {
        "label": "Sideout",
        "description": "Receiving team wins with a clean return and no rally.",
    },
    "defensive_break": {
        "label": "Defensive Break",
        "description": "Receiving team stops the serving team and converts the defensive touch into a point.",
    },
    "defensive_hold": {
        "label": "Defensive Hold",
        "description": "Serving team stops the receiving team and converts the defensive touch into a point.",
    },
    "error": {
        "label": "Error",
        "description": "A specific player's unforced error; the opposing team scores.",
    },
    "redo": {
        "label": "Redo",
        "description": "Keep the clip without a point or statistics; server and receiver stay the same.",
    },
}

_SERVING_KINDS = frozenset(("ace", "service_break", "defensive_hold"))
_RECEIVING_KINDS = frozenset(("double_fault", "sideout", "defensive_break"))


def normalize_classification(value: Any) -> dict[str, Any]:
    """Validate and copy one versioned clip classification, or an empty draft."""

    if value == {}:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Classification must be empty or a versioned mapping")
    if type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("Unsupported classification version")
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in CLASSIFICATIONS:
        raise ValueError("Unknown clip classification")
    expected_keys = {"version", "kind", "player_id"} if kind == "error" else {"version", "kind"}
    if set(value) != expected_keys:
        if kind == "error":
            raise ValueError("An error classification needs exactly one player_id")
        raise ValueError("Only an error classification can name a player")
    if kind == "error":
        player_id = value["player_id"]
        if player_id not in PLAYER_IDS:
            raise ValueError("Error player_id must be a stable roster slot")
        return {"version": 1, "kind": kind, "player_id": player_id}
    return {"version": 1, "kind": kind}


def classification_winner(classification: Mapping[str, Any], server_id: str) -> str:
    """Return A, B, or an empty string for one classification."""

    normalized = normalize_classification(classification)
    kind = normalized.get("kind")
    if not kind or kind == "redo":
        return ""
    if server_id not in PLAYER_IDS:
        raise ValueError("Classification winner needs a valid server")
    if kind in _SERVING_KINDS:
        return server_id[0]
    if kind in _RECEIVING_KINDS:
        return _other_team(server_id[0])
    return _other_team(normalized["player_id"][0])


def match_timeline(rallies: list[Any], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return chronological serving assignments and scores for saved clips.

    ``index`` points to the clip in the caller's original list. ``scored_index``
    is the number of awarded points before that clip, excluding the optional
    initial score. An unknown clip leaves subsequent assignments provisional;
    the known points continue advancing so later edits can still be previewed.
    """

    if not isinstance(settings, Mapping):
        raise ValueError("Match settings must be a mapping")
    first_server = settings.get("starting_server", "A1")
    first_receiver = settings.get("starting_receiver", "B1")
    if (first_server not in PLAYER_IDS or first_receiver not in PLAYER_IDS
            or first_server[0] == first_receiver[0]):
        raise ValueError("Starting server and receiver must be opposing roster slots")
    target_score = settings.get("target_score", 21)
    if type(target_score) is not int or target_score < 2:
        raise ValueError("Target score must be an integer of at least two")
    score_a = settings.get("initial_score_a", 0)
    score_b = settings.get("initial_score_b", 0)
    if any(type(score) is not int or score < 0 for score in (score_a, score_b)):
        raise ValueError("Initial scores must be non-negative integers")

    server_partner = _partner(first_server)
    receiver_partner = _partner(first_receiver)
    # Opening server gets one point, then each player gets two. The next
    # turn belongs to the receiver's partner against the server's partner:
    # A1 -> B1, B2 -> A2/A1, A2 -> B1/B2, B1 -> A1/A2.
    server_cycle = (first_server, receiver_partner, server_partner, first_receiver)
    first_receivers = (first_receiver, server_partner, first_receiver, first_server)
    second_receivers = (receiver_partner, first_server, receiver_partner, server_partner)
    turn = 0
    serve_in_turn = 0
    scored_index = 0
    provisional = False
    overtime = _is_overtime(score_a, score_b, target_score)
    ordered = sorted(enumerate(rallies), key=lambda item: (_start(item[1]), item[0]))
    result: list[dict[str, Any]] = []

    for index, rally in ordered:
        server_id = server_cycle[turn % 4]
        receiver_id = (second_receivers if serve_in_turn else first_receivers)[turn % 4]
        rejected = bool(_field(rally, "rejected", False))
        classification = normalize_classification(_field(rally, "classification", {}))
        kind = classification.get("kind", "")
        legacy_winner = _field(rally, "winner", "")
        legacy_redo = not kind and _field(rally, "outcome", "") == "Replay / no point"
        if rejected or kind == "redo" or legacy_redo:
            winner = ""
        elif kind and provisional and kind != "error":
            # A serve-based category identifies a winner only after the
            # earlier scoring sequence fixes which team serves this clip.
            winner = ""
        elif kind:
            winner = classification_winner(classification, server_id)
        else:
            winner = legacy_winner if legacy_winner in ("A", "B") else ""

        result.append({
            "index": index,
            "rally_id": _field(rally, "rally_id", ""),
            "server_id": server_id,
            "receiver_id": receiver_id,
            "score_before": (score_a, score_b),
            "scored_index": scored_index,
            "winner": winner,
            "kind": kind,
            "provisional": provisional,
        })
        if winner:
            if winner == "A":
                score_a += 1
            else:
                score_b += 1
            scored_index += 1
            # Once a match enters win-by-two overtime, each later server gets
            # one point (IRF 4.2.2.1), even if a full turn was half complete.
            overtime = overtime or _is_overtime(score_a, score_b, target_score)
            serve_in_turn += 1
            if overtime or turn == 0 or serve_in_turn == 2:
                turn += 1
                serve_in_turn = 0
        elif not rejected and kind != "redo" and not legacy_redo:
            provisional = True
    return result


def _field(rally: Any, name: str, default: Any = None) -> Any:
    if isinstance(rally, Mapping):
        return rally.get(name, default)
    return getattr(rally, name, default)


def _start(rally: Any) -> float:
    if isinstance(rally, (tuple, list)) and len(rally) == 2:
        return float(rally[0])
    return float(_field(rally, "start_time", _field(rally, "start", 0)))


def _partner(player_id: str) -> str:
    return player_id[0] + ("2" if player_id[1] == "1" else "1")


def _other_team(team: str) -> str:
    return "B" if team == "A" else "A"


def _is_overtime(score_a: int, score_b: int, target: int) -> bool:
    return max(score_a, score_b) >= target and abs(score_a - score_b) <= 1
