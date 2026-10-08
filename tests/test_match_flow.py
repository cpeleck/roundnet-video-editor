"""Serving order and clip classification, independent of GUI state."""

import pytest

from models.match_flow import (
    CLASSIFICATIONS,
    classification_winner,
    match_timeline,
    normalize_classification,
)
from models.rally import Rally


SETTINGS = {"starting_server": "A1", "starting_receiver": "B1"}


def clip(start, kind=None, **extra):
    result = {"start_time": start, "rally_id": f"clip-{start}", **extra}
    if kind is not None:
        result["classification"] = {"version": 1, "kind": kind}
    return result


def test_all_classifications_have_labels_and_explanations():
    assert set(CLASSIFICATIONS) == {
        "ace", "double_fault", "service_break", "sideout",
        "defensive_break", "defensive_hold", "error", "redo",
    }
    assert all(entry["label"] and entry["description"] for entry in CLASSIFICATIONS.values())


def test_normalize_classification_accepts_only_the_versioned_shape():
    assert normalize_classification({}) == {}
    assert normalize_classification({"version": 1, "kind": "ace"}) == {
        "version": 1, "kind": "ace",
    }
    assert normalize_classification({"version": 1, "kind": "error", "player_id": "B2"}) == {
        "version": 1, "kind": "error", "player_id": "B2",
    }
    invalid = (
        None,
        {"kind": "ace"},
        {"version": True, "kind": "ace"},
        {"version": 2, "kind": "ace"},
        {"version": 1, "kind": "mystery"},
        {"version": 1, "kind": "error"},
        {"version": 1, "kind": "error", "player_id": "Alex"},
        {"version": 1, "kind": "ace", "player_id": "A1"},
        {"version": 1, "kind": "redo", "extra": True},
    )
    for value in invalid:
        with pytest.raises(ValueError):
            normalize_classification(value)


def test_initial_and_full_serving_turns_switch_receivers():
    rows = match_timeline([clip(n, "ace") for n in range(10)], SETTINGS)
    assert [row["server_id"] for row in rows] == [
        "A1", "B2", "B2", "A2", "A2", "B1", "B1", "A1", "A1", "B2",
    ]
    assert [row["receiver_id"] for row in rows] == [
        "B1", "A2", "A1", "B1", "B2", "A1", "A2", "B1", "B2", "A2",
    ]
    assert [row["scored_index"] for row in rows] == list(range(10))
    assert [row["winner"] for row in rows] == [
        "A", "B", "B", "A", "A", "B", "B", "A", "A", "B",
    ]


def test_rotation_works_when_team_b_serves_first():
    settings = {"starting_server": "B2", "starting_receiver": "A2"}
    rows = match_timeline([clip(n, "ace") for n in range(5)], settings)
    assert [(row["server_id"], row["receiver_id"]) for row in rows] == [
        ("B2", "A2"), ("A1", "B1"), ("A1", "B2"),
        ("B1", "A2"), ("B1", "A1"),
    ]


@pytest.mark.parametrize(("kind", "winner"), [
    ("ace", "A"),
    ("service_break", "A"),
    ("defensive_hold", "A"),
    ("double_fault", "B"),
    ("sideout", "B"),
    ("defensive_break", "B"),
    ("redo", ""),
])
def test_classification_winner_uses_serving_or_receiving_team(kind, winner):
    classification = {"version": 1, "kind": kind}
    assert classification_winner(classification, "A1") == winner
    assert match_timeline([clip(0, kind)], SETTINGS)[0]["winner"] == winner


def test_error_awards_opposite_team_of_named_player():
    for player_id, winner in (("A1", "B"), ("A2", "B"), ("B1", "A"), ("B2", "A")):
        classification = {"version": 1, "kind": "error", "player_id": player_id}
        assert classification_winner(classification, "A1") == winner
        assert match_timeline([clip(0, **{"classification": classification})], SETTINGS)[0]["winner"] == winner


def test_redo_and_rejected_clip_preserve_score_and_serving_order():
    rows = match_timeline([
        clip(0, "redo", winner="A"),
        clip(1, "ace", rejected=True),
        clip(2, winner="B", outcome="Replay / no point"),
        clip(3, "ace"),
        clip(4, "sideout"),
    ], SETTINGS)
    assert [(row["server_id"], row["receiver_id"]) for row in rows[:4]] == [("A1", "B1")] * 4
    assert [row["score_before"] for row in rows] == [
        (0, 0), (0, 0), (0, 0), (0, 0), (1, 0),
    ]
    assert [row["winner"] for row in rows] == ["", "", "", "A", "A"]
    assert [row["scored_index"] for row in rows] == [0, 0, 0, 0, 1]
    assert all(not row["provisional"] for row in rows)


def test_new_classification_overrides_an_old_replay_tag():
    row = match_timeline([clip(0, "ace", outcome="Replay / no point", winner="B")], SETTINGS)[0]
    assert row["winner"] == "A"
    assert row["kind"] == "ace"


def test_chronological_order_legacy_winners_and_provisional_unknowns():
    legacy = Rally(10, 11, rally_id="legacy", winner="A")
    rows = match_timeline([
        clip(30, "sideout"),
        legacy,
        clip(20),
        clip(25, "redo"),
        clip(40, "ace", rejected=True),
    ], SETTINGS)
    assert [row["index"] for row in rows] == [1, 2, 3, 0, 4]
    assert [row["rally_id"] for row in rows] == [
        "legacy", "clip-20", "clip-25", "clip-30", "clip-40",
    ]
    assert [row["score_before"] for row in rows] == [
        (0, 0), (1, 0), (1, 0), (1, 0), (1, 0),
    ]
    assert [row["provisional"] for row in rows] == [False, False, True, True, True]
    assert [(row["server_id"], row["receiver_id"]) for row in rows] == [
        ("A1", "B1"), ("B2", "A2"), ("B2", "A2"),
        ("B2", "A2"), ("B2", "A2"),
    ]
    assert rows[3]["winner"] == ""


def test_overtime_switches_to_one_point_per_server_at_target():
    settings = {**SETTINGS, "initial_score_a": 20, "initial_score_b": 19}
    rows = match_timeline([clip(n, "ace") for n in range(6)], settings)
    assert [row["score_before"] for row in rows[:3]] == [
        (20, 19), (21, 19), (21, 20),
    ]
    assert [row["server_id"] for row in rows] == [
        "A1", "B2", "A2", "B1", "A1", "B2",
    ]
    assert [row["receiver_id"] for row in rows] == [
        "B1", "A2", "B1", "A1", "B1", "A2",
    ]


def test_invalid_starting_pair_or_score_settings_are_rejected():
    invalid = (
        {"starting_server": "A1", "starting_receiver": "A2"},
        {"starting_server": "someone", "starting_receiver": "B1"},
        {**SETTINGS, "target_score": True},
        {**SETTINGS, "initial_score_a": -1},
    )
    for settings in invalid:
        with pytest.raises(ValueError):
            match_timeline([], settings)


@pytest.mark.parametrize('server,receiver', [
    (s, r) for s in ('A1', 'A2', 'B1', 'B2')
    for r in ('A1', 'A2', 'B1', 'B2') if s[0] != r[0]
])
def test_next_turn_is_receiver_partner_against_server_partner(server, receiver):
    partner = lambda pid: pid[0] + ('2' if pid[1] == '1' else '1')
    rows = match_timeline([clip(0, 'sideout'), clip(1, 'redo'), clip(2, 'ace')],
                          {'starting_server': server, 'starting_receiver': receiver})
    assert (rows[1]['server_id'], rows[1]['receiver_id']) == (partner(receiver), partner(server))
    assert (rows[2]['server_id'], rows[2]['receiver_id']) == (partner(receiver), partner(server))
