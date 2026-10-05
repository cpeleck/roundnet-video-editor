from copy import deepcopy
from dataclasses import replace
import csv
import json

import pytest

from models import Rally
from models.point_stats import new_event, normalize_point_stats, normalize_roster
from models.project import EditHistory, read_project, write_project
from models.statistics import COUNTERS, calculate_statistics, export_statistics, original_rpr, point_issues
from video.presentation import match_statistics, prepare_export_options, statistics_lines


def event(player, kind, result="auto", **kwargs):
    return new_event(player, kind, result, **kwargs)


def point(events, winner="A", *, server="A1", receiver="B1", complete=True, start=0, **kwargs):
    return Rally(start, start+20, winner=winner, point_stats={
        "version": 1, "server_id": server, "receiver_id": receiver,
        "complete": complete, "events": events}, **kwargs)


def player(report, identity):
    return next(p for p in report["players"] if p["player_id"] == identity)


def long_rally():
    return point([event("A1", "serve", "in"), event("B1", "receive", "strong"),
                  event("B2", "set", "strong"), event("B1", "hit"),
                  event("A2", "defense"), event("A1", "set", "strong"), event("A2", "hit")])


def test_touch_order_credits_actual_defender_and_hitter():
    r = long_rally()
    assert point_issues(r.point_stats, r.winner) == []
    report = calculate_statistics([r])
    assert player(report, "A2")["defensive_gets"] == 1
    assert player(report, "A1")["defensive_gets"] == 0
    assert player(report, "A1")["strong_sets"] == 1
    assert player(report, "A2")["put_aways"] == 1
    assert player(report, "B1")["hits_returned"] == 1
    assert player(report, "B1")["put_away_pct"] == 0
    assert player(report, "A2")["put_away_pct"] == 1
    assert report["teams"]["A"]["breaks"] == 1
    assert player(report, "A1")["breaks"] == player(report, "A2")["breaks"] == 1
    assert player(report, "B1")["broken"] == 1


def test_get_survives_later_loss_and_multiple_possessions():
    r = long_rally()
    r.point_stats["events"] += [event("B2", "defense"), event("B1", "set", "strong"), event("B2", "hit")]
    r.winner = "B"
    report = calculate_statistics([r])
    assert not report["warnings"]
    assert player(report, "A2")["defensive_gets"] == 1
    assert player(report, "B2")["defensive_gets"] == 1
    assert player(report, "B2")["put_aways"] == 1
    assert report["teams"]["B"]["sideouts"] == 1
    assert report["teams"]["A"]["breaks"] == 0


def test_putaway_when_opponent_touches_then_partner_misses_set():
    r = point([event("A1", "serve", "in"), event("B1", "receive", "strong"), event("B2", "hit"),
               event("A1", "defense"), event("A2", "set", "error")], winner="B")
    report = calculate_statistics([r])
    assert not report["warnings"]
    assert player(report, "B2")["put_aways"] == 1
    assert player(report, "A1")["defensive_not_returned"] == 1
    assert player(report, "A1")["defensive_gets"] == 0
    assert player(report, "A2")["set_errors"] == player(report, "A2")["errors"] == 1


def test_serve_retries_lets_aces_and_opponent_matrix():
    r = point([event("A1", "serve", "let"), event("A1", "serve", "rim"), event("A1", "serve", "ace")])
    report = calculate_statistics([r])
    p = player(report, "A1")
    assert (p["serve_attempts"], p["serves_in"], p["rims"], p["lets"]) == (2, 1, 1, 1)
    assert p["serve_pct"] == p["ace_pct"] == .5
    assert player(report, "B1")["aced"] == player(report, "B1")["receives"] == 1
    assert report["aces_by_opponent"] == [{"server_id": "A1", "receiver_id": "B1", "aces": 1}]


@pytest.mark.parametrize("fault_count,expected", [(1, 0), (2, 1)])
def test_terminal_fault_point(fault_count, expected):
    r = point([event("A1", "serve", "fault") for _ in range(fault_count)], winner="B")
    report = calculate_statistics([r])
    assert not report["warnings"]
    p = player(report, "A1")
    assert p["double_faults"] == expected
    assert p["serve_attempts"] == fault_count
    assert p["errors"] == 0
    assert player(report, "B1")["receives"] == 0


def test_partial_auto_does_not_invent_success_or_failure():
    r = long_rally()
    r.point_stats["complete"] = False
    r.winner = ""
    report = calculate_statistics([r])
    assert report["coverage"]["partial"] == 1
    assert player(report, "A2")["defensive_gets"] == 1  # earlier net return is observed
    assert player(report, "A2")["put_aways"] == 0
    assert player(report, "A2")["unknown_results"] == 1
    assert player(report, "A2")["put_away_pct"] is None
    assert not report["rpr_eligible"]


def test_no_touch_is_not_defensive_touch():
    r = point([event("A1", "serve", "in"), event("B1", "hit"), event("A1", "defense", "no_touch")], winner="B")
    report = calculate_statistics([r])
    assert not report["warnings"]
    assert player(report, "A1")["no_touches"] == 1
    assert player(report, "A1")["defensive_touches"] == 0
    assert player(report, "B1")["put_aways"] == 1


def test_errors_and_tough_touches_are_distinct_from_weak_quality():
    r = point([event("A1", "serve", "in"), event("B1", "receive", "weak", tough=True),
               event("B2", "set", "weak"), event("B1", "hit", "error")])
    report = calculate_statistics([r])
    assert not report["warnings"]
    assert player(report, "B1")["errors"] == 1
    assert player(report, "B1")["tough_touches"] == 1
    assert player(report, "B2")["weak_sets"] == 1
    assert player(report, "B2")["tough_touches"] == 0
    assert player(report, "B1")["error_pct"] == .5


def test_disabled_points_count_rejected_and_replays_do_not():
    ace = point([event("A1", "serve", "ace")], enabled=False)
    rejected = replace(ace, rally_id="rejected", rejected=True)
    replay = replace(ace, rally_id="replay", outcome="Replay / no point")
    report = calculate_statistics([ace, rejected, replay])
    assert player(report, "A1")["aces"] == 1
    assert report["coverage"]["points"] == 1
    assert report["coverage"]["rejected"] == report["coverage"]["replays"] == 1


def test_empty_and_legacy_points_do_not_fabricate_player_stats():
    report = calculate_statistics([Rally(0, 2, winner="A", player="Someone", outcome="Ace")],
                                  {"initial_score_a": 5})
    assert report["coverage"]["untagged"] == 1
    assert report["teams"]["A"]["points_won"] == 1
    assert report["teams"]["A"]["score"] == 6
    assert report["player_tags"] == {"Someone": 1}
    assert report["outcome_tags"] == {"Ace": 1}
    assert all(p["serve_pct"] is None and p["aces"] == 0 and p["rpr"]["overall"] is None for p in report["players"])


def test_original_rpr_exact_formula_unclamped_and_long_game_scaling():
    p = dict.fromkeys(COUNTERS, 0)
    p.update(hit_attempts=10, hits_returned=2, serve_attempts=8, serves_in=6,
             aces=4, defensive_gets=8, defensive_not_returned=2, set_errors=1, hit_errors=1,
             tough_touches=1, aced=2)
    rating = original_rpr(p, 44, eligible=True)
    assert rating == pytest.approx({"hitting": 16, "serving": 33.25, "defense": 53.2,
                                    "efficiency": 4, "overall": 106.45})
    assert original_rpr(p, 88, eligible=True)["defense"] == pytest.approx(26.6)
    p["set_errors"] = 10
    assert original_rpr(p, 44, eligible=True)["efficiency"] < 0
    p["serve_attempts"] = 0
    assert original_rpr(p, 44, eligible=True)["overall"] is None


def test_rpr_requires_full_coverage_and_explicit_match_confirmation():
    r = long_rally()
    assert not calculate_statistics([r])["rpr_eligible"]
    assert calculate_statistics([r], {"stats_complete": True})["rpr_eligible"]
    assert player(calculate_statistics([r], {"stats_complete": True}), "A1")["rpr"]["overall"] is None  # no hit
    assert not calculate_statistics([r], {"stats_complete": True, "initial_score_a": 1})["rpr_eligible"]
    assert not calculate_statistics([r, Rally(25, 27)], {"stats_complete": True})["rpr_eligible"]
    r.point_stats["events"][1]["result"] = "unknown"
    assert not calculate_statistics([r], {"stats_complete": True})["rpr_eligible"]


@pytest.mark.parametrize("mutate,fragment", [
    (lambda p: p.update(receiver_id="A2"), "opposite"),
    (lambda p: p["events"][0].update(player_id="A2"), "selected server"),
    (lambda p: p["events"][1].update(player_id="B2"), "selected receiver"),
    (lambda p: p["events"][3].update(result="put_away"), "put-away"),
    (lambda p: p["events"][4].update(result="touch_not_returned"), "successful net return"),
    (lambda p: p["events"][2].update(result="error"), "final event"),
    (lambda p: p["events"].append(event("A1", "serve", "ace")), "precede"),
    (lambda p: p["events"][4].update(player_id="B2"), "wrong team"),
])
def test_inconsistent_logs_are_flagged_and_excluded(mutate, fragment):
    r = long_rally()
    mutate(r.point_stats)
    report = calculate_statistics([r])
    assert report["coverage"]["invalid"] == 1
    assert fragment in " ".join(report["warnings"])
    assert all(p["serve_attempts"] == 0 for p in report["players"])


def test_complete_point_needs_terminal_evidence():
    r = point([event("A1", "serve", "in"), event("B1", "receive", "strong")])
    assert "terminal" in " ".join(point_issues(r.point_stats, r.winner))


def test_second_defensive_touch_in_same_possession_cannot_double_credit_get():
    r = long_rally()
    r.point_stats["events"].insert(5, event("A1", "defense"))
    assert "first contact" in " ".join(point_issues(r.point_stats, r.winner))
    assert calculate_statistics([r])["coverage"]["invalid"] == 1


def test_draft_receive_counts_even_without_selected_receiver():
    r = point([event("A1", "serve", "in"), event("B1", "receive", "strong")],
              receiver="", complete=False, winner="")
    stats = calculate_statistics([r])
    assert player(stats, "B1")["receives"] == 1
    assert player(stats, "B1")["receive_pct"] == 1


def test_replay_never_changes_any_score_even_with_legacy_winner():
    from video.presentation import scores_before_rallies
    replay = point([event("A1", "serve", "ace")], outcome="Replay / no point")
    following = long_rally()
    following.start_time = 22
    following.end_time = 30
    options = prepare_export_options()
    assert scores_before_rallies([replay, following], options)[1] == (0, 0)
    summary = match_statistics([replay, following])
    assert summary["score_a"] == summary["rally_count"] == 1
    assert summary["player_statistics"]["coverage"]["replays"] == 1


def test_timestamps_checked_against_order_and_cut_bounds():
    r = long_rally()
    r.point_stats["events"][0]["time"] = 5
    r.point_stats["events"][1]["time"] = 4
    r.point_stats["events"][-1]["time"] = 40
    issues = point_issues(r.point_stats, "A", start=0, end=20)
    assert any("order" in i for i in issues)
    assert any("outside" in i for i in issues)


@pytest.mark.parametrize("bad", [None, [], {"version": 9}, {"version": 1, "complete": "yes"},
                                 {"version": 1, "server_id": "X"}, {"version": 1, "events": "abc"}])
def test_invalid_storage_is_rejected(bad):
    with pytest.raises(ValueError):
        normalize_point_stats(bad)


def test_event_validation_and_nested_copy():
    e = event("A1", "serve", "in")
    with pytest.raises(ValueError):
        normalize_point_stats({"version": 1, "events": [e, e]})
    with pytest.raises(ValueError):
        event("A1", "hit", "ace")
    with pytest.raises(ValueError):
        event("A1", "set", "strong", tough=True)
    with pytest.raises(ValueError):
        event("A1", "set", "strong", time=float("nan"))
    r = long_rally()
    other = Rally.from_dict(r.to_dict())
    other.point_stats["events"].clear()
    assert r.point_stats["events"]


def test_roster_rename_preserves_credits_and_validation():
    roster = normalize_roster()
    roster[0]["name"] = "Christian"
    report = calculate_statistics([point([event("A1", "serve", "ace")])], {"players": roster})
    assert player(report, "A1")["name"] == "Christian"
    assert player(report, "A1")["aces"] == 1
    with pytest.raises(ValueError):
        normalize_roster(roster[:3])
    roster[1]["name"] = "CHRISTIAN"
    with pytest.raises(ValueError):
        normalize_roster(roster)


def test_projects_history_and_legacy_roundtrip(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    r = long_rally()
    state = {"video_path": str(source), "video_duration": 20, "rallies": [r.to_dict()],
             "match_settings": {"players": normalize_roster(), "stats_complete": True}}
    saved = write_project(tmp_path / "project.json", state)
    restored = read_project(saved)
    assert restored["rallies"][0]["point_stats"] == r.point_stats
    assert restored["match_settings"] == state["match_settings"]
    history = EditHistory()
    history.push(state)
    state["rallies"][0]["point_stats"]["events"].clear()
    assert history.undo(state)["rallies"][0]["point_stats"]["events"]
    assert Rally.from_dict({"start": 0, "end": 1}).point_stats == {}


def test_json_csv_and_presentation_exports(tmp_path):
    roster = normalize_roster()
    roster[0]["name"] = "=untrusted()"
    report = calculate_statistics([long_rally()], {"players": roster})
    path = export_statistics(tmp_path / "stats.json", report)
    assert json.loads(path.read_text())["points"][0]["events"][-1]["result"] == "put_away"
    csv_path = export_statistics(tmp_path / "stats.csv", report)
    with csv_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["name"].startswith("'=")
    assert rows[0]["rpr"] == ""
    with pytest.raises(ValueError):
        export_statistics(path, report, protected_paths=[path])
    with pytest.raises(ValueError):
        export_statistics(tmp_path / "bad.txt", report)
    options = prepare_export_options({"players": roster, "stats_complete": True})
    summary = match_statistics([long_rally()], options)
    assert summary["player_statistics"]["coverage"]["complete"] == 1
    assert any("put-away" in line for line in statistics_lines(summary))
