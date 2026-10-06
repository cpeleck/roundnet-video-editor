"""Portable project saves, recoverable edits, and rally metadata regression tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from models.project import EditHistory, read_project, recovery_path, validate_project, write_project
from models.rally import Rally


def _annotated_rally() -> Rally:
    return Rally(
        3.5, 8.25, confidence=0.72, serve_confidence=0.84, rally_id="point-one",
        reviewed=True, starred=True, outcome="Ace", winner="A", player="Alex",
        note="Near sideline", crop_keyframes=[
            {"time": 3.5, "x": 0.3, "y": 0.4},
            {"time": 7.0, "x": 0.65, "y": 0.5},
        ],
    )


def _state(source: Path) -> dict:
    return {
        "video_path": str(source), "video_duration": 30.0,
        "rallies": [_annotated_rally().to_dict()],
        "settings": {"rally_threshold": 0.55},
        "match_settings": {"team_a": "North", "team_b": "South", "initial_score_a": 2},
        "court_context": {"net": [0.5, 0.6], "net_radius": 0.06, "serving_zones": []},
        "roi": [0.1, 0.1, 0.8, 0.8], "selected": 0, "complete": False,
        "rejected_detections": [{"start": 15, "end": 18, "reason": "retrieval"}],
        "signals": {"timestamps": [0, 1], "motion_scores": [0.1, 0.8]},
        "profile_path": "", "playhead": 5.25,
    }


def test_rally_metadata_round_trip_and_json_owned_snapshot() -> None:
    rally = _annotated_rally()
    snapshot = rally.to_dict()
    assert Rally.from_dict(json.loads(json.dumps(snapshot))) == rally
    snapshot["crop_keyframes"][0]["x"] = 0.9
    assert rally.crop_keyframes[0]["x"] == 0.3


def test_loaded_rally_does_not_alias_serialized_crop_keyframes() -> None:
    data = _annotated_rally().to_dict()
    restored = Rally.from_dict(data)
    data["crop_keyframes"][0]["x"] = 0.9
    assert restored.crop_keyframes[0]["x"] == 0.3


def test_legacy_rally_defaults_remain_compatible() -> None:
    rally = Rally.from_dict({"start": 1, "end": 4})
    assert rally.start_time == 1
    assert rally.end_time == 4
    assert rally.enabled and not rally.rejected and not rally.reviewed
    assert rally.crop_keyframes == []
    assert rally.rally_id != Rally.from_dict({"start": 1, "end": 4}).rally_id


def test_boundary_edits_preserve_point_identity_and_annotations() -> None:
    original = _annotated_rally()
    edited = original.with_bounds(4, 7)
    assert edited.rally_id == original.rally_id
    assert edited.winner == "A"
    assert edited.starred and edited.reviewed
    assert edited.crop_keyframes == original.crop_keyframes
    assert original.start_time == 3.5
    assert original.end_time == 8.25


@pytest.mark.parametrize("changes,error", [
    ({"reviewed": 1}, TypeError),
    ({"starred": "yes"}, TypeError),
    ({"rejected": True, "enabled": True}, ValueError),
    ({"winner": "C"}, ValueError),
    ({"rally_id": ""}, ValueError),
    ({"note": 42}, TypeError),
    ({"crop_keyframes": [{"time": 4, "x": 1.1, "y": 0.5}]}, ValueError),
    ({"crop_keyframes": [{"time": float("nan"), "x": 0.5, "y": 0.5}]}, ValueError),
    ({"crop_keyframes": [{"time": 4, "x": 0.5, "y": 0.5}, {"time": 4, "x": 0.6, "y": 0.5}]}, ValueError),
])
def test_rally_rejects_invalid_annotation_metadata(changes: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        replace(_annotated_rally(), **changes)


def test_rejected_rally_is_persisted_for_restore() -> None:
    rally = replace(_annotated_rally(), enabled=False, rejected=True)
    recovered = Rally.from_dict(rally.to_dict())
    assert recovered == rally
    restored = replace(recovered, enabled=True, rejected=False)
    assert restored.rally_id == rally.rally_id
    assert restored.winner == "A"


def test_project_round_trip_preserves_all_review_context(tmp_path: Path) -> None:
    source = tmp_path / "game.mp4"
    source.write_bytes(b"unchanged original footage")
    state = _state(source)
    original = deepcopy(state)
    destination = tmp_path / "projects" / "game.roundnet.json"
    assert write_project(destination, state) == destination.resolve()
    recovered = read_project(destination)
    for key, value in original.items():
        assert recovered[key] == value
    assert recovered["schema_version"] == 1
    assert recovered["kind"] == "roundnet_project"
    assert recovered["source_stat"]["size"] == source.stat().st_size
    assert recovered["source_stat"]["mtime_ns"] == source.stat().st_mtime_ns
    assert recovered["saved_utc"]
    assert state == original
    assert source.read_bytes() == b"unchanged original footage"


def test_project_and_media_can_be_moved_together(tmp_path: Path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    source = old / "game.mp4"
    source.write_bytes(b"original")
    project = old / "edit.roundnet.json"
    write_project(project, _state(source))
    new = tmp_path / "moved"
    shutil.move(str(old), str(new))
    assert read_project(new / project.name)["video_path"] == str(new / source.name)


def test_project_save_cannot_replace_source_video(tmp_path: Path) -> None:
    source = tmp_path / "game.mp4"
    source.write_bytes(b"original")
    with pytest.raises(ValueError, match="source video"):
        write_project(source, _state(source))
    assert source.read_bytes() == b"original"


def test_missing_source_field_fails_validation_before_save(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source video"):
        write_project(tmp_path / "project.json", {})


def test_non_object_project_file_is_rejected_with_readable_error(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    path.write_text("[]")
    with pytest.raises(ValueError):
        read_project(path)


def test_source_fingerprint_detects_replaced_and_missing_media(tmp_path: Path) -> None:
    from models.project import source_matches
    source = tmp_path / "game.mp4"
    source.write_bytes(b"original")
    destination = tmp_path / "project.json"
    write_project(destination, _state(source))
    state = read_project(destination)
    assert source_matches(state, source)
    source.write_bytes(b"replacement video")
    assert not source_matches(state, source)
    assert not source_matches(state, tmp_path / "missing.mp4")


def test_invalid_project_cannot_replace_previous_good_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "game.mp4"
    source.write_bytes(b"original")
    state = _state(source)
    destination = tmp_path / "project.json"
    write_project(destination, state)
    saved = destination.read_bytes()
    state["video_duration"] = float("nan")
    with pytest.raises(ValueError, match="duration"):
        write_project(destination, state)
    assert destination.read_bytes() == saved


def test_starting_server_and_receiver_survive_project_save(tmp_path: Path) -> None:
    source = tmp_path / "game.mp4"
    source.write_bytes(b"video")
    state = _state(source)
    state["match_settings"].update(starting_server="B2", starting_receiver="A1")
    saved = write_project(tmp_path / "match.json", state)
    assert read_project(saved)["match_settings"]["starting_server"] == "B2"
    state["match_settings"]["starting_receiver"] = "B1"
    with pytest.raises(ValueError, match="opposite teams"):
        validate_project(state)


def test_interrupted_atomic_replace_leaves_previous_save_and_no_temp_file(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "game.mp4"
    source.write_bytes(b"original")
    destination = tmp_path / "project.json"
    state = _state(source)
    write_project(destination, state)
    saved = destination.read_bytes()
    state["playhead"] = 8

    def fail_replace(*_args):
        raise OSError("simulated interrupted filesystem write")

    monkeypatch.setattr("models.project.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        write_project(destination, state)
    assert destination.read_bytes() == saved
    assert list(tmp_path.glob(".roundnet-*.tmp")) == []


@pytest.mark.parametrize("field,value", [
    ("video_path", None), ("video_duration", -1), ("video_duration", float("inf")),
    ("settings", []), ("match_settings", "invalid"),
])
def test_project_validation_rejects_invalid_fields(tmp_path: Path, field: str, value) -> None:
    state = _state(tmp_path / "game.mp4")
    state[field] = value
    with pytest.raises(ValueError):
        validate_project(state)


def test_duplicate_rally_ids_are_rejected(tmp_path: Path) -> None:
    state = _state(tmp_path / "game.mp4")
    state["rallies"].append(deepcopy(state["rallies"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_project(state)


@pytest.mark.parametrize("metadata", [
    {"kind": "another_application", "schema_version": 1},
    {"kind": "roundnet_project", "schema_version": 999},
])
def test_unrecognized_project_schema_is_rejected(tmp_path: Path, metadata: dict) -> None:
    path = tmp_path / "project.json"
    path.write_text(json.dumps(dict(_state(tmp_path / "game.mp4"), **metadata)))
    with pytest.raises(ValueError, match="Unsupported"):
        read_project(path)


def test_recovery_path_is_stable_and_distinguishes_same_named_sources(tmp_path: Path) -> None:
    recovery = tmp_path / "recovery"
    first = tmp_path / "first" / "game.mp4"
    same = tmp_path / "first" / ".." / "first" / "game.mp4"
    second = tmp_path / "second" / "game.mp4"
    assert recovery_path(str(first), recovery) == recovery_path(str(same), recovery)
    assert recovery_path(str(first), recovery) != recovery_path(str(second), recovery)
    assert recovery_path(str(first), recovery).parent == recovery


def test_undo_redo_snapshots_are_independent_and_branching_clears_redo() -> None:
    history = EditHistory()
    initial = {"rallies": [_annotated_rally().to_dict()]}
    expected = deepcopy(initial)
    history.push(initial)
    initial["rallies"][0]["note"] = "later edit"
    undone = history.undo(initial)
    assert undone == expected
    initial["rallies"][0]["note"] = "unrelated mutation"
    redone = history.redo(undone)
    assert redone["rallies"][0]["note"] == "later edit"
    history.undo(redone)
    history.push({"rallies": []})
    assert history.redo({"rallies": []}) is None


def test_history_is_bounded_and_empty_history_is_safe() -> None:
    history = EditHistory(limit=2)
    assert history.undo({}) is None
    assert history.redo({}) is None
    for value in range(4):
        history.push({"value": value})
    assert history.undo_stack == [{"value": 2}, {"value": 3}]
    assert history.undo({"value": 4}) == {"value": 3}
    assert history.undo({"value": 3}) == {"value": 2}
    assert history.undo({"value": 2}) is None
    assert history.redo({"value": 2}) == {"value": 3}
    assert history.redo({"value": 3}) == {"value": 4}


@pytest.mark.parametrize("limit", [0, -1])
def test_history_rejects_non_positive_limits(limit: int) -> None:
    with pytest.raises(ValueError):
        EditHistory(limit=limit)
