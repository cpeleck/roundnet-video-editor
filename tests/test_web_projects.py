"""Application commands update domain views and durable state together."""

from copy import deepcopy
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from application.projects import ProjectError, ProjectService
from backend.app import create_app
from models.point_stats import new_event
from models.project import read_project, write_project
from models.rally import Rally
from models.statistics import calculate_statistics


@pytest.fixture
def service(tmp_path):
    return ProjectService(tmp_path / "data", metadata_probe=lambda path: dict(
        path=path, duration_seconds=60, width=640, height=360, fps=30,
        video_codec="h264", has_audio=False, rotation_degrees=0))


def new_project(service, tmp_path):
    source = tmp_path / "match.mp4"
    source.write_bytes(b"test recording")
    project = service.create(str(source), title="Saturday match")
    pid = project["project_id"]
    return service.command(pid, {"type": "setup", "revision": project["revision"], "mode": "manual", "match_settings": {
        "team_a": "North", "team_b": "South", "starting_server": "A1", "starting_receiver": "B1",
        "players": [{"player_id": p, "team": p[0], "name": name} for p, name in zip(
            ("A1", "A2", "B1", "B2"), ("Connor", "Morgan", "Jacob", "Alex"))]}})


def add(service, project, start, end):
    return service.command(project["project_id"], {"type": "add", "revision": project["revision"],
                                                 "start_time": start, "end_time": end})


def edit(service, project, command_type, **changes):
    return service.command(project["project_id"], {"type": command_type, "revision": project["revision"], **changes})


def test_classification_credit_rotation_autosave_and_history(service, tmp_path):
    project = add(service, new_project(service, tmp_path), 2, 5)
    project = add(service, project, 10, 14)
    first, second = [r["rally_id"] for r in project["rallies"]]
    project = edit(service, project, "classify", rally_id=first, kind="ace")
    assert project["timeline"][1]["server_id"] == "B2"
    assert project["selected"] == 1
    assert project["statistics"]["teams"]["A"]["score"] == 1
    connor, _, jacob, _ = project["statistics"]["players"]
    assert connor["aces"] == jacob["aced"] == 1
    assert connor["serve_pct"] is None
    assert "Connor +1 ace; Jacob +1 aced" in project["explanation"]
    saved = read_project(service.project_dir / f"{project['project_id']}.roundnet.json")
    assert saved["revision"] == project["revision"]
    assert saved["rallies"][0]["winner"] == "A"
    project = edit(service, project, "classify", rally_id=second, kind="double_fault")
    assert project["statistics"]["teams"]["A"]["score"] == 2
    restored = edit(service, project, "undo")
    assert restored["statistics"]["teams"]["A"]["score"] == 1
    assert restored["can_redo"]
    redone = edit(service, restored, "redo")
    assert redone["statistics"]["teams"]["A"]["score"] == 2
    restarted = ProjectService(service.data_dir, metadata_probe=service.metadata_probe)
    assert restarted.snapshot(project["project_id"])["statistics"] == redone["statistics"]
    assert restarted.snapshot(project["project_id"])["stage"] == "review"


def test_missing_outcome_errors_culprit_and_redo(service, tmp_path):
    project = add(service, new_project(service, tmp_path), 2, 5)
    project = add(service, project, 10, 14)
    first, second = [r["rally_id"] for r in project["rallies"]]
    with pytest.raises(ProjectError) as missing:
        edit(service, project, "classify", rally_id=second, kind="ace")
    assert missing.value.code == "missing_outcome"
    assert missing.value.details["rally_id"] == first
    with pytest.raises(ValueError, match="player_id"):
        edit(service, project, "classify", rally_id=first, kind="error")
    project = edit(service, project, "classify", rally_id=first, kind="redo")
    assert project["timeline"][1]["server_id"] == "A1"
    project = edit(service, project, "classify", rally_id=second, kind="error", player_id="A2")
    assert project["statistics"]["teams"]["B"]["score"] == 1
    assert project["statistics"]["players"][1]["errors"] == 1


def test_presentation_edits_and_highlights_preserve_confirmation_and_scoring(service, tmp_path):
    project = add(service, new_project(service, tmp_path), 2, 5)
    identity = project["rallies"][0]["rally_id"]
    project = edit(service, project, "classify", rally_id=identity, kind="ace")
    point = dict(version=1, server_id="A1", receiver_id="B1", complete=True,
                 events=[new_event("A1", "serve", "ace", time=3)])
    project = edit(service, project, "touches", rally_id=identity, point_stats=point)
    project = edit(service, project, "full_match", confirmed=True)
    assert project["statistics"]["rpr_eligible"]
    revision = project["revision"]
    project = edit(service, project, "classify", rally_id=identity, kind="ace")
    assert project["revision"] == revision
    assert project["rallies"][0]["point_stats"] == point
    assert project["match_settings"]["stats_complete"]
    before = deepcopy(project["statistics"])
    for kind, fields in [("star", {"starred": True}), ("caption", {"note": "Great serve"}),
                         ("export_enabled", {"enabled": False}),
                         ("crop", {"crop_keyframes": [{"time": 2, "x": .4, "y": .5}]})]:
        project = edit(service, project, kind, rally_id=identity, **fields)
        assert project["match_settings"]["stats_complete"]
        assert project["statistics"]["rpr_eligible"]
        assert project["statistics"]["teams"] == before["teams"]
    canonical = deepcopy(project["rallies"])
    project = edit(service, project, "highlight_queue", clips=[{
        "rally_id": identity, "start_time": 2.5, "end_time": 4.5,
        "crop_keyframes": [{"time": 2.5, "x": .6, "y": .5}]}])
    assert project["rallies"] == canonical
    assert project["statistics"]["rpr_eligible"]
    project = edit(service, project, "reject", rally_id=identity, rejected=True)
    assert project["statistics"]["teams"]["A"]["score"] == 0
    assert not project["match_settings"]["stats_complete"]
    assert project["highlight_queue"] == []


def test_revision_transaction_and_busy_job_guards(service, tmp_path):
    project = add(service, new_project(service, tmp_path), 2, 5)
    identity = project["rallies"][0]["rally_id"]
    with pytest.raises(ProjectError, match="project changed"):
        service.command(project["project_id"], {"type": "star", "revision": 0, "rally_id": identity})
    with pytest.raises(ValueError):
        edit(service, project, "trim", rally_id=identity, start_time=8, end_time=6)
    assert service.snapshot(project["project_id"])["rallies"] == project["rallies"]
    assert service.revision(project["project_id"]) == project["revision"]
    service.lock_job(project["project_id"], "job1", project["revision"])
    with pytest.raises(ProjectError) as busy:
        edit(service, project, "star", rally_id=identity)
    assert busy.value.code == "project_busy"
    result = service.finish_job(project["project_id"], "job1", {
        "rallies": [Rally(8, 12).to_dict()], "stage": "review", "signals": {"timestamps": [8], "motion": [.4], "rally_score": [.6]}})
    assert not result["busy_job_id"]
    assert result["rallies"][0]["start_time"] == 8
    assert result["signals"]["motion_scores"] == [.4]
    assert result["signals"]["rally_scores"] == [.6]
    assert edit(service, result, "undo")["rallies"] == project["rallies"]


def test_changed_source_during_detection_is_never_committed_or_blessed_by_recovery(service, tmp_path):
    project = add(service, new_project(service, tmp_path), 2, 5)
    before = service.get_state(project["project_id"])
    service.lock_job(project["project_id"], "job", project["revision"])
    Path(project["video_path"]).write_bytes(b"replacement source during analysis")
    with pytest.raises(ProjectError) as mismatch:
        service.finish_job(project["project_id"], "job", {"rallies": [Rally(8, 12).to_dict()]})
    assert mismatch.value.code == "source_mismatch"
    service.unlock_job(project["project_id"], "job")
    assert service.get_state(project["project_id"]) == before
    recovered = ProjectService(service.data_dir, metadata_probe=service.metadata_probe)
    assert recovered.snapshot(project["project_id"])["source_changed"]
    assert recovered.get_state(project["project_id"])["source_stat"] == before["source_stat"]


def test_open_old_project_relative_source_and_relink_validation(service, tmp_path):
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original recording")
    path = tmp_path / "older.roundnet.json"
    old = dict(video_path=str(source), video_duration=60,
               settings={"debug_mode": True, "prefer_hardware": False, "save_labels_after_export": True}, match_settings={},
               rallies=[Rally(2, 5, classification={"version": 1, "kind": "ace"}, winner="A").to_dict()])
    write_project(path, old)
    opened = service.open(str(path))
    assert opened["match_settings"]["setup_complete"]
    assert opened["settings"]["debug_mode"]
    assert opened["settings"]["save_labels_after_export"]
    assert not opened["export_settings"]["prefer_hardware"]
    assert opened["statistics"] == calculate_statistics([Rally.from_dict(r) for r in opened["rallies"]], opened["match_settings"])
    assert opened["rallies"][0]["outcome"] == "Ace"
    moved = tmp_path / "moved"
    moved.mkdir()
    moved_path = moved / path.name
    moved_source = moved / source.name
    path.rename(moved_path)
    source.rename(moved_source)
    reopened = service.open(str(moved_path))
    assert reopened["video_path"] == str(moved_source)
    moved_source.write_bytes(b"a different recording")
    with pytest.raises(ProjectError) as mismatch:
        service.open(str(moved_path))
    assert mismatch.value.code == "source_mismatch"
    accepted = service.open(str(moved_path), allow_source_mismatch=True)
    assert accepted["rallies"][0]["classification"] == {}
    assert not accepted["match_settings"]["stats_complete"]
    assert accepted["source_status"] == "ready"
    accepted = edit(service, accepted, "classify", rally_id=accepted["rallies"][0]["rally_id"], kind="ace")
    moved_source.unlink()
    assert service.snapshot(accepted["project_id"])["source_missing"]
    with pytest.raises(ProjectError):
        edit(service, accepted, "star", rally_id=accepted["rallies"][0]["rally_id"])


def test_split_merge_clear_counts_and_validate_source_bounds(service, tmp_path):
    project = add(service, new_project(service, tmp_path), 2, 8)
    identity = project["rallies"][0]["rally_id"]
    project = edit(service, project, "classify", rally_id=identity, kind="ace")
    project = edit(service, project, "split", rally_id=identity, time=5)
    assert len(project["rallies"]) == 2
    assert all(not r["classification"] for r in project["rallies"])
    assert project["statistics"]["teams"]["A"]["score"] == 0
    project = edit(service, project, "merge", rally_ids=[r["rally_id"] for r in project["rallies"]])
    assert len(project["rallies"]) == 1
    assert (project["rallies"][0]["start_time"], project["rallies"][0]["end_time"]) == (2, 8)
    with pytest.raises(ValueError, match="beyond"):
        add(service, project, 55, 65)


def test_save_target_survives_undo_and_changed_source_cannot_restore_old_credits(service, tmp_path):
    project = add(service, new_project(service, tmp_path), 2, 5)
    identity = project["rallies"][0]["rally_id"]
    project = edit(service, project, "classify", rally_id=identity, kind="ace")
    destination = tmp_path / "saved.roundnet.json"
    project = edit(service, project, "save_as", path=str(destination))
    project = edit(service, project, "undo")
    assert project["project_path"] == str(destination)
    assert read_project(destination)["rallies"][0]["classification"] == {}
    project = edit(service, project, "redo")
    assert read_project(destination)["rallies"][0]["classification"]["kind"] == "ace"
    source = Path(project["video_path"])
    source.write_bytes(b"new recording replacing original")
    project = edit(service, project, "relink", video_path=str(source), allow_source_mismatch=True)
    assert not project["can_undo"]
    assert not project["can_redo"]
    assert project["rallies"][0]["classification"] == {}


def test_reopening_an_earlier_snapshot_keeps_revisions_monotonic_and_blocks_while_busy(service, tmp_path):
    project = add(service, new_project(service, tmp_path), 2, 5)
    identity = project["rallies"][0]["rally_id"]
    earlier = tmp_path / "earlier.roundnet.json"
    write_project(earlier, service.get_state(project["project_id"]))
    project = edit(service, project, "classify", rally_id=identity, kind="ace")
    revision = project["revision"]
    service.lock_job(project["project_id"], "export-job", revision)
    with pytest.raises(ProjectError) as busy:
        service.open(str(earlier))
    assert busy.value.code == "project_busy"
    assert service.revision(project["project_id"]) == revision
    service.finish_job(project["project_id"], "export-job")
    assert service.revision(project["project_id"]) == revision
    reopened = service.open(str(earlier))
    assert reopened["revision"] == revision + 1
    assert not reopened["rallies"][0]["classification"]
    with pytest.raises(ProjectError) as stale:
        edit(service, project, "star", rally_id=identity)
    assert stale.value.code == "revision_conflict"


def test_http_commands_conflict_and_local_origin_boundary(service, tmp_path):
    with TestClient(create_app(service=service)) as client:
        assert client.get("/api/health").status_code == 200
        assert len(client.get("/api/defaults").json()["classifications"]) == 8
        source = tmp_path / "http.mp4"
        source.write_bytes(b"recording")
        response = client.post("/api/projects", json={"video_path": str(source)})
        assert response.status_code == 200
        project = response.json()
        pid = project["project_id"]
        assert client.get("/api/projects").json()["projects"][0]["project_id"] == pid
        result = client.post(f"/api/projects/{pid}/commands", json={"type": "setup", "revision": 0,
            "mode": "manual", "match_settings": {}})
        assert result.status_code == 200
        conflict = client.post(f"/api/projects/{pid}/commands", json={"type": "stage", "revision": 0, "stage": "review"})
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["code"] == "revision_conflict"
        assert client.post("/api/projects", json={}, headers={"Origin": "https://untrusted.example"}).status_code == 403
        assert client.get("/api/health", headers={"Host": "untrusted.example"}).status_code == 403
        assert client.post("/api/projects", json={}).status_code == 422
        download = client.get(f"/api/projects/{pid}/statistics?format=csv")
        assert download.status_code == 200
        assert "attachment" in download.headers["content-disposition"]
        assert "player_id,name,team" in download.text
        assert client.get(f"/api/projects/{pid}/statistics?format=json").json()["players"][0]["player_id"] == "A1"
        assert client.get(f"/api/projects/{pid}/statistics?format=exe").status_code == 422


@pytest.mark.parametrize("changes", [
    {"type": "trim", "start_time": True, "end_time": 5},
    {"type": "classify", "kind": "invalid"},
    {"type": "export_enabled", "enabled": "yes"},
    {"type": "highlight_queue", "clips": [{"start_time": True, "end_time": 5}]},
    {"type": "touches", "point_stats": {"version": 1, "complete": "yes"}},
    {"type": "setup", "match_settings": {"starting_server": "A1", "starting_receiver": "A2"}},
    {"type": "export_settings", "settings": {"initial_score_a": 99}},
])
def test_malformed_commands_leave_revision_history_and_saved_state_unchanged(service, tmp_path, changes):
    project = add(service, new_project(service, tmp_path), 2, 5)
    identity = project["rallies"][0]["rally_id"]
    body = {"revision": project["revision"], "rally_id": identity, **deepcopy(changes)}
    if body["type"] == "highlight_queue":
        body["clips"][0]["rally_id"] = identity
    before = service.get_state(project["project_id"])
    with TestClient(create_app(service=service)) as client:
        response = client.post(f"/api/projects/{project['project_id']}/commands", json=body)
        assert response.status_code == 422, response.text
    assert service.get_state(project["project_id"]) == before
    saved = read_project(service.project_dir / f"{project['project_id']}.roundnet.json")
    assert saved["revision"] == before["revision"]
    assert saved["rallies"] == before["rallies"]
