"""Correction-label downloads retain training provenance and unknown footage."""

from io import BytesIO
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient
import numpy as np

from application.projects import ProjectService
from backend.app import create_app
from models.rally import Rally


def _project(tmp_path, *, detected=True):
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"source recording, not a rendered montage")
    service = ProjectService(tmp_path / "data", metadata_probe=lambda path: {
        "path": path, "duration_seconds": 12, "width": 640, "height": 360, "fps": 30})
    project = service.create(str(source))
    identity = project["project_id"]
    project = service.command(identity, {"type": "setup", "match_settings": {}, "mode": "manual"})
    if detected:
        rallies = [Rally(.5, 2.5), Rally(4, 5), Rally(6, 8), Rally(9, 11)]
        service.lock_job(identity, "detector")
        project = service.finish_job(identity, "detector", {
            "rallies": [r.to_dict() for r in rallies], "initial_predictions": [r.to_dict() for r in rallies],
            "signals": {"timestamps": np.arange(0, 12, .5).tolist(), "motion_scores": [.5] * 24},
            "analysis_settings": {"pre_roll": 0, "post_roll": 0, "rally_threshold": .55},
            "analysis_roi": [.1, .1, .8, .8], "analysis_court_context": {"net": [.5, .5], "serving_zones": []},
        })
        first, rejected, omitted, _ = [r["rally_id"] for r in project["rallies"]]
        service.command(identity, {"type": "classify", "rally_id": first, "kind": "ace"})
        service.command(identity, {"type": "reject", "rally_id": rejected, "rejected": True})
        service.command(identity, {"type": "export_enabled", "rally_id": omitted, "enabled": False})
        # Statistical confirmation and stars must not certify training coverage.
        service.command(identity, {"type": "full_match", "confirmed": True})
        service.command(identity, {"type": "star", "rally_id": first, "starred": True})
        service.command(identity, {"type": "settings", "settings": {"rally_threshold": .8}})
    return service, service.snapshot(identity), source


def _download(client, project, **options):
    endpoint = f"/api/projects/{project['project_id']}/correction-labels"
    response = client.post(endpoint, json={"expected_revision": project["revision"], **options})
    assert response.status_code == 200, response.text
    result = response.json()
    download = client.get(result["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    return result, ZipFile(BytesIO(download.content))


def test_download_keeps_rejections_export_omissions_and_unseen_footage_distinct(tmp_path):
    service, project, source = _project(tmp_path)
    with TestClient(create_app(service=service)) as client:
        result, bundle = _download(client, project)
        assert result["features_available"] and result["sample_count"] == 24
        metadata = json.loads(bundle.read("corrections.json"))
        assert metadata["annotation_completeness"] == "unknown"
        assert not metadata["annotation_completeness_confirmed"]
        assert metadata["detection_settings"]["rally_threshold"] == .55
        assert metadata["roi"] == [.1, .1, .8, .8]
        assert len(metadata["initial_predictions"]) == 4
        assert len(metadata["rejected_detections"]) == 1
        assert metadata["final_disabled_intervals"] == [{"start": 6, "end": 8}]
        assert metadata["source"]["path"] == str(source)
        features = np.load(BytesIO(bundle.read("corrections.npz")), allow_pickle=False)
        labels = features["labels"]
        assert labels[0] == -1  # Unseen background stays unknown.
        assert labels[3] == 1   # Reviewed play supplies a positive example.
        assert labels[9] == 0   # Explicit false detection supplies a negative.
        assert labels[14] == -1 # Export omission supplies no training judgment.
        assert labels[20] == -1 # Unreviewed detector proposal stays unknown.
        assert service.snapshot(project["project_id"])["revision"] == project["revision"]


def test_explicit_recording_review_confirms_background_only_for_downloaded_snapshot(tmp_path):
    service, project, _ = _project(tmp_path)
    with TestClient(create_app(service=service)) as client:
        result, bundle = _download(client, project, full_review_confirmed=True)
        assert result["full_review_confirmed"]
        metadata = json.loads(bundle.read("corrections.json"))
        assert metadata["annotation_completeness_confirmed"]
        assert metadata["annotation_completeness"] == "all_rallies"
        features = np.load(BytesIO(bundle.read("corrections.npz")), allow_pickle=False)
        assert features["labels"][0] == 0
        assert features["labels"][14] == -1
        assert service.snapshot(project["project_id"])["revision"] == project["revision"]


def test_manual_project_download_and_validation(tmp_path):
    service, project, source = _project(tmp_path, detected=False)
    with TestClient(create_app(service=service)) as client:
        result, bundle = _download(client, project)
        assert not result["features_available"]
        assert bundle.namelist() == ["corrections.json"]
        endpoint = f"/api/projects/{project['project_id']}/correction-labels"
        assert client.post(endpoint, json={"full_review_confirmed": "true"}).status_code == 422
        assert client.post(endpoint, json={"expected_revision": -1}).status_code == 409
        assert client.get(endpoint + "/missing-download").status_code == 404
        source.unlink()
        assert client.post(endpoint, json={}).status_code == 409


def test_training_confirmation_persists_and_clip_edits_clear_it(tmp_path):
    service, project, _ = _project(tmp_path)
    identity = project["project_id"]
    confirmed = service.command(identity, {"type": "complete_review", "confirmed": True})
    assert confirmed["complete"] and confirmed["complete_review_confirmed"]
    restarted = ProjectService(service.data_dir, metadata_probe=service.metadata_probe)
    recovered = restarted.snapshot(identity)
    assert recovered["complete"] and recovered["complete_review_confirmed"]
    edited = restarted.command(identity, {"type": "trim", "rally_id": project["rallies"][0]["rally_id"],
                                          "start_time": .6, "end_time": 2.5})
    assert not edited["complete"] and not edited["complete_review_confirmed"]
