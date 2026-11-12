"""Synthetic benchmark validation; these fixtures are not real-game accuracy."""

from copy import deepcopy
import json
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from application.projects import ProjectService
from backend.app import create_app
from models.rally import Rally
from training.benchmark import correction_counts, run_benchmark
from training.calibration import _aggregate_events, evaluate_intervals


def detected_project(service, tmp_path, index):
    source = tmp_path / f"source-{index}.mp4"
    source.write_bytes(f"synthetic video {index}".encode())
    project = service.create(str(source))
    identity = project["project_id"]
    service.command(identity, {"type": "setup", "match_settings": {}, "mode": "automatic"})
    times = np.arange(0, 20, .125)
    playing = ((times >= 3) & (times < 7)) | ((times >= 11) & (times < 15))
    retrieval = (times >= 17) & (times < 19)
    proposals = [Rally(3, 7), Rally(11, 15), Rally(17, 19)]
    service.lock_job(identity, "fixture")
    service.finish_job(identity, "fixture", {
        "rallies": [r.to_dict() for r in proposals], "initial_predictions": [r.to_dict() for r in proposals],
        "signals": {"timestamps": times.tolist(), "motion": np.where(playing | retrieval, .9, .05).tolist(),
                    "motion_spread": np.where(playing, .9, .02).tolist(),
                    "rally_score": np.where(playing | retrieval, .9, .05).tolist()},
        "analysis_settings": {"pre_roll": 0, "post_roll": 0, "min_rally_duration": .5,
                              "merge_gap": .1, "retrieval_filter_enabled": False},
        "detection_run": {"detection_elapsed_seconds": 2 + index, "machine": "synthetic-test"},
    })
    service.command(identity, {"type": "reject", "rally_id": proposals[2].rally_id, "rejected": True})
    # A real point excluded from export still belongs in benchmark truth.
    return service.command(identity, {"type": "export_enabled", "rally_id": proposals[1].rally_id, "enabled": False})


@pytest.fixture
def service(tmp_path):
    return ProjectService(tmp_path / "data", metadata_probe=lambda path: {
        "duration_seconds": 20, "width": 640, "height": 360, "fps": 30})


def capture(client, project, **overrides):
    return client.post("/api/admin/benchmarks/recordings", json={
        "project_id": project["project_id"], "expected_revision": project["revision"],
        "full_review_confirmed": True, **overrides})


def test_boundary_tolerance_requires_both_edges_and_one_to_one():
    metric = evaluate_intervals([(1, 5), (1, 5), (10, 15), (20, 29)],
                                [(2, 6), (10, 16.01), (20, 30)], 40,
                                boundary_tolerance_seconds=1)
    assert (metric["true_positives"], metric["false_positives"], metric["false_negatives"]) == (2, 2, 1)
    assert metric["precision"] == .5
    assert metric["recall"] == pytest.approx(2 / 3)
    assert metric["median_boundary_error_seconds"] == 1
    assert metric["matching_iou"] is None
    # A long overlapping clip fails the strict edges rule despite a high IoU.
    assert evaluate_intervals([(0, 20)], [(2, 18)], 30, boundary_tolerance_seconds=1)["true_positives"] == 0
    assert evaluate_intervals([], [(2, 3)], 10, boundary_tolerance_seconds=1)["median_boundary_error_seconds"] is None
    for invalid in (-1, float("nan"), True, "1"):
        with pytest.raises(ValueError, match="boundary_tolerance_seconds"):
            evaluate_intervals([], [], 20, boundary_tolerance_seconds=invalid)


def test_pooled_boundary_median_is_not_a_median_of_fold_medians():
    first = evaluate_intervals([(1, 3)], [(1.9, 3.9)], 20, boundary_tolerance_seconds=1)
    second = evaluate_intervals([(1, 3), (5, 7), (10, 12)],
                                [(1.1, 3.1), (5.1, 7.1), (10.1, 12.1)], 20,
                                boundary_tolerance_seconds=1)
    pooled = _aggregate_events([first, second])
    assert pooled["median_boundary_error_seconds"] == pytest.approx(.1)
    assert pooled["precision"] == pooled["recall"] == 1


def test_capture_confirmation_revision_dedup_and_export_omissions(service, tmp_path):
    project = detected_project(service, tmp_path, 1)
    before = deepcopy(service.get_state(project["project_id"]))
    with TestClient(create_app(service=service)) as client:
        assert client.get("/api/session").json()["role"] == "admin"
        assert capture(client, project, full_review_confirmed=False).status_code == 422
        assert capture(client, project, expected_revision=0).status_code == 409
        assert capture(client, project).status_code == 200
        assert capture(client, project).status_code == 200
        overview = client.get("/api/admin/benchmarks").json()
        assert len(overview["recordings"]) == 1
        row = overview["recordings"][0]
        assert row["rally_count"] == 2
        assert row["corrections"] == {"added": 0, "removed": 1, "trimmed": 0, "total": 1}
        assert service.get_state(project["project_id"]) == before
    # Dataset survives a service restart and retains frozen labels.
    with TestClient(create_app(service=service)) as restarted:
        assert len(restarted.get("/api/admin/benchmarks").json()["recordings"]) == 1
    metadata = json.loads(next((service.data_dir / "benchmarks/dataset").glob("*/corrections.json")).read_text())
    assert len(metadata["final_enabled_intervals"]) == 2
    assert metadata["final_disabled_intervals"] == []
    assert metadata["annotation_completeness_confirmed"] is True


def test_remote_paired_editor_cannot_use_admin_or_spoof_local_host(service, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUNDNET_PHONE_HOST", "192.168.1.42")
    monkeypatch.setenv("ROUNDNET_PHONE_TOKEN", "pair-token")
    app = create_app(service=service)
    with TestClient(app, base_url="http://192.168.1.42:5173",
                    headers={"Origin": "http://192.168.1.42:5173"}) as phone:
        app.state.jobs._jobs["private-benchmark"] = {"id": "private-benchmark", "kind": "benchmark", "status": "succeeded", "result": {"private": True}}
        assert phone.post("/api/session", json={"token": "pair-token"}).status_code == 200
        assert phone.get("/api/session").json()["role"] == "editor"
        assert phone.get("/api/admin/benchmarks").status_code == 403
        assert phone.post("/api/admin/benchmarks/run", json={"recording_ids": []}).status_code == 403
        assert phone.get("/api/admin/benchmarks", headers={"Host": "localhost"}).status_code == 403
        assert phone.get("/api/jobs/private-benchmark").status_code == 403
        assert phone.post("/api/jobs/private-benchmark/cancel", json={}).status_code == 403
        assert phone.get("/api/jobs/private-benchmark/output").status_code == 403
    with TestClient(app, client=("192.168.1.20", 2345), base_url="http://localhost:8000",
                    cookies={"roundnet_pair": "pair-token"}) as remote:
        assert remote.get("/api/session").json()["role"] == "editor"
        assert remote.get("/api/admin/benchmarks").status_code == 403


def test_correction_counts_track_net_intervals_not_scoring_or_export(service, tmp_path):
    project = detected_project(service, tmp_path, 0)
    identity = project["project_id"]
    service.command(identity, {"type": "trim", "rally_id": project["rallies"][0]["rally_id"],
                               "start_time": 3.25, "end_time": 7.25})
    service.command(identity, {"type": "add", "start_time": 8, "end_time": 9})
    assert correction_counts(service.get_state(identity)) == {"added": 1, "removed": 1, "trimmed": 1, "total": 3}
    assert correction_counts({"signals": None}) is None


def test_background_benchmark_and_saved_download_survive_restart(service, tmp_path):
    projects = [detected_project(service, tmp_path, i) for i in range(8)]
    with TestClient(create_app(service=service)) as client:
        ids = [capture(client, project).json()["id"] for project in projects]
        assert client.post("/api/admin/benchmarks/run", json={"recording_ids": ids[:2]}).status_code == 422
        assert client.post("/api/admin/benchmarks/run", json={"recording_ids": [ids[0]] * 3}).status_code == 422
        assert client.post("/api/admin/benchmarks/run", json={"recording_ids": ["../../"] * 3}).status_code == 422
        response = client.post("/api/admin/benchmarks/run", json={"recording_ids": ids})
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(.03)
        assert job["status"] == "succeeded", job
        report = job["result"]
        assert report["matching"]["boundary_tolerance_seconds"] == 1
        assert report["recording_count"] == 8
        assert report["motion_events"]["false_positives"] == 8
        assert report["learned_events"]["false_positives"] == 0
        assert report["learned_events"]["median_boundary_error_seconds"] == 0
        assert report["operations"]["median_corrections_per_recording"] == 1
        assert report["operations"]["estimated_25_minute_seconds"] == 412.5
        assert report["operations"]["median_measured_25_minute_seconds"] is None
        for fold in report["folds"]:
            assert fold["held_out_source_id"] not in fold["training_source_ids"]
            assert len(fold["training_source_ids"]) == 7
            assert len(fold["reference_intervals"]) == 2
            assert len(fold["predicted_intervals"]["motion"]) == 3
        report_id = report["report_id"]
        assert client.get(f"/api/admin/benchmarks/reports/{report_id}/download").json() == report
        assert client.get(f"/api/jobs/{job_id}/output").status_code == 409
    with TestClient(create_app(service=service)) as restarted:
        assert restarted.get("/api/admin/benchmarks").json()["reports"][0]["id"] == report_id
        assert restarted.get(f"/api/admin/benchmarks/reports/{report_id}").json() == report


def test_benchmark_cancellation_preserves_projects_and_does_not_publish(service, tmp_path):
    projects = [detected_project(service, tmp_path, i) for i in range(3)]
    before = [service.get_state(project["project_id"]) for project in projects]
    app = create_app(service=service)
    with TestClient(app) as client:
        ids = [capture(client, project).json()["id"] for project in projects]
        app.state.jobs._slots.acquire()
        app.state.jobs._slots.acquire()
        try:
            job = client.post("/api/admin/benchmarks/run", json={"recording_ids": ids}).json()
            assert client.post("/api/admin/benchmarks/run", json={"recording_ids": ids}).status_code == 422
            client.post(f"/api/jobs/{job['id']}/cancel")
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if client.get(f"/api/jobs/{job['id']}").json()["status"] == "cancelled":
                    break
                time.sleep(.03)
            assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "cancelled"
        finally:
            app.state.jobs._slots.release()
            app.state.jobs._slots.release()
        assert not client.get("/api/admin/benchmarks").json()["reports"]
        assert [service.get_state(project["project_id"]) for project in projects] == before


def test_measured_25_minute_time_and_estimate_are_separate(tmp_path, monkeypatch):
    import training.benchmark as benchmark
    paths = []
    folds = []
    for i, (duration, elapsed) in enumerate([(1500, 30), (750, 20), (3000, None)]):
        path = tmp_path / f"metadata-{i}.json"
        path.write_text(json.dumps({"source": {"id": str(i)}, "created_utc": "2026-01-01",
            "benchmark": {"detection_run": {"detection_elapsed_seconds": elapsed},
                          "corrections": {"total": i + 1}}}))
        paths.append(path)
        folds.append({"held_out_source_id": str(i), "source_title": str(i), "duration_seconds": duration})
    monkeypatch.setattr(benchmark, "train_from_feedback", lambda *args, **kwargs: {
        "folds": folds, "recording_count": 3, "warnings": []})
    operations = run_benchmark(paths, tmp_path / "unused.json")["operations"]
    assert operations["median_measured_25_minute_seconds"] == 30
    assert operations["estimated_25_minute_seconds"] == 35
    assert operations["timed_recording_count"] == 2
    assert operations["median_corrections_per_recording"] == 2
