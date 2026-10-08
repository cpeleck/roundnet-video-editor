"""Local browser media and worker integration, using generated source footage."""

from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from application.jobs import JobManager, TERMINAL_STATUSES, export_options, prepare_detection, prepare_export
from application.projects import ProjectService
from backend.app import create_app
from backend.media import MediaService
from models import Rally
from video.exporter import _prepare_presentation, generate_export_segments
from video.metadata import probe_video_metadata
from video.presentation import match_statistics


@pytest.fixture
def recording(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Generated video integration requires FFmpeg")
    path = tmp_path / "source.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
        "color=c=red:s=320x180:r=10:d=1", "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=10:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(command, check=True, capture_output=True, timeout=20)
    return path


def configured_project(recording, tmp_path):
    service = ProjectService(tmp_path / "projects")
    snapshot = service.create(str(recording))
    identity = snapshot["project_id"]
    service.command(identity, {"type": "setup", "match_settings": {}, "mode": "manual"})
    service.command(identity, {"type": "add", "start_time": 0, "end_time": .8})
    service.command(identity, {"type": "add", "start_time": 1, "end_time": 1.8})
    return service, identity


def wait_job(manager, identity, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(identity)
        if job["status"] in TERMINAL_STATUSES:
            return job
        time.sleep(.03)
    pytest.fail(f"Job did not finish: {manager.get(identity)}")


def test_highlight_queue_uses_source_seconds_without_changing_point_data(recording, tmp_path):
    service, identity = configured_project(recording, tmp_path)
    state = service.get_state(identity)
    before = deepcopy(state)
    first, second = state["rallies"]
    payload = prepare_export(state, {"mode": "highlights", "output_path": str(tmp_path / "reel.mp4"),
        "queue": [{"rally_id": second["rally_id"], "start_time": 1.2, "end_time": 1.6},
                  {"rally_id": first["rally_id"], "start_time": .1, "end_time": .5}]})
    segments = generate_export_segments(payload["rallies"], source_duration=2, segment_order=payload["segment_order"])
    assert [(segment.start, segment.end) for segment in segments] == [(1.2, 1.6), (.1, .5)]
    assert payload["presentation_rallies"] == before["rallies"]
    assert state == before
    with pytest.raises(ValueError, match="source"):
        prepare_export(state, {"output_path": str(recording)})
    with pytest.raises(ValueError, match="unique"):
        prepare_export(state, {"mode": "highlights", "output_path": str(tmp_path / "reel.mp4"),
            "queue": [{"rally_id": first["rally_id"]}, {"rally_id": first["rally_id"]}]})


def test_export_reel_process_renders_queue_in_user_order(recording, tmp_path):
    service, identity = configured_project(recording, tmp_path)
    first, second = service.get_state(identity)["rallies"]
    original = recording.read_bytes()
    manager = JobManager(service)
    try:
        job = manager.start(identity, "export", {"mode": "highlights", "output_path": str(tmp_path / "reel.mp4"),
            "prefer_hardware": False, "queue": [{"rally_id": second["rally_id"]}, {"rally_id": first["rally_id"]}]})
        result = wait_job(manager, job["id"])
        assert result["status"] == "succeeded", result
        assert result["progress"] == 100
        assert result["result"]["segment_count"] == 2
        assert result["result"]["output_duration_seconds"] == pytest.approx(6.6)
        capture = cv2.VideoCapture(result["result"]["output_path"])
        try:
            ok, first_frame = capture.read()
            assert ok
            assert first_frame[:, :, 0].mean() > first_frame[:, :, 2].mean() + 100  # blue clip first
            capture.set(cv2.CAP_PROP_POS_MSEC, 1100)
            ok, next_frame = capture.read()
            assert ok
            assert next_frame[:, :, 2].mean() > next_frame[:, :, 0].mean() + 100  # red clip second
        finally:
            capture.release()
        assert recording.read_bytes() == original
        assert service.get(identity).busy_job_id is None
    finally:
        manager.close()


def test_card_preview_matches_final_renderer_even_when_reel_trim_reorders_points(recording, tmp_path):
    service, identity = configured_project(recording, tmp_path)
    first, second = service.get_state(identity)["rallies"]
    service.command(identity, {"type": "classify", "rally_id": first["rally_id"], "kind": "ace"})
    service.command(identity, {"type": "classify", "rally_id": second["rally_id"], "kind": "double_fault"})
    state = service.get_state(identity)
    manager = JobManager(service)
    media = MediaService(service, manager)
    try:
        preview = media.card(identity)
        payload = prepare_export(state, {"mode": "highlights", "output_path": str(tmp_path / "render.mp4"),
            "options": {"include_stats": True, "stats_duration": 3},
            "queue": [{"rally_id": second["rally_id"], "start_time": 0, "end_time": .3}]})
        final_assets = tmp_path / "final-assets"
        final_assets.mkdir()
        segments = generate_export_segments(payload["rallies"], source_duration=2, segment_order=payload["segment_order"])
        paths, filters = _prepare_presentation(final_assets, payload["rallies"], segments,
            probe_video_metadata(recording), payload["options"], payload["presentation_rallies"])
        assert paths[-1].read_bytes() == preview.read_bytes()
        assert filters["stats_duration"] == 3
        assert match_statistics(payload["presentation_rallies"], payload["options"]) == match_statistics(state["rallies"], payload["options"])
    finally:
        manager.close()


def test_detection_process_commits_recoverable_analysis(recording, tmp_path):
    service, identity = configured_project(recording, tmp_path)
    before = service.get_state(identity)
    manager = JobManager(service)
    try:
        job = manager.start(identity, "detection", {"expected_revision": before["revision"],
            "settings": {"analysis_fps": 2}})
        result = wait_job(manager, job["id"])
        assert result["status"] == "succeeded", result
        state = service.get_state(identity)
        assert state["revision"] == before["revision"] + 1
        assert state["stage"] == "review"
        assert state["signals"]["timestamps"]
        assert result["result"]["project"]["revision"] == state["revision"]
        service.command(identity, {"type": "undo"})
        assert service.get_state(identity)["rallies"] == before["rallies"]
    finally:
        manager.close()


def test_queued_cancellation_preserves_project_and_releases_worker_lock(recording, tmp_path):
    service, identity = configured_project(recording, tmp_path)
    before = service.get_state(identity)
    manager = JobManager(service, max_workers=1)
    manager._slots.acquire()
    try:
        job = manager.start(identity, "detection")
        assert job["status"] == "queued"
        manager.cancel(job["id"])
        result = wait_job(manager, job["id"])
        assert result["status"] == "cancelled"
        assert service.get_state(identity) == before
        assert service.get(identity).busy_job_id is None
    finally:
        manager._slots.release()
        manager.close()


def test_accepted_detection_cancel_wins_before_result_commit(recording, tmp_path):
    service, identity = configured_project(recording, tmp_path)
    before = service.get_state(identity)
    result_ready, allow_commit = threading.Event(), threading.Event()

    class DelayedCommitManager(JobManager):
        def _finish(self, identity, status, **details):
            if status == "succeeded":
                result_ready.set()
                allow_commit.wait(timeout=10)
            super()._finish(identity, status, **details)

    manager = DelayedCommitManager(service)
    try:
        job = manager.start(identity, "detection", {"settings": {"analysis_fps": 2}})
        assert result_ready.wait(timeout=10)
        manager.cancel(job["id"])
        allow_commit.set()
        assert wait_job(manager, job["id"])["status"] == "cancelled"
        assert service.get_state(identity) == before
        assert service.get(identity).busy_job_id is None
    finally:
        allow_commit.set()
        manager.close()


def test_worker_failure_is_explicit_and_preserves_review(recording, tmp_path):
    service, identity = configured_project(recording, tmp_path)
    before = service.get_state(identity)
    manager = JobManager(service)
    try:
        # Preview worker probes invalid media in a separate process; no point edit is committed.
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"invalid source")
        job = manager.start(identity, "preview", {"video_path": str(broken), "output_path": str(tmp_path / "proxy.mp4"), "duration": 1})
        result = wait_job(manager, job["id"])
        assert result["status"] == "failed"
        assert "Preview conversion failed" in result["error"]
        assert service.get_state(identity) == before
        assert not (tmp_path / "proxy.mp4").exists()
    finally:
        manager.close()


def test_local_media_ranges_thumbnail_and_preview_proxy(recording, tmp_path):
    service, identity = configured_project(recording, tmp_path)
    app = create_app(service=service)
    with TestClient(app) as client:
        ranged = client.get(f"/api/projects/{identity}/media", headers={"Range": "bytes=0-31"})
        assert ranged.status_code == 206
        assert ranged.content == recording.read_bytes()[:32]
        assert ranged.headers["content-range"].startswith("bytes 0-31/")
        thumbnail = client.get(f"/api/projects/{identity}/thumbnail?time=1.2&width=160")
        assert thumbnail.status_code == 200
        image = cv2.imdecode(np.frombuffer(thumbnail.content, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (90, 160)
        assert image[:, :, 0].mean() > image[:, :, 2].mean() + 100
        assert client.get(f"/api/projects/{identity}/thumbnail?time=10").status_code == 422
        info = client.get(f"/api/projects/{identity}/media-info?proxy=true").json()
        assert info["source_timestamps"] is True
        proxy_job = wait_job(app.state.jobs, info["job"]["id"])
        assert proxy_job["status"] == "succeeded", proxy_job
        assert client.get(f"/api/projects/{identity}/media-info?proxy=true").json()["status"] == "ready"
        proxy = client.get(f"/api/projects/{identity}/media?proxy=true", headers={"Range": "bytes=0-31"})
        assert proxy.status_code == 206
        assert probe_video_metadata(proxy_job["result"]["output_path"]).duration == pytest.approx(2, abs=.1)
        assert client.get("/api/jobs/missing").status_code == 404


def test_export_order_validation_is_unambiguous():
    rallies = [Rally(3, 4, rally_id="late"), Rally(0, 1, rally_id="early")]
    assert [segment.start for segment in generate_export_segments(rallies)] == [0, 3]
    assert [segment.start for segment in generate_export_segments(rallies, segment_order=["late", "early"])] == [3, 0]
    with pytest.raises(Exception, match="duplicate"):
        generate_export_segments(rallies, segment_order=["late", "late"])
    with pytest.raises(Exception, match="unknown"):
        generate_export_segments(rallies, segment_order=["missing"])


def test_render_settings_cannot_override_canonical_match_setup():
    options = export_options({"match_settings": {"team_a": "Original team", "initial_score_a": 3},
        "export_settings": {"team_a": "Stale team", "initial_score_a": 99, "starting_server": "B1", "scoreboard": True}})
    assert options["team_a"] == "Original team"
    assert options["initial_score_a"] == 3
    assert options["starting_server"] == "A1"
    assert options["scoreboard"] is True
    with pytest.raises(ValueError, match="Unknown presentation option"):
        export_options({}, {"initial_score_a": 99})
    with pytest.raises(ValueError, match="object"):
        export_options({}, [])
    with pytest.raises(ValueError, match="object"):
        prepare_detection({"settings": {}}, {"settings": []})


def test_export_protects_saved_project_and_rejects_boolean_trim_times(recording, tmp_path):
    service, identity = configured_project(recording, tmp_path)
    state = service.get_state(identity)
    project_path = tmp_path / "project-with-video-extension.mp4"
    state["project_path"] = str(project_path)
    with pytest.raises(ValueError, match="saved project"):
        prepare_export(state, {"output_path": str(project_path)})
    with pytest.raises(ValueError, match="finite numbers"):
        prepare_export(state, {"mode": "highlights", "output_path": str(tmp_path / "reel.mp4"),
            "queue": [{"rally_id": state["rallies"][0]["rally_id"], "start_time": False}]})


def test_branding_preview_serves_only_the_saved_valid_png(recording, tmp_path):
    service, identity = configured_project(recording, tmp_path)
    logo = tmp_path / "saved-logo.png"
    other = tmp_path / "other-logo.png"
    cv2.imwrite(str(logo), np.full((12, 16, 3), (30, 180, 90), np.uint8))
    cv2.imwrite(str(other), np.full((12, 16, 3), (230, 30, 20), np.uint8))
    app = create_app(service=service)
    with TestClient(app) as client:
        assert client.get(f"/api/projects/{identity}/branding", params={"path": str(other)}).status_code == 404
        service.command(identity, {"type": "export_settings", "settings": {"overlay_path": str(logo)}})
        result = client.get(f"/api/projects/{identity}/branding", params={"path": str(other)})
        assert result.status_code == 200
        assert result.content == logo.read_bytes()
        assert result.headers["content-type"] == "image/png"
        logo.write_bytes(b"invalid replacement")
        assert client.get(f"/api/projects/{identity}/branding").status_code == 422


def test_failed_proxy_restarts_only_after_explicit_retry(recording, tmp_path):
    service, identity = configured_project(recording, tmp_path)
    app = create_app(service=service)

    class PreviewJobs:
        def __init__(self):
            self.jobs = {}

        def start(self, project_id, kind, payload):
            identity = str(len(self.jobs) + 1)
            job = {"id": identity, "kind": kind, "status": "queued", "progress": 0, "error": None}
            self.jobs[identity] = job
            return deepcopy(job)

        def get(self, identity):
            return deepcopy(self.jobs[identity])

    jobs = PreviewJobs()
    app.state.media.jobs = jobs
    with TestClient(app) as client:
        first = client.get(f"/api/projects/{identity}/media-info?proxy=true").json()
        jobs.jobs[first["job"]["id"]].update(status="failed", error="Encoder unavailable")
        failed = client.get(f"/api/projects/{identity}/media-info?proxy=true").json()
        assert failed["status"] == "failed"
        assert len(jobs.jobs) == 1
        retried = client.get(f"/api/projects/{identity}/media-info?proxy=true&retry=true").json()
        assert retried["status"] == "queued"
        assert retried["job"]["id"] != first["job"]["id"]
        assert len(jobs.jobs) == 2
        client.get(f"/api/projects/{identity}/media-info?proxy=true&retry=true")
        assert len(jobs.jobs) == 2


def test_export_appends_current_card_with_rpr_by_default(recording, tmp_path):
    from video.exporter import FFmpegExporter
    from video.presentation import render_player_end_card_image
    service, identity = configured_project(recording, tmp_path)
    first, second = service.get_state(identity)['rallies']
    for rally, kind, details in (
        (first, 'defensive_hold', {'version': 1, 'possessions': [
            {'first_touch_id': 'B1', 'hitter_id': 'B1'}, {'first_touch_id': 'A1', 'hitter_id': 'A1'}]}),
        (second, 'sideout', {'version': 1, 'possessions': []}),
    ):
        service.command(identity, {'type': 'classify', 'rally_id': rally['rally_id'], 'kind': kind, 'quick_log': details})
    state = service.get_state(identity)
    payload = prepare_export(state, {'output_path': str(tmp_path / 'automatic-card.mp4')})
    assert payload['options']['include_stats'] is True
    summary = match_statistics(payload['presentation_rallies'], payload['options'])
    assert summary['player_statistics']['players'][0]['rpr']['overall'] == pytest.approx(63)
    result = FFmpegExporter().export(recording, payload['output_path'], payload['rallies'],
        prefer_hardware=False, export_options=payload['options'], presentation_rallies=payload['presentation_rallies'])
    assert result.output_duration_seconds == pytest.approx(6.6)
    capture = cv2.VideoCapture(str(result.output_path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, 2000)
        ok, frame = capture.read()
        assert ok
        expected = render_player_end_card_image(320, 180, summary)
        assert np.mean(np.abs(frame.astype(float) - expected.astype(float))) < 6
    finally:
        capture.release()
    # Disabling the automatic card remains an explicit export preference.
    assert not export_options(state, {'include_stats': False})['include_stats']
