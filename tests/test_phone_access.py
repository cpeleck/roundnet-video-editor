"""Pairing, disk-backed imports, and bounded phone file capabilities."""

import shutil
import subprocess
import time
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from starlette.requests import ClientDisconnect

from application.projects import ProjectService
from backend.app import create_app
from backend.phone import MAX_UPLOAD
from run_web import _phone_address

HOST = "192.168.1.42"
ORIGIN = f"http://{HOST}:5173"
TOKEN = "private-test-pairing-token"


@pytest.fixture
def phone(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUNDNET_PHONE_HOST", HOST)
    monkeypatch.setenv("ROUNDNET_PHONE_TOKEN", TOKEN)
    service = ProjectService(tmp_path / "projects")
    app = create_app(service=service)
    with TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN}) as client:
        yield client, service, app


@pytest.fixture
def video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for a real phone import/export")
    path = tmp_path / "phone.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=160x90:r=10:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True)
    return path.read_bytes()


def connect(client):
    response = client.post("/api/session", json={"token": TOKEN})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie


def test_phone_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("ROUNDNET_PHONE_TOKEN", raising=False)
    monkeypatch.delenv("ROUNDNET_PHONE_HOST", raising=False)
    with TestClient(create_app(data_dir=tmp_path)) as client:
        assert client.get("/api/projects", headers={"Host": HOST}).status_code == 403
        assert client.get("/api/projects").status_code == 200


def test_pairing_required_for_media_and_commands_even_with_local_host(phone):
    client, _, _ = phone
    assert client.get("/api/session").json()["paired"] is False
    for path in ("/api/projects", "/api/projects/id/media", "/api/jobs/id/output"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Host": "localhost"}).status_code == 401
    assert client.post("/api/session", json={"token": "wrong"}).status_code == 401
    assert client.post("/api/session", json={"token": "💥"}).status_code == 401
    assert client.post("/api/session", json={"token": TOKEN}, headers={"Origin": "https://other.example"}).status_code == 403
    connect(client)
    assert client.get("/api/projects").status_code == 200
    assert client.get("/api/session").json()["paired"] is True
    assert client.get("/api/session").headers["cache-control"] == "no-store"
    assert client.get("/api/projects", headers={"Host": "other.example"}).status_code == 403


def test_upload_validation_and_disconnect_cleanup(phone):
    client, service, _ = phone
    connect(client)
    assert client.post("/api/recordings?filename=oops.exe", content=b"x").status_code == 422
    assert client.post("/api/recordings?filename=oops.mov", content=b"not a video").status_code == 422
    assert not list((service.data_dir / "recordings").iterdir())
    assert client.post("/api/recordings?filename=large.mov", content=b"x",
                       headers={"Content-Length": str(MAX_UPLOAD + 1)}).status_code == 413

    def disconnect():
        yield b"partial recording"
        raise ClientDisconnect()

    with pytest.raises(ClientDisconnect):
        client.post("/api/recordings?filename=cancelled.mov", content=disconnect())
    assert not list((service.data_dir / "recordings").iterdir())


def test_real_phone_import_trim_export_and_download(phone, video, tmp_path):
    client, service, app = phone
    connect(client)
    upload = client.post("/api/recordings?filename=../../game.MOV", content=video)
    assert upload.status_code == 200, upload.text
    source = Path(upload.json()["path"])
    assert source.parent == service.data_dir / "recordings"
    assert source.suffix == ".mov" and source.read_bytes() == video
    assert not list(source.parent.glob("*.part"))
    assert client.post("/api/projects", json={"video_path": str(tmp_path / "phone.mov")}).status_code == 403
    assert client.post("/api/projects/open", json={"path": "/private/file"}).status_code == 403
    assert client.post("/api/files/pick", json={"kind": "video"}).status_code == 403
    project = client.post("/api/projects", json={"video_path": str(source), "title": "Phone match"}).json()
    identity = project["project_id"]
    endpoint = f"/api/projects/{identity}"
    assert client.post(endpoint + "/commands", json={"type": "save_as", "path": "/private/file"}).status_code == 403
    assert client.post(endpoint + "/commands", json={"type": "settings", "settings": {"ffmpeg_path": "/bin/sh"}}).status_code == 403
    assert client.post(endpoint + "/detect", json={"settings": {"ffmpeg_path": "/bin/sh"}}).status_code == 403
    assert client.post(endpoint + "/commands", json={"type": "setup", "match_settings": {}, "mode": "manual"}).status_code == 200
    assert client.post(endpoint + "/commands", json={"type": "add", "start_time": 0, "end_time": .8}).status_code == 200
    media = client.get(endpoint + "/media", headers={"Range": "bytes=0-99"})
    assert media.status_code == 206 and len(media.content) == 100
    output = tmp_path / "must-not-be-overwritten.mp4"
    output.write_bytes(b"original")
    response = client.post(endpoint + "/export", json={"output_path": str(output), "overwrite": True,
        "prefer_hardware": False, "options": {"include_stats": False, "scoreboard": False}})
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = app.state.jobs.get(job_id)
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(.05)
    assert job["status"] == "succeeded", job
    assert Path(job["result"]["output_path"]).parent == service.data_dir / "exports"
    downloaded = client.get(f"/api/jobs/{job_id}/output")
    assert downloaded.status_code == 200 and downloaded.headers["content-type"] == "video/mp4"
    assert "attachment" in downloaded.headers["content-disposition"]
    assert output.read_bytes() == b"original" and source.read_bytes() == video


def test_pairing_expires_when_launcher_token_changes(phone, monkeypatch):
    client, service, _ = phone
    connect(client)
    cookies = dict(client.cookies)
    monkeypatch.setenv("ROUNDNET_PHONE_TOKEN", "new-launcher-token")
    with TestClient(create_app(service=service), base_url=ORIGIN, headers={"Origin": ORIGIN}, cookies=cookies) as restarted:
        assert restarted.get("/api/projects").status_code == 401


@pytest.mark.parametrize("address", ["127.0.0.1", "0.0.0.0", "8.8.8.8", "169.254.1.2", "example.com"])
def test_launcher_rejects_non_lan_addresses(address):
    with pytest.raises(RuntimeError):
        _phone_address(address)


def test_launcher_accepts_private_wifi_address():
    assert _phone_address(HOST) == HOST
