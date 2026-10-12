"""Long native file selections can complete or cancel without a deadline."""

import importlib
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from application.projects import ProjectService


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("kind", ["video", "project", "destination"])
def test_picker_waits_without_deadline(platform, kind, monkeypatch, tmp_path):
    module = importlib.import_module("backend.app")
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform=platform, executable="python"))
    path = "/recordings/large match.mov"

    def complete_long_selection(command, **kwargs):
        assert kwargs.get("timeout") is None
        assert command[0] == ("osascript" if platform == "darwin" else "python")
        return SimpleNamespace(returncode=0, stdout=path + "\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", complete_long_selection)
    with TestClient(module.create_app(service=ProjectService(tmp_path / "projects"))) as client:
        response = client.post("/api/files/pick", json={"kind": kind})
        assert response.status_code == 200
        assert response.json() == {"path": path}


@pytest.mark.parametrize("platform,code,stderr", [
    ("darwin", 1, "User canceled. (-128)"), ("linux", 0, ""),
])
def test_picker_still_allows_cancellation(platform, code, stderr, monkeypatch):
    module = importlib.import_module("backend.app")
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform=platform, executable="python"))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=code, stdout="\n", stderr=stderr))
    assert module._pick_file("video") is None


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_picker_reports_native_errors(platform, monkeypatch):
    module = importlib.import_module("backend.app")
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform=platform, executable="python"))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=1, stdout="", stderr="GUI unavailable"))
    with pytest.raises(module.ProjectError) as error:
        module._pick_file("video")
    assert error.value.code == "picker_unavailable"
