"""Local FastAPI service for the focused browser editor."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import os
import subprocess
import sys
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from application.projects import ProjectError, ProjectService, STAGES
from config import DetectionSettings
from models.match_flow import CLASSIFICATIONS
from models.point_stats import RESULTS
from models.statistics import export_statistics
from video.errors import VideoIOError


def _pick_file(kind):
    """Use the desktop host's native picker; no GUI framework enters the API."""
    if kind not in ("video", "project", "destination"):
        raise ValueError("Choose a video, project, or destination picker")
    if sys.platform == "darwin":
        script = ('POSIX path of (choose file name with prompt "Save exported video" default name "roundnet.mp4")'
                  if kind == "destination" else
                  'POSIX path of (choose file with prompt "Choose ' + ("recording" if kind == "video" else "Roundnet project") + '")')
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=300)
        if result.returncode:
            if "(-128)" in result.stderr:
                return None
            raise ProjectError("The native file picker could not open. Enter an absolute path instead.", code="picker_unavailable", status=503)
        return result.stdout.strip() or None
    script = (
        "import tkinter as tk; from tkinter import filedialog; "
        "root=tk.Tk(); root.withdraw(); "
        + ("path=filedialog.asksaveasfilename(defaultextension='.mp4',initialfile='roundnet.mp4'); "
           if kind == "destination" else "path=filedialog.askopenfilename(); ")
        + "print(path); root.destroy()"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise ProjectError("The native file picker is unavailable. Enter an absolute path instead.", code="picker_unavailable", status=503)
    return result.stdout.strip() or None


def create_app(data_dir=None, service=None):
    projects = service or ProjectService(data_dir)
    frontend_port = int(os.environ.get("ROUNDNET_FRONTEND_PORT", "5173"))
    api_port = int(os.environ.get("ROUNDNET_API_PORT", "8000"))
    trusted_origins = {f"http://{host}:{port}" for host in ("127.0.0.1", "localhost")
                       for port in (frontend_port, api_port)}

    @asynccontextmanager
    async def lifespan(app):
        yield
        if hasattr(app.state, "jobs"):
            app.state.jobs.close()

    app = FastAPI(title="Roundnet Video Editor", lifespan=lifespan)
    app.state.projects = projects
    app.add_middleware(CORSMiddleware, allow_origins=sorted(trusted_origins),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Range"],
                       expose_headers=["Content-Range", "Accept-Ranges"])

    @app.middleware("http")
    async def local_access(request: Request, call_next):
        # Local paths and jobs are private desktop capabilities. A remote page
        # must never turn this loopback service into a filesystem write proxy.
        host = urlsplit("http://" + request.headers.get("host", "")).hostname
        origin = request.headers.get("origin")
        if host not in {"127.0.0.1", "localhost", "testserver", "::1"} or (origin and origin not in trusted_origins):
            return JSONResponse(status_code=403, content={"detail": {"code": "local_access_only", "message": "Use the local Roundnet editor to access this service."}})
        return await call_next(request)

    @app.exception_handler(ProjectError)
    async def project_error(request, exc):
        return JSONResponse(status_code=exc.status, content={"detail": exc.to_dict()})

    @app.exception_handler(ValueError)
    @app.exception_handler(TypeError)
    @app.exception_handler(KeyError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={"detail": {"code": "invalid_input", "message": str(exc)}})

    @app.exception_handler(OSError)
    async def filesystem_error(request, exc):
        return JSONResponse(status_code=422, content={"detail": {"code": "file_error", "message": str(exc)}})

    @app.exception_handler(VideoIOError)
    async def media_error(request, exc):
        return JSONResponse(status_code=422, content={"detail": {"code": "media_error", "message": str(exc)}})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "application": "roundnet"}

    @app.get("/api/defaults")
    def defaults():
        return {"detection_settings": DetectionSettings().to_dict(), "classifications": CLASSIFICATIONS,
                "touch_results": RESULTS, "stages": STAGES}

    @app.get("/api/projects")
    def list_projects():
        return {"projects": projects.list(), "recovery_errors": projects.recovery_errors}

    @app.post("/api/projects")
    def new_project(body: dict = Body(...)):
        return projects.create(body["video_path"], title=body.get("title"))

    @app.post("/api/projects/open")
    def open_project(body: dict = Body(...)):
        return projects.open(body["path"], video_path=body.get("video_path"),
                             allow_source_mismatch=body.get("allow_source_mismatch", False))

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        return projects.snapshot(project_id)

    @app.get("/api/projects/{project_id}/statistics")
    def statistics_download(project_id: str, format: str = "json"):
        if format not in {"json", "csv"}:
            raise ValueError("Choose JSON or CSV statistics")
        snapshot = projects.snapshot(project_id)
        directory = projects.data_dir / "statistics"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{project_id}-{snapshot['revision']}.{format}"
        export_statistics(path, snapshot["statistics"], protected_paths=(snapshot["video_path"], snapshot.get("project_path")))
        return FileResponse(path, media_type="application/json" if format == "json" else "text/csv",
                            filename=f"{snapshot['title']}-statistics.{format}")

    @app.post("/api/projects/{project_id}/commands")
    def command(project_id: str, body: dict = Body(...)):
        return projects.command(project_id, body)

    @app.post("/api/files/pick")
    def pick_file(body: dict = Body(...)):
        return {"path": _pick_file(body.get("kind", "video"))}

    from .jobs import install_job_routes
    from .labels import install_label_routes
    from .media import install_media_routes
    install_job_routes(app, projects)
    install_media_routes(app, projects)
    install_label_routes(app, projects)
    return app


app = create_app()
