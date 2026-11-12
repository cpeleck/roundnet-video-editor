"""Local FastAPI service for the focused browser editor."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import os
import subprocess
import sys
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from application.projects import ProjectError, ProjectService, STAGES
from config import DetectionSettings
from models.match_flow import CLASSIFICATIONS
from models.point_stats import RESULTS
from models.statistics import export_statistics
from video.errors import VideoIOError
from .phone import LOCAL_HOSTS, check_settings, install_phone_routes, is_remote, managed_recording, paired


def _pick_file(kind):
    """Use the desktop host's native picker; no GUI framework enters the API."""
    # Selection can include downloading a cloud-backed recording. Let the user
    # finish or cancel the dialog instead of killing it after five minutes.
    if kind not in ("video", "project", "destination"):
        raise ValueError("Choose a video, project, or destination picker")
    if sys.platform == "darwin":
        script = ('POSIX path of (choose file name with prompt "Save exported video" default name "roundnet.mp4")'
                  if kind == "destination" else
                  'POSIX path of (choose file with prompt "Choose ' + ("recording" if kind == "video" else "Roundnet project") + '")')
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=None)
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
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=None)
    if result.returncode:
        raise ProjectError("The native file picker is unavailable. Enter an absolute path instead.", code="picker_unavailable", status=503)
    return result.stdout.strip() or None


def create_app(data_dir=None, service=None):
    projects = service or ProjectService(data_dir)
    frontend_port = int(os.environ.get("ROUNDNET_FRONTEND_PORT", "5173"))
    api_port = int(os.environ.get("ROUNDNET_API_PORT", "8000"))
    phone_token = os.environ.get("ROUNDNET_PHONE_TOKEN", "")
    phone_host = os.environ.get("ROUNDNET_PHONE_HOST", "") if phone_token else ""
    trusted_origins = {f"http://{host}:{port}" for host in ("127.0.0.1", "localhost")
                       for port in (frontend_port, api_port)}
    if phone_host:
        trusted_origins.add(f"http://{phone_host}:{frontend_port}")

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
        host = urlsplit("http://" + request.headers.get("host", "")).hostname
        origin = request.headers.get("origin")
        if host not in LOCAL_HOSTS | ({phone_host} if phone_host else set()) or (origin and origin not in trusted_origins):
            return JSONResponse(status_code=403, content={"detail": {"code": "local_access_only", "message": "Use the local Roundnet editor to access this service."}})
        if is_remote(request) and request.method == "POST" and not origin:
            return JSONResponse(status_code=403, content={"detail": "Open the editor using the pairing link."})
        exempt = request.url.path == "/api/session" or (request.url.path == "/api/health" and not is_remote(request))
        if phone_token and not exempt and not paired(request, phone_token):
            return JSONResponse(status_code=401, content={"detail": "Open the pairing link shown on your Mac."})
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
    def new_project(request: Request, body: dict = Body(...)):
        if is_remote(request):
            body["video_path"] = managed_recording(projects, body["video_path"])
        return projects.create(body["video_path"], title=body.get("title"))

    @app.post("/api/projects/open")
    def open_project(request: Request, body: dict = Body(...)):
        if is_remote(request):
            raise HTTPException(403, "Open existing matches from the Projects screen on your phone.")
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
    def command(project_id: str, request: Request, body: dict = Body(...)):
        if is_remote(request):
            kind = body.get("type")
            if kind == "save_as":
                raise HTTPException(403, "Project edits are saved automatically on your Mac.")
            if kind == "relink":
                body["video_path"] = managed_recording(projects, body["video_path"])
            if kind == "settings":
                check_settings(body.get("settings", {}), projects.snapshot(project_id)["settings"])
            if kind == "export_settings":
                changes = body.get("settings", {})
                stored = projects.snapshot(project_id)["export_settings"]
                if changes.get("overlay_path") and changes["overlay_path"] != stored.get("overlay_path"):
                    raise HTTPException(403, "Choose branding files from the Mac.")
                changes.pop("output_path", None)
        return projects.command(project_id, body)

    @app.post("/api/files/pick")
    def pick_file(request: Request, body: dict = Body(...)):
        if is_remote(request):
            raise HTTPException(403, "Use your phone's recording chooser.")
        return {"path": _pick_file(body.get("kind", "video"))}

    from .jobs import install_job_routes
    from .labels import install_label_routes
    from .media import install_media_routes
    from .admin import install_admin_routes
    jobs = install_job_routes(app, projects)
    install_media_routes(app, projects)
    install_label_routes(app, projects)
    install_phone_routes(app, projects, phone_token)
    install_admin_routes(app, projects, jobs)
    return app


app = create_app()
