"""FastAPI routes for managed detector and exporter workers."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse

from application.jobs import JobManager
from .phone import check_settings, is_remote
from .admin import require_admin


def install_job_routes(app: Any, service: Any, manager: JobManager | None = None) -> JobManager:
    jobs = manager or JobManager(service)
    app.state.jobs = jobs
    router = APIRouter()

    @router.post("/api/projects/{project_id}/detect", status_code=202)
    def detect(project_id: str, request: Request, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        if is_remote(request):
            check_settings(body.get("settings", {}), service.snapshot(project_id)["settings"])
        return jobs.start(project_id, "detection", body)

    @router.post("/api/projects/{project_id}/export", status_code=202)
    def export(project_id: str, request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if is_remote(request):
            directory = service.data_dir / "exports"
            directory.mkdir(parents=True, exist_ok=True)
            body = {**body, "output_path": str(directory / f"{uuid4().hex}.mp4"), "overwrite": False}
            options = dict(body.get("options", {}))
            options["overlay_path"] = service.snapshot(project_id)["export_settings"].get("overlay_path")
            body["options"] = options
        return jobs.start(project_id, "export", body)

    @router.get("/api/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict[str, Any]:
        try:
            job = jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc
        if job["kind"] == "benchmark":
            require_admin(request)
        return job

    @router.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
        get_job(job_id, request)
        try:
            return jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

    @router.get("/api/jobs/{job_id}/output")
    def output(job_id: str, request: Request) -> FileResponse:
        job = get_job(job_id, request)
        if job["kind"] != "export" or job["status"] != "succeeded":
            raise HTTPException(409, "This job has no completed export")
        return FileResponse(job["result"]["output_path"], media_type="video/mp4", filename="roundnet-export.mp4")

    app.include_router(router)
    return jobs
