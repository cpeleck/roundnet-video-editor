"""FastAPI routes for managed detector and exporter workers."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse

from application.jobs import JobManager


def install_job_routes(app: Any, service: Any, manager: JobManager | None = None) -> JobManager:
    jobs = manager or JobManager(service)
    app.state.jobs = jobs
    router = APIRouter()

    @router.post("/api/projects/{project_id}/detect", status_code=202)
    def detect(project_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return jobs.start(project_id, "detection", body)

    @router.post("/api/projects/{project_id}/export", status_code=202)
    def export(project_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return jobs.start(project_id, "export", body)

    @router.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

    @router.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

    @router.get("/api/jobs/{job_id}/output")
    def output(job_id: str) -> FileResponse:
        job = get_job(job_id)
        if job["kind"] != "export" or job["status"] != "succeeded":
            raise HTTPException(409, "This job has no completed export")
        return FileResponse(job["result"]["output_path"], media_type="video/mp4", filename="roundnet-export.mp4")

    app.include_router(router)
    return jobs
