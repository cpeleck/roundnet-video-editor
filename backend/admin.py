"""Local-owner benchmark capabilities; phone pairing does not grant admin access."""

from copy import deepcopy
import json
import shutil
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from application.projects import ProjectError
from models.project import source_matches
from training.benchmark import correction_counts
from training.feedback import ANNOTATION_ALL_RALLIES, _atomic_write_text, save_feedback
from .labels import _rejections


def is_admin(request: Request) -> bool:
    # Check the connection as well as Host/Origin. Forwarded headers never grant
    # privileges. The dev proxy preserves the browser's Host and Origin.
    peer = request.client.host if request.client else ""
    host = request.url.hostname
    origin = request.headers.get("origin")
    local = {"localhost", "127.0.0.1", "::1", "testserver"}
    return (peer in local | {"testclient"} and host in local
            and (not origin or urlsplit(origin).hostname in local))


def require_admin(request: Request):
    if not is_admin(request):
        raise HTTPException(403, "Admin benchmarks are available from the local Mac editor.")


def install_admin_routes(app, service, jobs):
    root = service.data_dir / "benchmarks"
    dataset = root / "dataset"
    reports = root / "reports"
    dataset.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    router = APIRouter(prefix="/api/admin", dependencies=[Depends(require_admin)])

    def entries():
        latest = {}
        for path in dataset.glob("*/corrections.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            source = metadata["source"]
            old = latest.get(source["id"])
            if old is None or metadata["created_utc"] > old[1]["created_utc"]:
                latest[source["id"]] = (path, metadata)
        return latest

    @router.get("/benchmarks")
    def overview():
        recordings = []
        for source_id, (path, metadata) in entries().items():
            recordings.append({"id": path.parent.name, "source_id": source_id,
                               "duration_seconds": metadata["source"]["duration_seconds"],
                               "saved_utc": metadata["created_utc"],
                               "rally_count": len(metadata["final_enabled_intervals"]), **metadata["benchmark"]})
        saved_reports = []
        for path in sorted(reports.glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            report = json.loads(path.read_text(encoding="utf-8"))
            saved_reports.append({"id": path.parent.name, "created_utc": report["created_utc"],
                                  "recording_count": report["recording_count"]})
        return {"recordings": recordings, "reports": saved_reports, "active_job": jobs.active_benchmark(),
                "target_recordings": 8, "minimum_recordings": 3}

    @router.post("/benchmarks/recordings")
    def capture(body: dict = Body(...)):
        if body.get("full_review_confirmed") is not True:
            raise ValueError("Confirm reviewing the whole recording, including missed rallies, before saving ground truth.")
        project = service.get(body["project_id"])
        with project.lock:
            service._check_revision(project, body.get("expected_revision"))
            service.require_source(body["project_id"])
            if project.busy_job_id:
                raise ProjectError("Wait for detection to finish before saving ground truth.", code="project_busy", status=409)
            state = deepcopy(project.state)
        if not state.get("signals"):
            raise ValueError("Run detection before saving ground truth so the benchmark has aligned features.")
        identity = uuid4().hex
        workspace = root / "staging" / identity
        workspace.mkdir(parents=True)
        try:
            settings = state.get("analysis_settings") or state["settings"]
            # Excluding a true point from a highlight/export must not remove it
            # from ground truth. Only explicit 'not a rally' flags are negative.
            truth = [{**r, "enabled": True, "reviewed": True} for r in state["rallies"] if not r.get("rejected", False)]
            result = save_feedback(workspace / "corrections.json", source_path=state["video_path"],
                duration_seconds=state["video_duration"], roi=state.get("analysis_roi", state.get("roi")),
                initial_predictions=state.get("initial_predictions", []), final_intervals=truth,
                rejected_detections=_rejections(state), annotation_completeness=ANNOTATION_ALL_RALLIES,
                signal_data=state["signals"], detection_settings={**settings,
                    "court_context": state.get("analysis_court_context", state.get("court_context"))})
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            if not {"motion", "roi_motion"}.intersection(metadata["features"]["feature_names"]):
                raise ValueError("Re-run detection to save motion features for the baseline.")
            if "rally_score" not in metadata["features"]["feature_names"]:
                raise ValueError("Re-run detection to save the original heuristic signal for comparison.")
            metadata["benchmark"] = {"project_id": state["project_id"], "title": state["title"],
                "revision": state["revision"], "detection_run": state.get("detection_run"),
                "corrections": correction_counts(state)}
            _atomic_write_text(result.metadata_path, json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            if not source_matches(state, state["video_path"]):
                raise ProjectError("The source changed while saving labels. Relink the original recording and try again.",
                                   code="source_mismatch", status=409)
            workspace.rename(dataset / identity)
            return {"id": identity, "source_id": result.source_id}
        except Exception:
            shutil.rmtree(workspace)
            raise

    @router.post("/benchmarks/run", status_code=202)
    def run(body: dict = Body(...)):
        identities = body.get("recording_ids")
        if not isinstance(identities, list) or any(not isinstance(item, str) for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("Choose unique saved ground-truth recordings.")
        available = {path.parent.name: path for path, _ in entries().values()}
        if any(identity not in available for identity in identities):
            raise ValueError("A selected label snapshot was replaced; refresh the dataset.")
        if len(identities) < 3:
            raise ValueError("Choose at least three distinct recordings; aim for 8–10.")
        workspace = reports / uuid4().hex
        workspace.mkdir()
        try:
            return jobs.start_benchmark({"paths": [str(available[identity]) for identity in identities],
                "profile_path": str(workspace / "profile.json"), "report_path": str(workspace / "report.json"),
                "boundary_tolerance_seconds": 1.0, "report_id": workspace.name})
        except Exception:
            shutil.rmtree(workspace)
            raise

    def report_file(identity, filename):
        if len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise HTTPException(404, "Benchmark report not found")
        path = reports / identity / filename
        if not path.is_file() or not (path.parent / "report.json").is_file():
            raise HTTPException(404, "Benchmark report not found")
        return path

    @router.get("/benchmarks/reports/{identity}")
    def report(identity: str):
        return json.loads(report_file(identity, "report.json").read_text(encoding="utf-8"))

    @router.get("/benchmarks/reports/{identity}/download")
    def download(identity: str):
        return FileResponse(report_file(identity, "report.json"), media_type="application/json",
                            filename="roundnet-benchmark.json", headers={"Cache-Control": "no-store"})

    app.include_router(router)
