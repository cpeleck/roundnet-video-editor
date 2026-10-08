"""Download source-time correction labels using the established training format."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import threading
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse

from application.projects import ProjectError
from training.feedback import ANNOTATION_ALL_RALLIES, ANNOTATION_UNKNOWN, save_feedback


def _rejections(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep legacy deleted intervals and the current explicit rejection flags."""
    rallies = state.get("rallies", [])
    identities = {r["rally_id"] for r in rallies}
    rejections = [deepcopy(entry) for entry in state.get("rejected_detections", [])
                  if entry.get("rally_id") not in identities]
    rejections.extend({**r, "rejection_id": r["rally_id"], "reason": "not_a_rally"}
                      for r in rallies if r.get("rejected", False))
    return rejections


def install_label_routes(app: Any, service: Any) -> None:
    directory = service.data_dir / "labels"
    directory.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, tuple[str, Path]] = {}
    bundle_lock = threading.Lock()
    router = APIRouter()

    @router.post("/api/projects/{project_id}/correction-labels")
    def prepare_labels(project_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        unknown = set(body) - {"expected_revision", "full_review_confirmed"}
        if unknown:
            raise ValueError(f"Unknown correction-label option: {sorted(unknown)[0]}")
        confirmed = body.get("full_review_confirmed", False)
        if not isinstance(confirmed, bool):
            raise ValueError("Full-recording review confirmation must be true or false")
        project = service.get(project_id)
        with project.lock:
            service._check_revision(project, body.get("expected_revision"))
            service.require_source(project_id)
            if project.busy_job_id:
                raise ProjectError("Wait for the running job or cancel it before saving correction labels.",
                                   code="project_busy", status=409)
            state = deepcopy(project.state)

        identity = uuid4().hex
        workspace = directory / identity
        workspace.mkdir()
        try:
            # Explicit confirmation applies only to this downloaded snapshot.
            # Stars, reel membership, and match-statistics completeness cannot
            # turn unseen source footage into negative training examples.
            settings = state.get("analysis_settings") or state.get("settings", {})
            roi = state.get("analysis_roi", state.get("roi"))
            court = state.get("analysis_court_context", state.get("court_context"))
            result = save_feedback(
                workspace / "corrections.json",
                source_path=state["video_path"], duration_seconds=state["video_duration"], roi=roi,
                initial_predictions=state.get("initial_predictions", []),
                final_intervals=[r for r in state["rallies"] if not r.get("rejected", False)],
                rejected_detections=_rejections(state),
                annotation_completeness=ANNOTATION_ALL_RALLIES if confirmed else ANNOTATION_UNKNOWN,
                signal_data=state.get("signals") or None,
                detection_settings={**settings, "court_context": court},
            )
            archive = workspace / "roundnet-correction-labels.zip"
            with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
                bundle.write(result.metadata_path, result.metadata_path.name)
                if result.features_path:
                    bundle.write(result.features_path, result.features_path.name)
            with bundle_lock:
                bundles[identity] = (project_id, archive)
            return {"download_url": f"/api/projects/{project_id}/correction-labels/{identity}",
                    "revision": state["revision"], "features_available": result.features_path is not None,
                    "sample_count": result.sample_count, "positive_count": result.positive_count,
                    "negative_count": result.negative_count, "ignored_count": result.ignored_count,
                    "full_review_confirmed": confirmed}
        except Exception:
            shutil.rmtree(workspace)
            raise

    @router.get("/api/projects/{project_id}/correction-labels/{identity}")
    def download_labels(project_id: str, identity: str) -> FileResponse:
        with bundle_lock:
            bundle = bundles.get(identity)
        if bundle is None or bundle[0] != project_id or not bundle[1].is_file():
            raise HTTPException(404, "Correction-label download not found; prepare a new download.")
        return FileResponse(bundle[1], media_type="application/zip", filename="roundnet-correction-labels.zip",
                            headers={"Cache-Control": "private, no-store"})

    app.include_router(router)
