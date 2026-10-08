"""Source video, cached thumbnails, compatible proxies and shared card previews."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from application.jobs import export_options
from video.metadata import probe_video_metadata


class MediaService:
    def __init__(self, projects: Any, jobs: Any, cache_dir: str | Path | None = None) -> None:
        self.projects, self.jobs = projects, jobs
        self.directory = Path(cache_dir or projects.data_dir / "media")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._metadata: dict[str, Any] = {}
        self._previews: dict[str, str] = {}

    def source(self, project_id: str) -> tuple[Path, str, Any]:
        source = self.projects.require_source(project_id)
        stat = source.stat()
        identity = hashlib.sha256(f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
        with self._lock:
            if identity not in self._metadata:
                self._metadata[identity] = probe_video_metadata(source)
            metadata = self._metadata[identity]
        return source, identity, metadata

    @staticmethod
    def compatible(metadata: Any) -> bool:
        # H.264/AAC MP4 is the lowest common denominator for desktop browsers.
        return (metadata.video_codec == "h264" and metadata.audio_codec in {None, "aac", "mp3"}
                and metadata.format_name is not None and "mp4" in metadata.format_name)

    def info(self, project_id: str, *, force_proxy: bool = False, retry: bool = False) -> dict[str, Any]:
        source, identity, metadata = self.source(project_id)
        proxy_required = force_proxy or not self.compatible(metadata)
        result = {"url": f"/api/projects/{project_id}/media" + ("?proxy=true" if force_proxy else ""),
            "duration": metadata.duration_seconds, "width": metadata.display_resolution[0], "height": metadata.display_resolution[1],
            "video_codec": metadata.video_codec, "source_timestamps": True, "proxy_required": proxy_required,
            "status": "ready", "job": None}
        if not proxy_required:
            return result
        destination = self.directory / f"{identity}-preview.mp4"
        with self._lock:
            if destination.is_file():
                return result
            job_id = self._previews.get(identity)
            job = self.jobs.get(job_id) if job_id else None
            if job is None or job["status"] in {"cancelled", "succeeded"} or (retry and job["status"] == "failed"):
                job = self.jobs.start(project_id, "preview", {"video_path": str(source),
                    "output_path": str(destination), "duration": metadata.duration_seconds})
                self._previews[identity] = job["id"]
            result.update(status=job["status"], job=job)
        return result

    def playback(self, project_id: str, *, force_proxy: bool = False) -> Path:
        source, identity, metadata = self.source(project_id)
        if not force_proxy and self.compatible(metadata):
            return source
        status = self.info(project_id, force_proxy=force_proxy)
        destination = self.directory / f"{identity}-preview.mp4"
        if not destination.is_file():
            if status["status"] == "failed":
                raise HTTPException(422, status["job"]["error"] or "Playback conversion failed")
            raise HTTPException(503, "Compatible playback is being prepared. Check media-info for progress.", headers={"Retry-After": "1"})
        return destination

    def branding(self, project_id: str) -> Path:
        state = self.projects.get_state(project_id)
        raw_path = state.get("export_settings", {}).get("overlay_path")
        if not raw_path:
            raise HTTPException(404, "No branding PNG has been selected")
        selected = Path(raw_path).expanduser().resolve()
        protected = [state["video_path"], state.get("project_path"),
            self.projects.project_dir / f"{project_id}.roundnet.json"]
        if selected in {Path(path).expanduser().resolve() for path in protected if path}:
            raise ValueError("Select branding separately from the source video and project")
        options = export_options(state)
        # Validation is repeated when serving, since the chosen file can change.
        path = Path(options["overlay_path"])
        with path.open("rb") as image:
            if image.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError("The saved branding file must be a PNG image")
        return path

    def thumbnail(self, project_id: str, timestamp: float, width: int) -> Path:
        import cv2
        from video.video_reader import VideoReader

        source, identity, metadata = self.source(project_id)
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp > metadata.duration_seconds:
            raise ValueError("Thumbnail time must be within the source recording")
        if not 80 <= width <= 1920:
            raise ValueError("Thumbnail width must be between 80 and 1920 pixels")
        key = hashlib.sha256(f"{identity}:{timestamp:.6f}:{width}".encode()).hexdigest()
        destination = self.directory / f"{key}.jpg"
        with self._lock:
            if not destination.is_file():
                with VideoReader(source, metadata=metadata) as reader:
                    image = reader.read_frame(timestamp, max_size=(width, width))
                success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    raise ValueError("Could not render video thumbnail")
                temporary = destination.with_name(f".{destination.name}-{uuid4().hex}")
                try:
                    temporary.write_bytes(encoded.tobytes())
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
        return destination

    def card(self, project_id: str, aspect_ratio: str = "source") -> Path:
        from video.presentation import crop_dimensions, match_statistics, render_player_end_card

        state = self.projects.get_state(project_id)
        _, _, metadata = self.source(project_id)
        options = export_options(state, {"aspect_ratio": aspect_ratio})
        width, height = crop_dimensions(*metadata.display_resolution, options["aspect_ratio"])
        summary = match_statistics(state.get("rallies", []), options)
        # Render at final output size, then let the browser scale the same PNG.
        key = hashlib.sha256(json.dumps({"width": width, "height": height, "summary": summary}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        destination = self.directory / f"{key}-card.png"
        with self._lock:
            if not destination.is_file():
                temporary = destination.with_name(f".{destination.stem}-{uuid4().hex}.png")
                try:
                    render_player_end_card(temporary, width, height, summary)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
        return destination


def install_media_routes(app: Any, service: Any, media: MediaService | None = None, *, cache_dir: str | Path | None = None) -> MediaService:
    media = media or MediaService(service, app.state.jobs, cache_dir)
    app.state.media = media
    router = APIRouter()

    @router.get("/api/projects/{project_id}/media-info")
    def info(project_id: str, proxy: bool = False, retry: bool = False) -> dict[str, Any]:
        return media.info(project_id, force_proxy=proxy, retry=retry)

    @router.get("/api/projects/{project_id}/media")
    def playback(project_id: str, proxy: bool = False) -> FileResponse:
        return FileResponse(media.playback(project_id, force_proxy=proxy), media_type="video/mp4",
            headers={"Cache-Control": "private, max-age=3600", "X-Source-Timestamps": "true"})

    @router.get("/api/projects/{project_id}/thumbnail")
    def thumbnail(project_id: str, time: float = Query(0, ge=0), width: int = Query(640, ge=80, le=1920)) -> FileResponse:
        return FileResponse(media.thumbnail(project_id, time, width), media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"})

    @router.get("/api/projects/{project_id}/card")
    def card(project_id: str, aspect_ratio: str = "source") -> FileResponse:
        return FileResponse(media.card(project_id, aspect_ratio), media_type="image/png",
            headers={"Cache-Control": "no-cache"})

    @router.get("/api/projects/{project_id}/branding")
    def branding(project_id: str) -> FileResponse:
        return FileResponse(media.branding(project_id), media_type="image/png",
            headers={"Cache-Control": "no-cache"})

    app.include_router(router)
    return media
