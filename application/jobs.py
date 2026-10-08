"""Managed local worker processes for expensive detection and video rendering.

Workers receive immutable snapshots. Only the coordinator commits project edits,
so a cancelled/failed detector never replaces reviewed points or autosaved work.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import math
import multiprocessing
from numbers import Real
import os
from pathlib import Path
from queue import Empty
import signal
import threading
import time
from typing import Any, Mapping
from uuid import uuid4


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def export_options(state: Mapping[str, Any], changes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Use canonical match setup for both preview and final export artwork."""
    from video.presentation import prepare_export_options

    defaults = prepare_export_options()
    options = {"include_stats": True, "scoreboard": True,
               **{key: value for key, value in state.get("match_settings", {}).items() if key in defaults}}
    presentation_keys = {"aspect_ratio", "highlights_only", "scoreboard", "overlay_path", "include_stats", "include_notes", "stats_duration"}
    stored = state.get("export_settings", {})
    if isinstance(stored, Mapping):
        options.update({key: value for key, value in stored.items() if key in presentation_keys})
    if changes is not None:
        if not isinstance(changes, Mapping):
            raise ValueError("Export options must be an object")
        # Match scoring and roster always come from the project, never a render request.
        unknown = set(changes) - presentation_keys
        if unknown:
            raise ValueError(f"Unknown presentation option: {sorted(unknown)[0]}")
        options.update(changes)
    return prepare_export_options(options)


def prepare_export(state: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an independent reel queue while retaining original source times."""
    from models.rally import Rally
    from video.exporter import generate_export_segments

    mode = request.get("mode", "full")
    if mode not in {"full", "highlights"}:
        raise ValueError("Export mode must be full or highlights")
    source = Path(state["video_path"]).expanduser().resolve()
    raw_destination = request.get("output_path")
    if not isinstance(raw_destination, str) or not raw_destination.strip():
        raise ValueError("Choose an output .mp4 path")
    destination = Path(raw_destination).expanduser().resolve()
    if destination == source:
        raise ValueError("Export cannot overwrite the source recording")
    if state.get("project_path") and destination == Path(state["project_path"]).expanduser().resolve():
        raise ValueError("Export cannot overwrite the saved project")
    if destination.suffix.lower() != ".mp4" or not destination.parent.is_dir():
        raise ValueError("Choose a .mp4 path in an existing directory")
    overwrite = request.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    if destination.exists() and not overwrite:
        raise ValueError("Output already exists; choose another filename or enable overwrite")
    prefer_hardware = request.get("prefer_hardware", True)
    if not isinstance(prefer_hardware, bool):
        raise ValueError("prefer_hardware must be a boolean")
    options = export_options(state, request.get("options"))
    rallies = [Rally.from_dict(entry) for entry in state.get("rallies", [])]
    order = None
    if mode == "highlights":
        queue = request.get("queue")
        if queue is None:
            queue = state.get("highlight_queue") or [{"rally_id": rally.rally_id} for rally in rallies if rally.starred and rally.enabled and not rally.rejected]
        if not isinstance(queue, list) or not queue:
            raise ValueError("Choose at least one highlight clip")
        by_id = {rally.rally_id: rally for rally in rallies}
        order = []
        for item in queue:
            if not isinstance(item, Mapping):
                raise ValueError("Each highlight queue entry must be an object")
            identity = item.get("rally_id")
            if identity not in by_id or identity in order:
                raise ValueError("Highlight queue needs unique, existing rally identities")
            rally = by_id[identity]
            if rally.rejected:
                raise ValueError("A rejected detection cannot be a highlight")
            # A reel has its own inclusion, trim and crop choices. It never edits points.
            rally.enabled = True
            for key in ("start_time", "end_time"):
                value = item.get(key, getattr(rally, key))
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                    raise ValueError("Highlight trim times must be finite numbers")
                setattr(rally, key, float(value))
            rally.crop_keyframes = deepcopy(item.get("crop_keyframes", rally.crop_keyframes))
            rally.validate()
            order.append(identity)
    elif request.get("queue") is not None:
        raise ValueError("A clip queue is only supported for highlights")
    # Queue membership governs highlights, so stars remain only a shortlist.
    options["highlights_only"] = False
    segments = generate_export_segments(rallies, source_duration=float(state["video_duration"]), segment_order=order)
    if not segments:
        raise ValueError("No clips are selected for export")
    return {"video_path": str(source), "output_path": str(destination), "rallies": [r.to_dict() for r in rallies],
            "options": options, "segment_order": order, "presentation_rallies": deepcopy(state.get("rallies", [])), "overwrite": overwrite,
            "prefer_hardware": prefer_hardware, "mode": mode}


def prepare_detection(state: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    from config import DetectionSettings
    from detection.court_context import normalize_court_context

    settings = deepcopy(state.get("settings") or {})
    if request.get("settings") is not None:
        if not isinstance(request["settings"], Mapping):
            raise ValueError("Detector settings must be an object")
        settings.update(request["settings"])
    for key in ("debug_mode", "prefer_hardware", "save_labels_after_export"):
        settings.pop(key, None)
    settings = DetectionSettings.from_dict(settings).to_dict()
    roi = request.get("roi", state.get("roi"))
    if roi is not None:
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise ValueError("Playing area requires x, y, width and height")
        roi = [float(value) for value in roi]
        if not all(math.isfinite(value) for value in roi) or min(roi) < 0 or min(roi[2:]) <= 0:
            raise ValueError("Playing area must be finite and have positive dimensions")
        if max(roi) <= 1 and (roi[0] + roi[2] > 1.000001 or roi[1] + roi[3] > 1.000001):
            raise ValueError("Playing area must fit inside the video")
    court = normalize_court_context(request.get("court_context", state.get("court_context")))
    return {"video_path": state["video_path"], "settings": settings, "roi": roi, "court_context": court}


def _run_job(kind: str, payload: dict[str, Any], messages: Any, cancelled: Any) -> None:
    # An isolated process group lets forced shutdown also stop a worker's FFmpeg.
    if os.name == "posix":
        os.setsid()

    def progress(fraction: float, message: str = "") -> None:
        messages.put({"type": "progress", "progress": min(100, max(0, round(float(fraction) * 100))), "message": message})

    try:
        if cancelled.is_set():
            messages.put({"type": "cancelled"})
            return
        if kind == "detection":
            from config import DetectionSettings
            from detection import RoundnetDetector
            result = RoundnetDetector(DetectionSettings.from_dict(payload["settings"])).analyze(
                payload["video_path"], roi=payload["roi"], court_context=payload["court_context"],
                progress_callback=progress, cancel_callback=cancelled.is_set)
            if cancelled.is_set():
                messages.put({"type": "cancelled"})
                return
            data = result.to_dict(include_signals=True)
        elif kind == "preview":
            from video.exporter import FFmpegExporter
            destination = Path(payload["output_path"])
            temporary = destination.with_name(f".{destination.stem}-{uuid4().hex}.mp4")
            exporter = FFmpegExporter()
            command = [exporter.ffmpeg_path, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", payload["video_path"], "-map", "0:v:0", "-map", "0:a:0?",
                "-vf", "scale=1280:1280:force_original_aspect_ratio=decrease:force_divisible_by=2",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-fps_mode", "passthrough", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                "-progress", "pipe:1", "-nostats", str(temporary)]
            try:
                outcome = exporter._run_ffmpeg(command, output_duration=payload["duration"],
                    progress_callback=lambda fraction: progress(fraction, "Preparing compatible playback"), cancel_callback=cancelled.is_set)
                if outcome.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError(f"Preview conversion failed: {outcome.stderr[:1800]}")
                os.replace(temporary, destination)
                data = {"output_path": str(destination), "source_timestamps": True}
            finally:
                temporary.unlink(missing_ok=True)
        else:
            from video.exporter import export_rallies
            result = export_rallies(payload["video_path"], payload["output_path"], payload["rallies"],
                export_options=payload["options"], segment_order=payload["segment_order"],
                presentation_rallies=payload["presentation_rallies"],
                prefer_hardware=payload["prefer_hardware"], overwrite=payload["overwrite"],
                progress_callback=lambda fraction: progress(fraction, "Rendering video"),
                cancel_callback=cancelled.is_set)
            data = asdict(result)
            data["output_path"] = str(data["output_path"])
        messages.put({"type": "succeeded", "result": data})
    except Exception as exc:
        if cancelled.is_set() or "cancel" in type(exc).__name__.lower():
            messages.put({"type": "cancelled"})
        else:
            messages.put({"type": "failed", "error": f"{type(exc).__name__}: {exc}"})


class JobManager:
    """Bounded process workers with progress, cooperative cancel and crash recovery."""

    def __init__(self, projects: Any, *, max_workers: int = 2, context: Any = None) -> None:
        if max_workers < 1:
            raise ValueError("At least one worker is required")
        self.projects = projects
        self.context = context or multiprocessing.get_context("spawn")
        self._slots = threading.BoundedSemaphore(max_workers)
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._controls: dict[str, dict[str, Any]] = {}
        self._closed = False

    def start(self, project_id: str, kind: str, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if kind not in {"detection", "export", "preview"}:
            raise ValueError("Unknown job type")
        body = dict(request or {})
        identity = uuid4().hex
        with self._lock:
            if self._closed:
                raise ValueError("The job manager is shutting down")
            state = self.projects.get_state(project_id) if kind == "preview" else self.projects.lock_job(project_id, identity, body.get("expected_revision"))
            try:
                payload = body if kind == "preview" else (prepare_detection(state, body) if kind == "detection" else prepare_export(state, body))
                self._jobs[identity] = {"id": identity, "project_id": project_id, "kind": kind, "status": "queued",
                    "progress": 0, "message": "Waiting for a worker", "error": None, "result": None,
                    "created_at": datetime.now(timezone.utc).isoformat(), "finished_at": None}
                control = {"cancelled": self.context.Event(), "process": None, "cancel_time": None}
                thread = threading.Thread(target=self._monitor, args=(identity, payload), name=f"roundnet-job-{identity[:8]}", daemon=True)
                control["thread"] = thread
                self._controls[identity] = control
                thread.start()
                return deepcopy(self._jobs[identity])
            except Exception:
                if kind != "preview":
                    self.projects.unlock_job(project_id, identity)
                raise

    def get(self, identity: str) -> dict[str, Any]:
        with self._lock:
            if identity not in self._jobs:
                raise KeyError(identity)
            return deepcopy(self._jobs[identity])

    def cancel(self, identity: str) -> dict[str, Any]:
        with self._lock:
            if identity not in self._jobs:
                raise KeyError(identity)
            if self._jobs[identity]["status"] not in TERMINAL_STATUSES:
                control = self._controls[identity]
                control["cancel_time"] = control["cancel_time"] or time.monotonic()
                control["cancelled"].set()
                self._jobs[identity]["message"] = "Cancelling safely"
            return deepcopy(self._jobs[identity])

    def _finish(self, identity: str, status: str, *, result: Any = None, error: str | None = None, payload: Any = None) -> None:
        # Serializing cancel with the commit means an accepted detection cancel
        # also wins when the finished worker result is already waiting in its queue.
        with self._lock:
            if status == "succeeded" and self._jobs[identity]["kind"] == "detection" and self._controls[identity]["cancelled"].is_set():
                status, result = "cancelled", None
            self._commit_finish(identity, status, result=result, error=error, payload=payload)

    def _commit_finish(self, identity: str, status: str, *, result: Any = None, error: str | None = None, payload: Any = None) -> None:
        job = self._jobs[identity]
        project_id = job["project_id"]
        try:
            if status == "succeeded":
                updates = None
                if job["kind"] == "detection":
                    updates = {"rallies": result["rallies"], "initial_predictions": deepcopy(result["rallies"]),
                        "rejected_detections": [], "signals": result["signals"], "analysis_settings": payload["settings"],
                        "analysis_roi": payload["roi"], "analysis_court_context": payload["court_context"],
                        "settings": payload["settings"], "roi": payload["roi"], "court_context": payload["court_context"],
                        "stage": "review", "video_duration": result["duration"], "selected_rally_index": 0 if result["rallies"] else -1}
                if job["kind"] != "preview":
                    self.projects.finish_job(project_id, identity, updates)
                if job["kind"] == "detection":
                    result["project"] = self.projects.snapshot(project_id)
            elif job["kind"] != "preview":
                self.projects.unlock_job(project_id, identity)
        except Exception as exc:
            if job["kind"] != "preview":
                self.projects.unlock_job(project_id, identity)
            status, error = "failed", f"Could not save job result: {type(exc).__name__}: {exc}"
        with self._lock:
            job.update(status=status, error=error, result=result if status == "succeeded" else None,
                message={"succeeded": {"detection": "Detection complete", "export": "Export complete", "preview": "Playback ready"}[job["kind"]], "failed": "Job failed", "cancelled": "Cancelled"}[status],
                finished_at=datetime.now(timezone.utc).isoformat())
            if status == "succeeded":
                job["progress"] = 100

    @staticmethod
    def _terminate(process: Any) -> None:
        if process is None or not process.is_alive():
            return
        try:
            if os.name == "posix" and os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except (OSError, ProcessLookupError):
            pass
        process.join(timeout=1)
        if process.is_alive():
            try:
                if os.name == "posix" and os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except (OSError, ProcessLookupError):
                pass
            process.join(timeout=1)

    def _monitor(self, identity: str, payload: dict[str, Any]) -> None:
        control = self._controls[identity]
        cancelled = control["cancelled"]
        acquired = False
        messages, process = None, None
        try:
            while not cancelled.is_set():
                if self._slots.acquire(timeout=.1):
                    acquired = True
                    break
            if not acquired:
                self._finish(identity, "cancelled")
                return
            messages = self.context.Queue()
            process = self.context.Process(target=_run_job, args=(self._jobs[identity]["kind"], payload, messages, cancelled), name=f"roundnet-worker-{identity[:8]}")
            with self._lock:
                control["process"] = process
                process.start()
                self._jobs[identity].update(status="running", message="Starting worker")
            while True:
                try:
                    event = messages.get(timeout=.15)
                except Empty:
                    if not process.is_alive():
                        # Queue feeder flushes before a normal worker exit; allow its final message.
                        try:
                            event = messages.get(timeout=.3)
                        except Empty:
                            self._finish(identity, "cancelled" if cancelled.is_set() else "failed", error=f"Worker exited without a result (exit code {process.exitcode})")
                            break
                    elif cancelled.is_set() and control["cancel_time"] is not None and time.monotonic() - control["cancel_time"] > 8:
                        self._terminate(process)
                        self._finish(identity, "cancelled")
                        break
                    else:
                        continue
                if event["type"] == "progress":
                    with self._lock:
                        self._jobs[identity]["progress"] = max(self._jobs[identity]["progress"], event["progress"])
                        if not cancelled.is_set():
                            self._jobs[identity]["message"] = event.get("message") or "Working"
                else:
                    self._finish(identity, event["type"], result=event.get("result"), error=event.get("error"), payload=payload)
                    break
        except Exception as exc:
            self._finish(identity, "failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if process is not None:
                process.join(timeout=2)
                self._terminate(process)
            if messages is not None:
                messages.close()
                messages.join_thread()
            if acquired:
                self._slots.release()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            identities = list(self._controls)
        for identity in identities:
            self.cancel(identity)
        deadline = time.monotonic() + 12
        for identity in identities:
            self._controls[identity]["thread"].join(timeout=max(0, deadline - time.monotonic()))
        for identity in identities:
            control = self._controls[identity]
            if control["thread"].is_alive():
                self._terminate(control["process"])
                control["thread"].join(timeout=2)
