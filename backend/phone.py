"""Opt-in paired Wi-Fi access and streamed recording imports."""

from pathlib import Path
import secrets
import shutil
from uuid import uuid4

from fastapi import Body, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from video.metadata import probe_video_metadata

COOKIE = "roundnet_pair"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "testserver", "::1"}
MAX_UPLOAD = 20 * 1024**3
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".webm"}


def is_remote(request: Request) -> bool:
    return request.url.hostname not in LOCAL_HOSTS


def paired(request: Request, token: str) -> bool:
    return bool(token) and secrets.compare_digest(request.cookies.get(COOKIE, "").encode(), token.encode())


def managed_recording(service, raw_path):
    path = Path(raw_path).resolve()
    directory = (service.data_dir / "recordings").resolve()
    if path.parent != directory or not path.is_file():
        raise HTTPException(403, "Choose a recording from your phone first.")
    return str(path)


def check_settings(changes, stored):
    # These fields name local files or executables, rather than detector knobs.
    for key in ("ffmpeg_path", "pose_model_path"):
        if key in changes and changes[key] != stored.get(key, "ffmpeg" if key == "ffmpeg_path" else ""):
            raise HTTPException(403, "Change local detector tools from the Mac.")


def install_phone_routes(app, service, token):
    @app.get("/api/session")
    def session(request: Request):
        from .admin import is_admin
        return JSONResponse({"paired": not token or paired(request, token), "phone_enabled": bool(token),
                             "role": "admin" if is_admin(request) else "editor"},
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/session")
    def pair(body: dict = Body(...)):
        value = body.get("token", "")
        if not token or not isinstance(value, str) or not secrets.compare_digest(value.encode(), token.encode()):
            raise HTTPException(401, "Pairing link expired. Use the current link shown on your Mac.")
        response = JSONResponse({"paired": True})
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict", path="/api")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/recordings")
    async def upload(request: Request, filename: str):
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(422, "Choose a video such as MOV or MP4.")
        size = int(request.headers.get("content-length", "0"))
        if size > MAX_UPLOAD:
            raise HTTPException(413, "Recordings can be up to 20 GB.")
        directory = service.data_dir / "recordings"
        directory.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(directory).free < size + 256 * 1024**2:
            raise HTTPException(507, "Your Mac needs more free space for this recording.")
        destination = directory / (uuid4().hex + suffix)
        partial = destination.with_suffix(suffix + ".part")
        received = 0
        try:
            with partial.open("xb") as output:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > MAX_UPLOAD:
                        raise HTTPException(413, "Recordings can be up to 20 GB.")
                    await run_in_threadpool(output.write, chunk)
            if not received or (size and received != size):
                raise HTTPException(422, "The transfer was incomplete. Choose the recording again.")
            # Probe before accepting the file so a failed import leaves no media behind.
            await run_in_threadpool(probe_video_metadata, partial)
            partial.replace(destination)
            return {"path": str(destination), "filename": Path(filename).name, "size": received}
        finally:
            partial.unlink(missing_ok=True)
