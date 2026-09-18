"""Typed video metadata extraction with ffprobe and OpenCV backends."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Literal, Mapping, Sequence

from .errors import FFprobeNotFoundError, MetadataError, VideoFileError


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """The source properties needed by analysis, display, and export.

    ``width`` and ``height`` are the encoded dimensions.  For phone footage that
    carries rotation metadata, ``display_resolution`` reports the dimensions a
    correctly orienting player displays.
    """

    path: Path
    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int | None
    has_audio: bool | None
    video_codec: str | None = None
    audio_codec: str | None = None
    format_name: str | None = None
    rotation_degrees: int = 0
    bit_rate: int | None = None
    probe_backend: Literal["ffprobe", "opencv"] = "ffprobe"

    @property
    def duration(self) -> float:
        """Compatibility alias for UI code that calls the value ``duration``."""

        return self.duration_seconds

    @property
    def resolution(self) -> tuple[int, int]:
        """Return the encoded ``(width, height)``."""

        return (self.width, self.height)

    @property
    def display_resolution(self) -> tuple[int, int]:
        """Return the display dimensions after applying rotation metadata."""

        if abs(self.rotation_degrees) % 180 == 90:
            return (self.height, self.width)
        return self.resolution


def _validated_input_path(path: str | os.PathLike[str]) -> Path:
    media_path = Path(path).expanduser()
    if not media_path.exists():
        raise VideoFileError(f"Video file does not exist: {media_path}")
    if not media_path.is_file():
        raise VideoFileError(f"Video path is not a file: {media_path}")
    return media_path.resolve()


def _resolve_executable(
    executable: str | os.PathLike[str] | None,
    default_name: str,
) -> str | None:
    if executable is None:
        return shutil.which(default_name)
    supplied = os.fspath(executable)
    if os.sep not in supplied and (os.altsep is None or os.altsep not in supplied):
        return shutil.which(supplied)
    candidate = Path(supplied).expanduser()
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())
    return None


def _positive_float(value: object) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _nonnegative_int(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _parse_frame_rate(value: object) -> float | None:
    if value is None or value in {"", "N/A", "0/0"}:
        return None
    try:
        parsed = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _parse_rotation(stream: Mapping[str, Any]) -> int:
    rotation: object | None = None
    tags = stream.get("tags")
    if isinstance(tags, Mapping):
        rotation = tags.get("rotate")
    side_data = stream.get("side_data_list")
    if isinstance(side_data, Sequence):
        for item in side_data:
            if isinstance(item, Mapping) and item.get("rotation") is not None:
                rotation = item["rotation"]
                break
    try:
        return int(round(float(rotation))) % 360 if rotation is not None else 0
    except (TypeError, ValueError):
        return 0


def _metadata_from_ffprobe_payload(path: Path, payload: Mapping[str, Any]) -> VideoMetadata:
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list):
        raise MetadataError(f"ffprobe returned no stream list for: {path}")
    streams = [item for item in raw_streams if isinstance(item, Mapping)]
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise MetadataError(f"No video stream was found in: {path}")

    width = _nonnegative_int(video_stream.get("width")) or 0
    height = _nonnegative_int(video_stream.get("height")) or 0
    if width <= 0 or height <= 0:
        raise MetadataError(
            f"ffprobe returned invalid video dimensions {width}x{height} for: {path}"
        )

    fps = _parse_frame_rate(video_stream.get("avg_frame_rate"))
    if fps is None:
        fps = _parse_frame_rate(video_stream.get("r_frame_rate"))
    if fps is None:
        raise MetadataError(f"ffprobe could not determine a valid frame rate for: {path}")

    format_data = payload.get("format")
    if not isinstance(format_data, Mapping):
        format_data = {}
    duration = _positive_float(video_stream.get("duration"))
    if duration is None:
        duration = _positive_float(format_data.get("duration"))
    frame_count = _nonnegative_int(video_stream.get("nb_frames"))
    if duration is None and frame_count:
        duration = frame_count / fps
    if duration is None:
        raise MetadataError(f"ffprobe could not determine a valid duration for: {path}")
    if frame_count is None:
        frame_count = max(1, round(duration * fps))

    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    bit_rate = _nonnegative_int(format_data.get("bit_rate"))
    return VideoMetadata(
        path=path,
        duration_seconds=duration,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        has_audio=audio_stream is not None,
        video_codec=_optional_string(video_stream.get("codec_name")),
        audio_codec=(
            _optional_string(audio_stream.get("codec_name")) if audio_stream else None
        ),
        format_name=_optional_string(format_data.get("format_name")),
        rotation_degrees=_parse_rotation(video_stream),
        bit_rate=bit_rate,
        probe_backend="ffprobe",
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _probe_with_ffprobe(
    path: Path,
    ffprobe_path: str,
    timeout_seconds: float,
) -> VideoMetadata:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise MetadataError(
            f"ffprobe timed out after {timeout_seconds:g} seconds while reading: {path}"
        ) from exc
    except OSError as exc:
        raise MetadataError(f"Could not start ffprobe ({ffprobe_path}): {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "ffprobe returned no error details"
        raise MetadataError(
            f"ffprobe could not read metadata from {path}: {_short_error(detail)}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MetadataError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise MetadataError(f"ffprobe returned an invalid metadata object for: {path}")
    return _metadata_from_ffprobe_payload(path, payload)


def _probe_with_opencv(path: Path) -> VideoMetadata:
    try:
        import cv2  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise MetadataError(
            "OpenCV fallback is unavailable. Install the Python requirements and "
            "install FFmpeg (including ffprobe)."
        ) from exc

    try:
        capture = cv2.VideoCapture(str(path))
    except Exception as exc:
        raise MetadataError(f"OpenCV could not open video file {path}: {exc}") from exc
    try:
        if not capture.isOpened():
            raise MetadataError(f"OpenCV could not open video file: {path}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count_value = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width <= 0 or height <= 0:
            raise MetadataError(
                f"OpenCV returned invalid video dimensions {width}x{height} for: {path}"
            )
        if not math.isfinite(fps) or fps <= 0:
            raise MetadataError(f"OpenCV could not determine a valid frame rate for: {path}")
        if frame_count_value <= 0:
            raise MetadataError(f"OpenCV could not determine a valid frame count for: {path}")
        duration = frame_count_value / fps
        return VideoMetadata(
            path=path,
            duration_seconds=duration,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count_value,
            # VideoCapture cannot reliably report the presence of an audio stream.
            has_audio=None,
            probe_backend="opencv",
        )
    finally:
        capture.release()


def _short_error(message: str, limit: int = 1200) -> str:
    collapsed = " ".join(message.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


def probe_video_metadata(
    path: str | os.PathLike[str],
    *,
    ffprobe_path: str | os.PathLike[str] | None = None,
    allow_opencv_fallback: bool = True,
    timeout_seconds: float = 30.0,
) -> VideoMetadata:
    """Read source metadata, preferring ffprobe over OpenCV.

    OpenCV is intentionally only a fallback: it cannot inspect audio streams and
    provides less reliable information for variable-frame-rate phone footage.
    When both backends fail, the raised error retains both failure reasons.
    """

    media_path = _validated_input_path(path)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")

    resolved_ffprobe = _resolve_executable(ffprobe_path, "ffprobe")
    probe_error: MetadataError | FFprobeNotFoundError | None = None
    if resolved_ffprobe is None:
        probe_error = FFprobeNotFoundError(
            "ffprobe was not found. Install FFmpeg (for example, `brew install ffmpeg`) "
            "or pass ffprobe_path explicitly."
        )
    else:
        try:
            return _probe_with_ffprobe(media_path, resolved_ffprobe, timeout_seconds)
        except MetadataError as exc:
            probe_error = exc

    if not allow_opencv_fallback:
        raise probe_error

    LOGGER.warning("Falling back to OpenCV metadata for %s: %s", media_path, probe_error)
    try:
        return _probe_with_opencv(media_path)
    except MetadataError as opencv_error:
        raise MetadataError(
            f"Could not read video metadata for {media_path}. "
            f"ffprobe: {probe_error} OpenCV fallback: {opencv_error}"
        ) from opencv_error


# A concise alias is convenient for call sites and backwards-compatible adapters.
get_video_metadata = probe_video_metadata
extract_video_metadata = probe_video_metadata


def is_ffprobe_available(
    ffprobe_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Return whether an executable ffprobe can be resolved."""

    return _resolve_executable(ffprobe_path, "ffprobe") is not None


__all__ = [
    "VideoMetadata",
    "extract_video_metadata",
    "get_video_metadata",
    "is_ffprobe_available",
    "probe_video_metadata",
]
