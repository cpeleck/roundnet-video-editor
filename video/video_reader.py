"""OpenCV-backed random-access and sampled video frame extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TypeAlias

from .errors import (
    FrameExtractionCancelledError,
    FrameExtractionError,
    VideoFileError,
)
from .metadata import VideoMetadata, probe_video_metadata


ProgressCallback: TypeAlias = Callable[[float], None]
CancelCallback: TypeAlias = Callable[[], bool]
ROI: TypeAlias = tuple[float, float, float, float] | tuple[int, int, int, int]
MaxSize: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class FrameSample:
    """One decoded analysis frame and its source timestamp."""

    timestamp_seconds: float
    frame: Any
    source_frame_index: int

    @property
    def timestamp(self) -> float:
        """Compatibility alias for consumers using the shorter field name."""

        return self.timestamp_seconds


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise FrameExtractionError(
            "OpenCV is required for frame extraction. Install the Python "
            "requirements before opening video previews."
        ) from exc
    return cv2


def _input_path(path: str | os.PathLike[str]) -> Path:
    media_path = Path(path).expanduser()
    if not media_path.exists():
        raise VideoFileError(f"Video file does not exist: {media_path}")
    if not media_path.is_file():
        raise VideoFileError(f"Video path is not a file: {media_path}")
    return media_path.resolve()


def _validate_timestamp(timestamp_seconds: float) -> float:
    try:
        timestamp = float(timestamp_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp_seconds must be a number") from exc
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp_seconds must be a non-negative finite number")
    return timestamp


def _crop_frame(
    frame: Any,
    roi: ROI | None,
    *,
    normalized_roi: bool,
) -> Any:
    if roi is None:
        return frame
    if not isinstance(roi, Sequence) or len(roi) != 4:
        raise ValueError("roi must be a four-value (x, y, width, height) tuple")
    try:
        x, y, width, height = (float(value) for value in roi)
    except (TypeError, ValueError) as exc:
        raise ValueError("roi values must be numeric") from exc
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("roi values must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("roi width and height must be greater than zero")

    frame_height, frame_width = frame.shape[:2]
    if normalized_roi:
        if x < 0 or y < 0 or x + width > 1 or y + height > 1:
            raise ValueError("normalized roi must fit inside the 0..1 frame bounds")
        x *= frame_width
        width *= frame_width
        y *= frame_height
        height *= frame_height

    left = int(round(x))
    top = int(round(y))
    right = int(round(x + width))
    bottom = int(round(y + height))
    if left < 0 or top < 0 or right > frame_width or bottom > frame_height:
        raise ValueError(
            f"roi ({left}, {top}, {right - left}, {bottom - top}) falls outside "
            f"the decoded frame bounds {frame_width}x{frame_height}"
        )
    if right <= left or bottom <= top:
        raise ValueError("roi rounds to an empty pixel region")
    return frame[top:bottom, left:right]


def _resize_to_fit(frame: Any, max_size: MaxSize | None, cv2: Any) -> Any:
    if max_size is None:
        return frame
    if not isinstance(max_size, Sequence) or len(max_size) != 2:
        raise ValueError("max_size must be a (maximum_width, maximum_height) tuple")
    try:
        maximum_width, maximum_height = (int(value) for value in max_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_size values must be integers") from exc
    if maximum_width <= 0 or maximum_height <= 0:
        raise ValueError("max_size values must be greater than zero")
    height, width = frame.shape[:2]
    scale = min(maximum_width / width, maximum_height / height, 1.0)
    if scale >= 1.0:
        return frame
    destination = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, destination, interpolation=cv2.INTER_AREA)


class VideoReader:
    """Decode source frames without creating an analysis-time transcode.

    The reader opens lazily and supports use as a context manager. Frames are BGR
    by default for OpenCV analysis; pass ``rgb=True`` for Qt/image display code.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: VideoMetadata | None = None,
    ) -> None:
        self.path = _input_path(path)
        self._metadata = metadata
        self._capture: Any | None = None
        self._cv2: Any | None = None

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            self._metadata = probe_video_metadata(self.path)
        return self._metadata

    @property
    def is_open(self) -> bool:
        return self._capture is not None and bool(self._capture.isOpened())

    def open(self) -> "VideoReader":
        if self.is_open:
            return self
        cv2 = _import_cv2()
        try:
            capture = cv2.VideoCapture(str(self.path))
        except Exception as exc:
            raise FrameExtractionError(
                f"OpenCV could not open video file {self.path}: {exc}"
            ) from exc
        if not capture.isOpened():
            capture.release()
            raise FrameExtractionError(f"OpenCV could not open video file: {self.path}")
        # Recent OpenCV builds autorotate phone video when this property is enabled.
        orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if orientation_auto is not None:
            capture.set(orientation_auto, 1)
        self._cv2 = cv2
        self._capture = capture
        return self

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    def __enter__(self) -> "VideoReader":
        return self.open()

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _opened_capture(self) -> tuple[Any, Any]:
        self.open()
        assert self._capture is not None
        assert self._cv2 is not None
        return self._capture, self._cv2

    def read_frame(
        self,
        timestamp_seconds: float,
        *,
        rgb: bool = False,
        roi: ROI | None = None,
        normalized_roi: bool = False,
        max_size: MaxSize | None = None,
    ) -> Any:
        """Decode a frame at (or immediately following) a source timestamp."""

        timestamp = _validate_timestamp(timestamp_seconds)
        duration = self.metadata.duration_seconds
        if timestamp > duration + 1e-6:
            raise FrameExtractionError(
                f"Requested frame at {timestamp:.3f}s is beyond the video duration "
                f"of {duration:.3f}s: {self.path}"
            )
        # Seeking to the exact container duration addresses the frame after EOF.
        if duration > 0 and timestamp >= duration:
            timestamp = max(0.0, duration - (1.0 / self.metadata.fps))

        capture, cv2 = self._opened_capture()
        if not capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0):
            frame_index = max(0, round(timestamp * self.metadata.fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise FrameExtractionError(
                f"OpenCV could not decode a frame near {timestamp:.3f}s from: {self.path}"
            )
        frame = _crop_frame(frame, roi, normalized_roi=normalized_roi)
        frame = _resize_to_fit(frame, max_size, cv2)
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    # A familiar alias for integrations that use ``get_frame`` terminology.
    get_frame = read_frame

    def iter_frames(
        self,
        sample_fps: float,
        *,
        start_time: float = 0.0,
        end_time: float | None = None,
        rgb: bool = False,
        roi: ROI | None = None,
        normalized_roi: bool = False,
        max_size: MaxSize | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> Iterator[FrameSample]:
        """Yield approximately ``sample_fps`` frames per second in source time.

        The capture walks forward and uses ``grab`` for skipped frames, avoiding
        inaccurate repeated keyframe seeks while only converting selected frames.
        """

        try:
            requested_fps = float(sample_fps)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_fps must be a number") from exc
        if not math.isfinite(requested_fps) or requested_fps <= 0:
            raise ValueError("sample_fps must be a positive finite number")
        start = _validate_timestamp(start_time)
        duration = self.metadata.duration_seconds
        finish = duration if end_time is None else _validate_timestamp(end_time)
        if finish > duration + 1e-6:
            raise ValueError(
                f"end_time {finish:.3f}s exceeds video duration {duration:.3f}s"
            )
        finish = min(finish, duration)
        if finish <= start:
            raise ValueError("end_time must be greater than start_time")

        capture, cv2 = self._opened_capture()
        source_fps = self.metadata.fps
        start_frame = max(0, int(math.floor(start * source_fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        next_sample_time = start
        sample_interval = 1.0 / min(requested_fps, source_fps)
        span = finish - start
        last_progress = 0.0

        if progress_callback is not None:
            progress_callback(0.0)
        while True:
            if cancel_callback is not None and cancel_callback():
                raise FrameExtractionCancelledError(
                    f"Frame extraction was cancelled at {max(start, next_sample_time):.3f}s"
                )
            if not capture.grab():
                break
            source_frame_index = max(
                0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            )
            timestamp = source_frame_index / source_fps
            if timestamp + (0.5 / source_fps) < next_sample_time:
                continue
            if timestamp >= finish:
                break
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise FrameExtractionError(
                    f"OpenCV could not decode source frame {source_frame_index} "
                    f"near {timestamp:.3f}s from: {self.path}"
                )
            frame = _crop_frame(frame, roi, normalized_roi=normalized_roi)
            frame = _resize_to_fit(frame, max_size, cv2)
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield FrameSample(timestamp, frame, source_frame_index)

            while next_sample_time <= timestamp + (0.5 / source_fps):
                next_sample_time += sample_interval
            if progress_callback is not None:
                progress = min(1.0, max(0.0, (timestamp - start) / span))
                if progress > last_progress:
                    progress_callback(progress)
                    last_progress = progress
        if progress_callback is not None:
            progress_callback(1.0)


def extract_frame(
    path: str | os.PathLike[str],
    timestamp_seconds: float = 0.0,
    *,
    rgb: bool = False,
    roi: ROI | None = None,
    normalized_roi: bool = False,
    max_size: MaxSize | None = None,
) -> Any:
    """Convenience wrapper for extracting one frame from a video."""

    with VideoReader(path) as reader:
        return reader.read_frame(
            timestamp_seconds,
            rgb=rgb,
            roi=roi,
            normalized_roi=normalized_roi,
            max_size=max_size,
        )


def extract_thumbnail(
    path: str | os.PathLike[str],
    timestamp_seconds: float | None = None,
    *,
    max_size: MaxSize = (1280, 720),
    rgb: bool = True,
) -> Any:
    """Extract a display-ready frame suitable for selecting a playing-area ROI."""

    with VideoReader(path) as reader:
        timestamp = timestamp_seconds
        if timestamp is None:
            # Avoid an often-black first frame but stay close to the game's start.
            timestamp = min(5.0, reader.metadata.duration_seconds * 0.1)
        return reader.read_frame(timestamp, rgb=rgb, max_size=max_size)


__all__ = [
    "CancelCallback",
    "FrameSample",
    "MaxSize",
    "ProgressCallback",
    "ROI",
    "VideoReader",
    "extract_frame",
    "extract_thumbnail",
]
