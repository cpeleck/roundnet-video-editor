"""Efficient, sampled motion analysis for mostly stationary game footage."""

from __future__ import annotations

from dataclasses import dataclass
import math
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np

from config import DetectionSettings
from .court_context import normalize_court_context
from ._callbacks import (
    CancelCallback,
    ProgressCallback,
    check_cancelled,
    report_progress,
)


ROI: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True)
class MotionAnalysis:
    """Raw motion features sampled from a source video."""

    timestamps: np.ndarray
    total_motion: np.ndarray
    roi_motion: np.ndarray
    duration: float
    source_fps: float
    width: int
    height: int
    motion_spread: np.ndarray | None = None
    player_count: np.ndarray | None = None
    readiness: np.ndarray | None = None
    player_motion: np.ndarray | None = None
    retrieval: np.ndarray | None = None
    pose_serve: np.ndarray | None = None
    player_tracks: tuple[dict[str, Any], ...] = ()
    player_backend: str = "disabled"
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        expected = self.timestamps.size
        if self.total_motion.size != expected or self.roi_motion.size != expected:
            raise ValueError("motion signals must have one item per timestamp")
        if self.motion_spread is None:
            object.__setattr__(
                self, "motion_spread", np.zeros(expected, dtype=np.float64)
            )
        elif self.motion_spread.size != expected:
            raise ValueError("motion_spread must have one item per timestamp")
        for name in ("player_count", "readiness", "player_motion", "retrieval", "pose_serve"):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, np.zeros(expected, dtype=np.float64))
            elif value.size != expected:
                raise ValueError(f"{name} must have one item per timestamp")

    @property
    def motion_scores(self) -> np.ndarray:
        """Compatibility alias for the full-frame signal."""

        return self.total_motion


class MotionDetector:
    """Measure frame-difference motion globally and inside a playing-area ROI."""

    def __init__(self, settings: DetectionSettings | None = None) -> None:
        self.settings = settings or DetectionSettings()
        self.settings.validate()

    def analyze(
        self,
        video_path: str | PathLike[str],
        roi: ROI | Mapping[str, float] | Any | None = None,
        *,
        court_context: Mapping[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> MotionAnalysis:
        """Analyze sampled, downscaled frames while preserving source timestamps."""

        cv2 = _import_cv2()
        court = normalize_court_context(court_context)
        tracker = None
        path = Path(video_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not open video: {path}")
        # Match VideoReader/ROI preview behavior for iPhone files that store
        # portrait orientation as metadata rather than rotated pixels.
        orientation_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if orientation_property is not None:
            capture.set(orientation_property, 1)

        try:
            if self.settings.player_tracking_enabled:
                from .player_tracker import PlayerTracker

                report_progress(progress_callback, 0.0, "Preparing local player detection")
                tracker = PlayerTracker(self.settings, cv2, court)
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(source_fps) or source_fps <= 0:
                source_fps = 30.0
            frame_count = int(max(0.0, capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            metadata_width = int(max(0.0, capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            metadata_height = int(max(0.0, capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            duration = frame_count / source_fps if frame_count else 0.0

            sample_period = 1.0 / self.settings.analysis_fps
            next_sample_time = 0.0
            timestamps: list[float] = []
            total_motion: list[float] = []
            roi_motion: list[float] = []
            motion_spread: list[float] = []
            player_values: dict[str, list[float]] = {
                name: [] for name in ("player_count", "readiness", "player_motion", "retrieval", "pose_serve")
            }
            track_samples: list[dict[str, Any]] = []
            previous_gray: np.ndarray | None = None
            roi_bounds: tuple[int, int, int, int] | None = None
            processed_frames = 0
            source_index = 0
            last_reported_progress = -1.0
            last_backend_timestamp = -math.inf
            last_frame_timestamp = -1.0 / source_fps

            report_progress(progress_callback, 0.0, "Sampling video motion")
            while capture.grab():
                if source_index % 32 == 0:
                    check_cancelled(cancel_callback)

                backend_timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                fallback_timestamp = source_index / source_fps
                timestamp = fallback_timestamp
                if (
                    math.isfinite(backend_timestamp)
                    and backend_timestamp >= 0
                    and (
                        source_index == 0
                        or backend_timestamp > last_backend_timestamp + 1e-9
                    )
                ):
                    timestamp = backend_timestamp
                    last_backend_timestamp = backend_timestamp
                # A few backends return the same nonzero POS_MSEC for many
                # frames.  Preserve strict monotonicity and continue sampling
                # from the frame-rate clock until that backend clock advances.
                timestamp = max(timestamp, last_frame_timestamp + 1.0 / source_fps)
                last_frame_timestamp = timestamp

                if timestamp + sample_period * 0.25 < next_sample_time:
                    source_index += 1
                    continue

                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    source_index += 1
                    continue
                original_height, original_width = frame.shape[:2]
                if original_width <= 0 or original_height <= 0:
                    source_index += 1
                    continue
                if processed_frames == 0:
                    # Decoded dimensions reflect orientation metadata, whereas
                    # CAP_PROP_FRAME_WIDTH/HEIGHT may still report the encoded
                    # (unrotated) raster on some macOS backends.
                    metadata_width, metadata_height = original_width, original_height

                frame = _downscale(frame, self.settings.downscale_width, cv2)
                processed_height, processed_width = frame.shape[:2]
                if roi_bounds is None:
                    roi_bounds = _resolve_roi(
                        roi,
                        original_width,
                        original_height,
                        processed_width,
                        processed_height,
                    )

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                if previous_gray is None:
                    total_value = 0.0
                    roi_value = 0.0
                    spread_value = 0.0
                else:
                    difference = cv2.absdiff(gray, previous_gray)
                    total_value = _motion_metric(
                        difference, self.settings.frame_difference_threshold
                    )
                    if roi_bounds is None:
                        roi_value = total_value
                        spread_difference = difference
                    else:
                        x, y, width, height = roi_bounds
                        roi_difference = difference[y : y + height, x : x + width]
                        roi_value = _motion_metric(
                            roi_difference, self.settings.frame_difference_threshold
                        )
                        spread_difference = roi_difference
                    spread_value = _motion_spread_metric(
                        spread_difference, self.settings.frame_difference_threshold
                    )

                timestamps.append(max(0.0, timestamp))
                total_motion.append(total_value)
                roi_motion.append(roi_value)
                motion_spread.append(spread_value)
                if tracker is not None:
                    features, people = tracker.process(frame, timestamp, roi_bounds)
                    for name, values in player_values.items():
                        values.append(features[name])
                    # Keep tracks at detection cadence for a compact visual QA
                    # record, including real keypoints when the backend has them.
                    if not track_samples or timestamp - track_samples[-1]["timestamp"] >= self.settings.person_detection_interval - 1e-8:
                        track_samples.append({"timestamp": timestamp, "players": people})
                else:
                    for values in player_values.values():
                        values.append(0.0)
                previous_gray = gray
                processed_frames += 1

                while next_sample_time <= timestamp + sample_period * 0.25:
                    next_sample_time += sample_period
                if duration > 0:
                    progress = min(1.0, timestamp / duration)
                    # Emitting a Qt signal for every sampled frame can flood the
                    # GUI event queue on a 30-minute recording.  Half-percent
                    # updates remain smooth while keeping analysis responsive.
                    if progress - last_reported_progress >= 0.005:
                        report_progress(
                            progress_callback,
                            progress,
                            f"Analyzing motion at {timestamp:.1f}s",
                        )
                        last_reported_progress = progress
                source_index += 1

            check_cancelled(cancel_callback)
            if not timestamps:
                raise RuntimeError(f"no decodable video frames found in: {path}")
            if duration <= 0:
                duration = timestamps[-1] + sample_period
            report_progress(progress_callback, 1.0, "Motion analysis complete")
            return MotionAnalysis(
                timestamps=np.asarray(timestamps, dtype=np.float64),
                total_motion=np.asarray(total_motion, dtype=np.float64),
                roi_motion=np.asarray(roi_motion, dtype=np.float64),
                motion_spread=np.asarray(motion_spread, dtype=np.float64),
                duration=max(duration, timestamps[-1]),
                source_fps=source_fps,
                width=metadata_width,
                height=metadata_height,
                **{name: np.asarray(values, dtype=np.float64) for name, values in player_values.items()},
                player_tracks=tuple(track_samples),
                player_backend=tracker.backend_name if tracker else "disabled",
                warnings=tuple(tracker.warnings) if tracker else (),
            )
        finally:
            capture.release()
            if tracker is not None:
                tracker.close()


def analyze_motion(
    video_path: str | PathLike[str],
    settings: DetectionSettings | None = None,
    roi: ROI | Mapping[str, float] | Any | None = None,
    *,
    court_context: Mapping[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> MotionAnalysis:
    """Functional convenience wrapper around :class:`MotionDetector`."""

    return MotionDetector(settings).analyze(
        video_path,
        roi,
        court_context=court_context,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


def _motion_metric(difference: np.ndarray, threshold: int) -> float:
    if difference.size == 0:
        return 0.0
    changed_fraction = float(np.count_nonzero(difference >= threshold)) / difference.size
    mean_intensity = float(np.mean(difference)) / 255.0
    # Changed area captures player displacement; scaled intensity adds speed and
    # prevents large, slow between-point movement from looking exactly like a
    # fast rally.
    return min(1.0, 0.7 * changed_fraction + 0.3 * min(1.0, mean_intensity * 4.0))


def _motion_spread_metric(
    difference: np.ndarray,
    threshold: int,
    *,
    grid_size: int = 3,
) -> float:
    """Measure how widely significant motion is distributed across the ROI.

    The first implementation counted active cells in a fixed grid.  A single
    player could then look more "distributed" merely by crossing a cell
    boundary.  This version uses the intensity-weighted spatial second moment
    of changed pixels instead.  It is translation invariant: moving the same
    compact silhouette within the ROI produces essentially the same result,
    while separated players on several sides of the court produce a high one.

    ``grid_size`` remains as a backwards-compatible tuning argument.  It only
    controls the small-motion noise floor; it no longer creates hard spatial
    boundaries.
    """

    if difference.size == 0:
        return 0.0
    if isinstance(grid_size, bool) or int(grid_size) != grid_size or grid_size < 1:
        raise ValueError("grid_size must be a positive integer")

    # OpenCV differences are normally 2-D grayscale arrays.  Accept a trailing
    # channel dimension defensively so the helper remains useful in tests and
    # for future color-derived motion masks.
    values = np.asarray(difference, dtype=np.float64)
    if values.ndim == 3:
        values = np.max(values, axis=2)
    if values.ndim != 2:
        raise ValueError("difference must be a 2-D image or 3-D color image")
    mask = values >= threshold
    total_changed = int(np.count_nonzero(mask))
    if total_changed == 0:
        return 0.0

    height, width = mask.shape[:2]
    y_indices, x_indices = np.nonzero(mask)
    weights = values[mask]
    weight_total = float(np.sum(weights))
    if weight_total <= 0:
        weights = np.ones(total_changed, dtype=np.float64)
        weight_total = float(total_changed)

    # Normalize each axis to 0..1 before measuring variance.  The measure then
    # behaves consistently for landscape, portrait, and cropped playing areas.
    xs = (x_indices.astype(np.float64) + 0.5) / max(1, width)
    ys = (y_indices.astype(np.float64) + 0.5) / max(1, height)
    mean_x = float(np.sum(xs * weights) / weight_total)
    mean_y = float(np.sum(ys * weights) / weight_total)
    std_x = math.sqrt(float(np.sum(np.square(xs - mean_x) * weights) / weight_total))
    std_y = math.sqrt(float(np.sum(np.square(ys - mean_y) * weights) / weight_total))

    radial_extent = math.hypot(std_x, std_y)
    # Compact people generally remain below about 0.15 in normalized radius;
    # activity on multiple sides of the court approaches 0.4 or more.
    extent_score = float(np.clip((radial_extent - 0.025) / 0.36, 0.0, 1.0))
    # A tall single silhouette has extent on only one axis.  Reward balanced
    # spatial dispersion without making axis balance a hard requirement.
    axis_balance = (min(std_x, std_y) + 0.015) / (max(std_x, std_y) + 0.015)
    shape_score = extent_score * (0.55 + 0.45 * math.sqrt(axis_balance))

    changed_fraction = total_changed / max(1, mask.size)
    # Suppress isolated compression speckles.  Larger ``grid_size`` historically
    # made the detector more spatially sensitive, so retain that relationship.
    activity_floor = 0.004 * 3.0 / int(grid_size)
    activity_gate = float(np.clip(changed_fraction / activity_floor, 0.0, 1.0))
    return float(np.clip(shape_score * activity_gate, 0.0, 1.0))


def _downscale(frame: np.ndarray, target_width: int, cv2: Any) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    scale = target_width / width
    target_height = max(1, round(height * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _resolve_roi(
    roi: ROI | Mapping[str, float] | Any | None,
    source_width: int,
    source_height: int,
    processed_width: int,
    processed_height: int,
) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None
    x, y, width, height = _roi_values(roi)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("ROI values must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")

    normalized = (
        0 <= x <= 1
        and 0 <= y <= 1
        and 0 < width <= 1
        and 0 < height <= 1
        and x + width <= 1 + 1e-9
        and y + height <= 1 + 1e-9
    )
    if normalized:
        source_x = x * source_width
        source_y = y * source_height
        source_roi_width = width * source_width
        source_roi_height = height * source_height
    else:
        source_x, source_y = x, y
        source_roi_width, source_roi_height = width, height

    left = round(source_x * processed_width / source_width)
    top = round(source_y * processed_height / source_height)
    right = round((source_x + source_roi_width) * processed_width / source_width)
    bottom = round((source_y + source_roi_height) * processed_height / source_height)
    left = min(processed_width - 1, max(0, left))
    top = min(processed_height - 1, max(0, top))
    right = min(processed_width, max(left + 1, right))
    bottom = min(processed_height, max(top + 1, bottom))
    if right <= left or bottom <= top:
        raise ValueError("ROI does not overlap the video frame")
    return left, top, right - left, bottom - top


def _roi_values(roi: ROI | Mapping[str, float] | Any) -> tuple[float, float, float, float]:
    if isinstance(roi, Mapping):
        try:
            return tuple(
                float(roi[name]) for name in ("x", "y", "width", "height")
            )  # type: ignore[return-value]
        except KeyError as exc:
            raise ValueError("ROI mapping needs x, y, width, and height") from exc
    if isinstance(roi, Sequence) and not isinstance(roi, (str, bytes)):
        if len(roi) != 4:
            raise ValueError("ROI sequence must contain x, y, width, and height")
        return tuple(float(value) for value in roi)  # type: ignore[return-value]

    # QRect/QRectF and similar UI rectangle objects expose callable accessors.
    try:
        values = []
        for name in ("x", "y", "width", "height"):
            attribute = getattr(roi, name)
            values.append(float(attribute() if callable(attribute) else attribute))
        return tuple(values)  # type: ignore[return-value]
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("ROI must be a 4-item sequence, mapping, or rectangle object") from exc


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video analysis; install project requirements first"
        ) from exc
    return cv2


__all__ = [
    "MotionAnalysis",
    "MotionDetector",
    "ROI",
    "_motion_spread_metric",
    "analyze_motion",
]
