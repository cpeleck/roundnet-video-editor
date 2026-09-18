"""Validated settings for heuristic rally detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from numbers import Integral, Real
from typing import Any, Mapping


@dataclass
class DetectionSettings:
    """Tunable parameters for the explainable rally detector.

    Values are deliberately ordinary dataclass fields so a desktop settings
    panel can edit them without depending on a framework-specific config type.
    Durations are seconds, rates are samples per second, and thresholds/scores
    use a normalized 0-to-1 scale.
    """

    analysis_fps: float = 8.0
    downscale_width: int = 640
    frame_difference_threshold: int = 12

    motion_sensitivity: float = 1.0
    audio_sensitivity: float = 1.0
    motion_weight: float = 0.20
    roi_motion_weight: float = 0.22
    motion_spread_weight: float = 0.20
    audio_weight: float = 0.10
    temporal_weight: float = 0.18
    serve_weight: float = 0.10

    serve_sensitivity: float = 1.0
    serve_quiet_window: float = 1.25
    serve_post_window: float = 0.9
    serve_hold_duration: float = 1.5

    # Native macOS body pose (or bundled OpenCV HOG fallback) is sampled and
    # tracked between detections. A supplied COCO-17 pose ONNX can replace it.
    player_tracking_enabled: bool = True
    person_detection_interval: float = 0.75
    pose_model_path: str = ""
    court_context_strength: float = 0.15

    # The retrieval filter is intentionally conservative.  It disables (rather
    # than deletes) a candidate only when motion remains compact, there is no
    # plausible serve cue, and no sample contains strongly distributed play.
    retrieval_filter_enabled: bool = True
    retrieval_spread_threshold: float = 0.30
    strong_play_spread_threshold: float = 0.55
    serve_confirmation_threshold: float = 0.50

    rally_threshold: float = 0.55
    end_threshold: float = 0.38
    start_active_duration: float = 0.375
    end_inactive_duration: float = 1.0
    min_rally_duration: float = 1.5
    max_rally_duration: float = 30.0
    pre_roll: float = 0.75
    post_roll: float = 1.25
    merge_gap: float = 1.0

    smoothing_window: float = 0.5
    temporal_window: float = 1.0
    audio_sample_rate: int = 16_000
    audio_window_duration: float = 0.08
    ffmpeg_path: str = "ffmpeg"
    ffmpeg_timeout: float = 600.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` with a user-facing description when invalid."""

        self._positive("analysis_fps", self.analysis_fps)
        if not isinstance(self.downscale_width, Integral) or isinstance(
            self.downscale_width, bool
        ) or self.downscale_width < 64:
            raise ValueError("downscale_width must be an integer of at least 64")
        if not isinstance(self.frame_difference_threshold, Integral) or isinstance(
            self.frame_difference_threshold, bool
        ) or not 0 <= self.frame_difference_threshold <= 255:
            raise ValueError("frame_difference_threshold must be between 0 and 255")

        self._positive("motion_sensitivity", self.motion_sensitivity)
        self._positive("audio_sensitivity", self.audio_sensitivity)
        self._positive("serve_sensitivity", self.serve_sensitivity)
        if not isinstance(self.retrieval_filter_enabled, bool):
            raise TypeError("retrieval_filter_enabled must be a bool")
        if not isinstance(self.player_tracking_enabled, bool):
            raise TypeError("player_tracking_enabled must be a bool")
        if not isinstance(self.pose_model_path, str):
            raise TypeError("pose_model_path must be a string")
        self._unit_interval("court_context_strength", self.court_context_strength)
        for name in (
            "motion_weight",
            "roi_motion_weight",
            "motion_spread_weight",
            "audio_weight",
            "temporal_weight",
            "serve_weight",
        ):
            self._non_negative(name, getattr(self, name))
        if self.total_weight <= 0:
            raise ValueError("at least one scoring weight must be greater than zero")

        self._unit_interval("rally_threshold", self.rally_threshold)
        self._unit_interval("end_threshold", self.end_threshold)
        self._unit_interval(
            "retrieval_spread_threshold", self.retrieval_spread_threshold
        )
        self._unit_interval(
            "strong_play_spread_threshold", self.strong_play_spread_threshold
        )
        self._unit_interval(
            "serve_confirmation_threshold", self.serve_confirmation_threshold
        )
        if self.end_threshold > self.rally_threshold:
            raise ValueError("end_threshold cannot be greater than rally_threshold")
        if self.strong_play_spread_threshold < self.retrieval_spread_threshold:
            raise ValueError(
                "strong_play_spread_threshold cannot be lower than "
                "retrieval_spread_threshold"
            )

        self._non_negative("start_active_duration", self.start_active_duration)
        for name in (
            "end_inactive_duration",
            "min_rally_duration",
            "smoothing_window",
            "temporal_window",
            "serve_quiet_window",
            "serve_post_window",
            "serve_hold_duration",
            "person_detection_interval",
            "audio_window_duration",
            "ffmpeg_timeout",
        ):
            self._positive(name, getattr(self, name))
        self._positive("max_rally_duration", self.max_rally_duration)
        if self.max_rally_duration < self.min_rally_duration:
            raise ValueError("max_rally_duration cannot be shorter than min_rally_duration")
        for name in ("pre_roll", "post_roll", "merge_gap"):
            self._non_negative(name, getattr(self, name))

        if not isinstance(self.audio_sample_rate, Integral) or isinstance(
            self.audio_sample_rate, bool
        ) or self.audio_sample_rate < 1_000:
            raise ValueError("audio_sample_rate must be an integer of at least 1000")
        if not isinstance(self.ffmpeg_path, str) or not self.ffmpeg_path.strip():
            raise ValueError("ffmpeg_path must be a non-empty string")

    @property
    def total_weight(self) -> float:
        return (
            self.motion_weight
            + self.roi_motion_weight
            + self.motion_spread_weight
            + self.audio_weight
            + self.temporal_weight
            + self.serve_weight
        )

    # Readable aliases used by some settings panels and older project files.
    @property
    def minimum_rally_duration(self) -> float:
        return self.min_rally_duration

    @minimum_rally_duration.setter
    def minimum_rally_duration(self, value: float) -> None:
        self.min_rally_duration = value

    @property
    def maximum_rally_duration(self) -> float:
        return self.max_rally_duration

    @maximum_rally_duration.setter
    def maximum_rally_duration(self, value: float) -> None:
        self.max_rally_duration = value

    @property
    def analysis_width(self) -> int:
        return self.downscale_width

    @analysis_width.setter
    def analysis_width(self, value: int) -> None:
        self.downscale_width = value

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible settings dictionary."""

        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectionSettings":
        """Load settings, accepting a small set of historical field aliases."""

        if not isinstance(data, Mapping):
            raise TypeError("settings data must be a mapping")
        aliases = {
            "minimum_rally_duration": "min_rally_duration",
            "maximum_rally_duration": "max_rally_duration",
            "analysis_width": "downscale_width",
            "minimum_inactive_duration": "end_inactive_duration",
        }
        valid = {field.name for field in fields(cls)}
        values: dict[str, Any] = {}
        unknown: list[str] = []
        for raw_name, value in data.items():
            name = aliases.get(raw_name, raw_name)
            if name not in valid:
                unknown.append(str(raw_name))
            else:
                values[name] = value
        if unknown:
            raise ValueError(f"unknown detection setting(s): {', '.join(sorted(unknown))}")
        return cls(**values)

    def updated(self, **changes: Any) -> "DetectionSettings":
        """Return a validated copy containing ``changes``."""

        aliases = {
            "minimum_rally_duration": "min_rally_duration",
            "maximum_rally_duration": "max_rally_duration",
            "analysis_width": "downscale_width",
        }
        normalized = {aliases.get(name, name): value for name, value in changes.items()}
        return replace(self, **normalized)

    @staticmethod
    def _positive(name: str, value: float) -> None:
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")

    @staticmethod
    def _non_negative(name: str, value: float) -> None:
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")

    @staticmethod
    def _unit_interval(name: str, value: float) -> None:
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be between 0 and 1")


DEFAULT_DETECTION_SETTINGS = DetectionSettings()
