"""Rally domain model.

All time values in the application are represented in seconds.  Keeping the
model independent from frame rates is important because phone footage can be
variable-frame-rate and because analysis is intentionally performed at a much
lower frame rate than export.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping
from uuid import uuid4

from .point_stats import normalize_point_stats


@dataclass
class Rally:
    """A contiguous interval of active play.

    Args:
        start_time: Inclusive start time in seconds.
        end_time: Exclusive end time in seconds.
        confidence: Detector confidence in the closed interval ``[0, 1]``.
        enabled: Whether this rally should be included during export.
        serve_confidence: Confidence that a serve-like start cue preceded this
            interval.  This is supporting evidence, not a requirement.

    The class is intentionally mutable so the review UI can adjust boundaries
    directly.  Call :meth:`validate` after a group of in-place edits, or prefer
    :meth:`with_bounds`, which validates immediately.
    """

    start_time: float
    end_time: float
    confidence: float = 1.0
    enabled: bool = True
    serve_confidence: float = 0.0
    rally_id: str = field(default_factory=lambda: uuid4().hex)
    reviewed: bool = False
    rejected: bool = False
    starred: bool = False
    outcome: str = ""
    winner: str = ""
    player: str = ""
    note: str = ""
    crop_keyframes: list[dict[str, float]] = field(default_factory=list)
    point_stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce common numeric scalar types (including numpy scalar values) to
        # plain floats so serialization remains predictable.
        self.start_time = float(self.start_time)
        self.end_time = float(self.end_time)
        self.confidence = float(self.confidence)
        self.serve_confidence = float(self.serve_confidence)
        self.crop_keyframes = deepcopy(self.crop_keyframes)
        self.point_stats = normalize_point_stats(self.point_stats)
        self.validate()

    @property
    def duration(self) -> float:
        """Duration of the rally in seconds."""

        return self.end_time - self.start_time

    def validate(self) -> None:
        """Raise :class:`ValueError` if the rally has invalid values."""

        normalize_point_stats(self.point_stats)
        if not math.isfinite(self.start_time) or self.start_time < 0:
            raise ValueError("start_time must be a finite, non-negative number")
        if not math.isfinite(self.end_time) or self.end_time <= self.start_time:
            raise ValueError("end_time must be finite and greater than start_time")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be a finite number between 0 and 1")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        if (
            not math.isfinite(self.serve_confidence)
            or not 0.0 <= self.serve_confidence <= 1.0
        ):
            raise ValueError("serve_confidence must be between 0 and 1")
        for name in ("reviewed", "rejected", "starred"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in ("rally_id", "outcome", "winner", "player", "note"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be text")
        if not self.rally_id or self.winner not in ("", "A", "B"):
            raise ValueError("rally_id is required and winner must be A, B, or empty")
        if self.rejected and self.enabled:
            raise ValueError("rejected rallies cannot be enabled")
        last_time = -1.0
        for key in self.crop_keyframes:
            t, x, y = (float(key[name]) for name in ("time", "x", "y"))
            if not all(math.isfinite(v) for v in (t, x, y)):
                raise ValueError("crop keyframes must be finite")
            if t < 0 or t <= last_time or not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("crop keyframes need increasing source times and normalized centers")
            last_time = t

    def with_bounds(self, start_time: float, end_time: float) -> "Rally":
        """Return a validated copy with different time boundaries."""

        return replace(self, start_time=start_time, end_time=end_time)

    def clamped(self, video_duration: float) -> "Rally":
        """Return a copy constrained to a video's time range.

        A rally wholly outside the video cannot be represented meaningfully and
        therefore raises ``ValueError`` rather than returning a zero-length
        interval.
        """

        duration = float(video_duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("video_duration must be a finite positive number")
        start = min(max(0.0, self.start_time), duration)
        end = min(max(0.0, self.end_time), duration)
        if end <= start:
            raise ValueError("rally does not overlap the video")
        return replace(self, start_time=start, end_time=end)

    def overlaps(self, other: "Rally", *, gap: float = 0.0) -> bool:
        """Return whether this rally overlaps or is within ``gap`` of another."""

        gap_value = float(gap)
        if not math.isfinite(gap_value) or gap_value < 0:
            raise ValueError("gap must be a finite non-negative number")
        return (
            self.start_time <= other.end_time + gap_value
            and other.start_time <= self.end_time + gap_value
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the rally to a JSON-compatible dictionary."""

        self.validate()
        return asdict(self)

    def to_annotation_dict(self) -> dict[str, float]:
        """Serialize in the compact shape used by future training labels."""

        self.validate()
        return {"start": self.start_time, "end": self.end_time}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Rally":
        """Create a rally from serialized data.

        Both ``start_time``/``end_time`` and the compact ``start``/``end``
        annotation keys are accepted.  Explicit time keys take precedence.
        """

        if not isinstance(data, Mapping):
            raise TypeError("rally data must be a mapping")
        start = data.get("start_time", data.get("start"))
        end = data.get("end_time", data.get("end"))
        if start is None or end is None:
            raise ValueError("rally data must contain start_time/end_time or start/end")
        return cls(
            start_time=start,
            end_time=end,
            confidence=data.get("confidence", 1.0),
            enabled=data.get("enabled", True),
            serve_confidence=data.get("serve_confidence", 0.0),
            **{name: data[name] for name in (
                "rally_id", "reviewed", "rejected", "starred", "outcome", "winner",
                "player", "note", "crop_keyframes", "point_stats"
            ) if name in data},
        )
