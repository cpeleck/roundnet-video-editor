"""Normalize independent detector signals and combine them into a rally score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from config import DetectionSettings


@dataclass(frozen=True)
class RallyScoreSeries:
    """Normalized component signals and their combined score."""

    timestamps: np.ndarray
    motion: np.ndarray
    roi_motion: np.ndarray
    audio: np.ndarray
    temporal: np.ndarray
    combined: np.ndarray
    motion_spread: np.ndarray | None = None
    serve: np.ndarray | None = None

    def __post_init__(self) -> None:
        expected = self.timestamps.size
        for name in ("motion", "roi_motion", "audio", "temporal", "combined"):
            if getattr(self, name).size != expected:
                raise ValueError(f"{name} must have one item per timestamp")
        for name in ("motion_spread", "serve"):
            values = getattr(self, name)
            if values is None:
                object.__setattr__(self, name, np.zeros(expected, dtype=np.float64))
            elif values.size != expected:
                raise ValueError(f"{name} must have one item per timestamp")


class RallyScorer:
    """Weighted heuristic scorer with robust, video-relative normalization."""

    def __init__(self, settings: DetectionSettings | None = None) -> None:
        self.settings = settings or DetectionSettings()
        self.settings.validate()

    def score(
        self,
        timestamps: Sequence[float] | np.ndarray,
        motion: Sequence[float] | np.ndarray,
        roi_motion: Sequence[float] | np.ndarray,
        audio: Sequence[float] | np.ndarray,
        *,
        motion_spread: Sequence[float] | np.ndarray | None = None,
        serve: Sequence[float] | np.ndarray | None = None,
    ) -> RallyScoreSeries:
        """Return normalized signals and a weighted 0-to-1 rally score."""

        times = _one_dimensional("timestamps", timestamps)
        if times.size > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if np.any(~np.isfinite(times)) or (times.size and times[0] < 0):
            raise ValueError("timestamps must be finite and non-negative")

        raw_motion = _signal("motion", motion, times.size)
        raw_roi = _signal("roi_motion", roi_motion, times.size)
        raw_audio = _signal("audio", audio, times.size)
        has_spread = motion_spread is not None
        has_serve = serve is not None
        raw_spread = (
            _signal("motion_spread", motion_spread, times.size)
            if motion_spread is not None
            else np.zeros(times.size, dtype=np.float64)
        )
        raw_serve = (
            _signal("serve", serve, times.size)
            if serve is not None
            else np.zeros(times.size, dtype=np.float64)
        )
        if not times.size:
            empty = np.empty(0, dtype=np.float64)
            return RallyScoreSeries(
                timestamps=times,
                motion=empty,
                roi_motion=empty,
                audio=empty,
                temporal=empty,
                combined=empty,
                motion_spread=empty,
                serve=empty,
            )

        motion_score = np.clip(
            robust_normalize(raw_motion) * self.settings.motion_sensitivity, 0.0, 1.0
        )
        roi_score = np.clip(
            robust_normalize(raw_roi) * self.settings.motion_sensitivity, 0.0, 1.0
        )
        audio_score = np.clip(
            robust_normalize(raw_audio) * self.settings.audio_sensitivity, 0.0, 1.0
        )
        # These two features already have stable semantic 0..1 scales.  Robust
        # per-video normalization would incorrectly turn the least-compact ball
        # retrieval in a no-rally video into a perfect distributed-play cue.
        spread_score = np.clip(
            np.nan_to_num(raw_spread, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0
        )
        serve_score = np.clip(
            np.nan_to_num(raw_serve, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0
        )

        sample_rate = _sample_rate(times, self.settings.analysis_fps)
        non_temporal_weight = (
            self.settings.motion_weight
            + self.settings.roi_motion_weight
            + self.settings.audio_weight
            + (self.settings.motion_spread_weight if has_spread else 0.0)
        )
        if non_temporal_weight > 0:
            instantaneous = (
                motion_score * self.settings.motion_weight
                + roi_score * self.settings.roi_motion_weight
                + audio_score * self.settings.audio_weight
                + spread_score
                * (self.settings.motion_spread_weight if has_spread else 0.0)
            ) / non_temporal_weight
        else:
            instantaneous = np.zeros_like(times)

        temporal_samples = max(1, round(self.settings.temporal_window * sample_rate))
        persistent_activity = moving_average(instantaneous, temporal_samples)
        # Repeated transients are useful confirmation, while a single clap or
        # shout should not make the temporal component dominate.
        transient_density = moving_average(audio_score, temporal_samples)
        temporal_score = np.clip(
            0.85 * persistent_activity + 0.15 * transient_density, 0.0, 1.0
        )

        combined_numerator = (
            motion_score * self.settings.motion_weight
            + roi_score * self.settings.roi_motion_weight
            + audio_score * self.settings.audio_weight
            + temporal_score * self.settings.temporal_weight
            + spread_score
            * (self.settings.motion_spread_weight if has_spread else 0.0)
            + serve_score * (self.settings.serve_weight if has_serve else 0.0)
        )
        effective_weight = (
            self.settings.motion_weight
            + self.settings.roi_motion_weight
            + self.settings.audio_weight
            + self.settings.temporal_weight
            + (self.settings.motion_spread_weight if has_spread else 0.0)
            + (self.settings.serve_weight if has_serve else 0.0)
        )
        combined = (
            combined_numerator / effective_weight
            if effective_weight > 0
            else np.zeros_like(times)
        )
        smoothing_samples = max(1, round(self.settings.smoothing_window * sample_rate))
        combined = np.clip(moving_average(combined, smoothing_samples), 0.0, 1.0)

        return RallyScoreSeries(
            timestamps=times,
            motion=motion_score,
            roi_motion=roi_score,
            audio=audio_score,
            temporal=temporal_score,
            combined=combined,
            motion_spread=spread_score,
            serve=serve_score,
        )


def score_rallies(
    timestamps: Sequence[float] | np.ndarray,
    motion: Sequence[float] | np.ndarray,
    roi_motion: Sequence[float] | np.ndarray,
    audio: Sequence[float] | np.ndarray,
    settings: DetectionSettings | None = None,
    *,
    motion_spread: Sequence[float] | np.ndarray | None = None,
    serve: Sequence[float] | np.ndarray | None = None,
) -> RallyScoreSeries:
    """Functional convenience wrapper around :class:`RallyScorer`."""

    return RallyScorer(settings).score(
        timestamps,
        motion,
        roi_motion,
        audio,
        motion_spread=motion_spread,
        serve=serve,
    )


def robust_normalize(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Normalize a noisy non-negative signal without letting outliers dominate."""

    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not signal.size:
        return signal.copy()
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    signal = np.maximum(signal, 0.0)

    baseline = float(np.percentile(signal, 20.0))
    upper = float(np.percentile(signal, 95.0))
    if upper - baseline <= _EPSILON:
        upper = float(np.max(signal))
    if upper - baseline <= _EPSILON:
        return np.zeros_like(signal)
    return np.clip((signal - baseline) / (upper - baseline), 0.0, 1.0)


def moving_average(values: Sequence[float] | np.ndarray, window_size: int) -> np.ndarray:
    """Centered moving average with edge padding and length preservation."""

    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if isinstance(window_size, bool) or int(window_size) != window_size or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    size = int(window_size)
    if signal.size == 0 or size == 1:
        return signal.copy()
    size = min(size, signal.size)
    left = (size - 1) // 2
    right = size - 1 - left
    padded = np.pad(signal, (left, right), mode="edge")
    kernel = np.full(size, 1.0 / size, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def _one_dimensional(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _signal(
    name: str, values: Sequence[float] | np.ndarray, expected_size: int
) -> np.ndarray:
    result = _one_dimensional(name, values)
    if result.size != expected_size:
        raise ValueError(f"{name} must have one item per timestamp")
    return result


def _sample_rate(timestamps: np.ndarray, fallback: float) -> float:
    if timestamps.size < 2:
        return fallback
    period = float(np.median(np.diff(timestamps)))
    return 1.0 / period if period > 0 else fallback


_EPSILON = np.finfo(np.float64).eps * 16


__all__ = [
    "RallyScoreSeries",
    "RallyScorer",
    "moving_average",
    "robust_normalize",
    "score_rallies",
]
