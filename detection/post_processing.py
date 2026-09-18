"""Convert a sampled rally-score signal into editable rally intervals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from config import DetectionSettings
from models import Rally


@dataclass(frozen=True)
class _RawSegment:
    start: float
    end: float
    confidence: float
    enabled: bool = True
    serve_confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


def segment_rallies(
    timestamps: Sequence[float] | np.ndarray,
    rally_scores: Sequence[float] | np.ndarray,
    settings: DetectionSettings | None = None,
    *,
    video_duration: float | None = None,
    motion_spread_scores: Sequence[float] | np.ndarray | None = None,
    serve_scores: Sequence[float] | np.ndarray | None = None,
) -> list[Rally]:
    """Segment normalized scores using temporal hysteresis.

    ``rally_threshold`` starts a candidate interval while the lower
    ``end_threshold`` ends it.  Requiring sustained activity/inactivity avoids
    flickering at either boundary.  Minimum duration is applied to the detected
    activity itself, before pre/post-roll is added.

    Args:
        timestamps: Strictly increasing sample timestamps in seconds.
        rally_scores: One normalized score per timestamp.
        settings: Detection configuration.  Defaults are used when omitted.
        video_duration: Optional source duration used to clamp post-roll.  When
            omitted, the inferred end of the final sample is used.
        motion_spread_scores: Optional bounded evidence that motion is spatially
            distributed around the court.
        serve_scores: Optional bounded serve-start likelihood.  When both
            optional signals are supplied, compact candidates without a serve
            cue are retained but disabled as likely ball retrievals.
    """

    config = settings or DetectionSettings()
    config.validate()
    times, scores, sample_period, available_end = _prepare_samples(
        timestamps, rally_scores, video_duration
    )
    if times.size == 0:
        return []

    spread = _optional_signal(
        "motion_spread_scores", motion_spread_scores, times.size
    )
    serve = _optional_signal("serve_scores", serve_scores, times.size)

    segments: list[_RawSegment] = []
    active = False
    armed = True
    candidate_start: float | None = None
    rally_start: float | None = None
    inactive_start: float | None = None
    reset_start: float | None = None

    for timestamp, score in zip(times, scores, strict=True):
        t = float(timestamp)
        value = float(score)

        if active:
            assert rally_start is not None
            # A continuously high signal should not create an arbitrarily long
            # clip or a chain of fake back-to-back rallies.  Close at the cap
            # and wait for a real inactive period before re-arming.
            if t - rally_start + sample_period >= config.max_rally_duration:
                end = min(rally_start + config.max_rally_duration, available_end)
                _append_if_long_enough(
                    segments, times, scores, rally_start, end, config.min_rally_duration
                )
                active = False
                armed = False
                rally_start = None
                candidate_start = None
                inactive_start = None
                reset_start = t if value < config.end_threshold else None
                continue

            if value < config.end_threshold:
                if inactive_start is None:
                    inactive_start = t
                inactive_span = t - inactive_start + sample_period
                if inactive_span + _EPSILON >= config.end_inactive_duration:
                    _append_if_long_enough(
                        segments,
                        times,
                        scores,
                        rally_start,
                        inactive_start,
                        config.min_rally_duration,
                    )
                    active = False
                    armed = True
                    rally_start = None
                    candidate_start = None
                    inactive_start = None
            else:
                inactive_start = None
            continue

        if not armed:
            if value < config.end_threshold:
                if reset_start is None:
                    reset_start = t
                if t - reset_start + sample_period + _EPSILON >= config.end_inactive_duration:
                    armed = True
                    reset_start = None
            else:
                reset_start = None
            continue

        if value >= config.rally_threshold:
            if candidate_start is None:
                candidate_start = t
            active_span = t - candidate_start + sample_period
            if active_span + _EPSILON >= config.start_active_duration:
                active = True
                rally_start = candidate_start
                candidate_start = None
                inactive_start = None
        else:
            candidate_start = None

    if active and rally_start is not None:
        # At EOF, retain an observed trailing inactive run as useful evidence of
        # the end even if the full inactivity duration was not available.
        end = inactive_start if inactive_start is not None else available_end
        _append_if_long_enough(
            segments, times, scores, rally_start, end, config.min_rally_duration
        )

    if config.retrieval_filter_enabled and spread is not None and serve is not None:
        segments = [
            _classify_candidate(segment, times, spread, serve, config)
            for segment in segments
        ]

    merged = _merge_raw_segments(
        segments,
        max_gap=config.merge_gap,
        max_duration=config.max_rally_duration,
    )
    padded = [
        Rally(
            start_time=max(0.0, segment.start - config.pre_roll),
            end_time=min(available_end, segment.end + config.post_roll),
            confidence=segment.confidence,
            enabled=segment.enabled,
            serve_confidence=segment.serve_confidence,
        )
        for segment in merged
        if min(available_end, segment.end + config.post_roll)
        > max(0.0, segment.start - config.pre_roll)
    ]

    # Padding can make otherwise separate intervals overlap.  Coalescing those
    # intervals prevents duplicate frames in the final exported video.
    return merge_rallies(padded, max_gap=0.0)


def merge_rallies(
    rallies: Iterable[Rally],
    max_gap: float,
    *,
    max_duration: float | None = None,
) -> list[Rally]:
    """Merge overlapping or nearby rallies without mutating the inputs.

    Confidence is averaged by duration.  Disabled rallies remain independent
    from enabled ones so a user's export choice is never silently lost.
    """

    gap = float(max_gap)
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("max_gap must be a finite non-negative number")
    duration_limit: float | None = None
    if max_duration is not None:
        duration_limit = float(max_duration)
        if not math.isfinite(duration_limit) or duration_limit <= 0:
            raise ValueError("max_duration must be a finite positive number")

    ordered = sorted(
        (Rally.from_dict(rally.to_dict()) for rally in rallies),
        key=lambda item: (item.start_time, item.end_time),
    )
    if not ordered:
        return []

    result: list[Rally] = [ordered[0]]
    for current in ordered[1:]:
        previous = result[-1]
        merged_duration = max(previous.end_time, current.end_time) - min(
            previous.start_time, current.start_time
        )
        can_merge = (
            previous.enabled == current.enabled
            and current.start_time - previous.end_time <= gap + _EPSILON
            and (duration_limit is None or merged_duration <= duration_limit + _EPSILON)
        )
        if not can_merge:
            result.append(current)
            continue

        total_evidence = previous.duration + current.duration
        confidence = (
            previous.confidence * previous.duration
            + current.confidence * current.duration
        ) / total_evidence
        result[-1] = Rally(
            start_time=min(previous.start_time, current.start_time),
            end_time=max(previous.end_time, current.end_time),
            confidence=min(1.0, max(0.0, confidence)),
            enabled=previous.enabled,
            serve_confidence=max(
                previous.serve_confidence, current.serve_confidence
            ),
        )
    return result


def post_process_scores(
    rally_scores: Sequence[float] | np.ndarray,
    timestamps: Sequence[float] | np.ndarray,
    settings: DetectionSettings | None = None,
    *,
    video_duration: float | None = None,
    motion_spread_scores: Sequence[float] | np.ndarray | None = None,
    serve_scores: Sequence[float] | np.ndarray | None = None,
) -> list[Rally]:
    """Compatibility wrapper with score-first argument ordering."""

    return segment_rallies(
        timestamps,
        rally_scores,
        settings,
        video_duration=video_duration,
        motion_spread_scores=motion_spread_scores,
        serve_scores=serve_scores,
    )


# Descriptive alias for callers that prefer the score-first API.
segments_from_scores = post_process_scores


def _prepare_samples(
    timestamps: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    video_duration: float | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    times = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if times.ndim != 1 or values.ndim != 1:
        raise ValueError("timestamps and rally_scores must be one-dimensional")
    if times.size != values.size:
        raise ValueError("timestamps and rally_scores must contain the same number of items")
    if times.size == 0:
        if video_duration is not None and (
            not math.isfinite(float(video_duration)) or float(video_duration) < 0
        ):
            raise ValueError("video_duration must be a finite non-negative number")
        return times, values, 0.0, float(video_duration or 0.0)
    if not np.all(np.isfinite(times)) or times[0] < 0:
        raise ValueError("timestamps must be finite and non-negative")
    if times.size > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    # A corrupt score should behave as inactivity rather than poisoning every
    # later operation with NaN.  Scores outside the normalized range are
    # clipped, allowing custom detector plugins to be slightly imperfect.
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)

    if times.size > 1:
        sample_period = float(np.median(np.diff(times)))
    else:
        sample_period = 0.0
    if sample_period <= 0:
        sample_period = 1e-6
    inferred_end = float(times[-1] + sample_period)

    if video_duration is None:
        end = inferred_end
    else:
        end = float(video_duration)
        if not math.isfinite(end) or end < 0:
            raise ValueError("video_duration must be a finite non-negative number")
        if end + _EPSILON < times[-1]:
            raise ValueError("video_duration cannot be earlier than the last timestamp")
    return times, values, sample_period, end


def _append_if_long_enough(
    output: list[_RawSegment],
    times: np.ndarray,
    scores: np.ndarray,
    start: float,
    end: float,
    minimum_duration: float,
) -> None:
    if end - start + _EPSILON < minimum_duration:
        return
    mask = (times >= start) & (times < end)
    confidence = float(np.mean(scores[mask])) if np.any(mask) else 0.0
    output.append(
        _RawSegment(start=start, end=end, confidence=min(1.0, max(0.0, confidence)))
    )


def _optional_signal(
    name: str,
    values: Sequence[float] | np.ndarray | None,
    expected_size: int,
) -> np.ndarray | None:
    if values is None:
        return None
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if signal.size != expected_size:
        raise ValueError(f"{name} must have one item per timestamp")
    return np.clip(
        np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
    )


def _classify_candidate(
    segment: _RawSegment,
    times: np.ndarray,
    spread: np.ndarray,
    serve: np.ndarray,
    config: DetectionSettings,
) -> _RawSegment:
    """Disable only high-confidence compact/retrieval-like candidates.

    This is deliberately a fail-open classifier.  A strong distributed-motion
    sample or a plausible serve cue keeps the candidate enabled.  Suppressed
    intervals remain in the review UI as disabled rallies, so an unusual real
    point is one click away from recovery and never silently discarded.
    """

    body_mask = (times >= segment.start) & (times < segment.end)
    segment_spread = spread[body_mask]
    if segment_spread.size == 0:
        return segment

    # The raw segment normally begins a few samples after the physical serve.
    # Look back by the configured hold duration, but only inspect the early
    # portion of the candidate so unrelated late sounds cannot confirm it.
    serve_end = min(segment.end, segment.start + config.serve_hold_duration)
    serve_mask = (
        (times >= max(0.0, segment.start - config.serve_hold_duration))
        & (times <= serve_end)
    )
    serve_confidence = float(np.max(serve[serve_mask])) if np.any(serve_mask) else 0.0
    spread_typical = float(np.percentile(segment_spread, 75.0))
    sample_period = (
        float(np.median(np.diff(times))) if times.size > 1 else 1.0 / config.analysis_fps
    )
    strong_samples_needed = max(1, math.ceil(0.25 / max(sample_period, _EPSILON)))
    strong_sample_count = int(
        np.count_nonzero(segment_spread >= config.strong_play_spread_threshold)
    )

    serve_confirmed = serve_confidence >= config.serve_confirmation_threshold
    # A quarter-second requirement ignores a single camera bump while retaining
    # the brief distributed reaction that follows a fast ace.
    strong_play = strong_sample_count >= strong_samples_needed
    compact_activity = spread_typical <= config.retrieval_spread_threshold
    enabled = not (compact_activity and not serve_confirmed and not strong_play)
    return _RawSegment(
        start=segment.start,
        end=segment.end,
        confidence=segment.confidence,
        enabled=enabled,
        serve_confidence=serve_confidence,
    )


def _merge_raw_segments(
    segments: Sequence[_RawSegment], max_gap: float, max_duration: float
) -> list[_RawSegment]:
    if not segments:
        return []
    result = [segments[0]]
    for current in segments[1:]:
        previous = result[-1]
        combined_duration = current.end - previous.start
        if (
            previous.enabled == current.enabled
            and current.start - previous.end <= max_gap + _EPSILON
            and combined_duration <= max_duration + _EPSILON
        ):
            evidence = previous.duration + current.duration
            confidence = (
                previous.confidence * previous.duration
                + current.confidence * current.duration
            ) / evidence
            result[-1] = _RawSegment(
                previous.start,
                current.end,
                confidence,
                enabled=previous.enabled,
                serve_confidence=max(
                    previous.serve_confidence, current.serve_confidence
                ),
            )
        else:
            result.append(current)
    return result


_EPSILON = 1e-9


__all__ = [
    "merge_rallies",
    "post_process_scores",
    "segment_rallies",
    "segments_from_scores",
]
