from __future__ import annotations

import numpy as np
import pytest

from config import DetectionSettings
from detection.post_processing import merge_rallies, segment_rallies
from models import Rally


def _settings(**overrides: float) -> DetectionSettings:
    defaults = {
        "rally_threshold": 0.6,
        "end_threshold": 0.4,
        "start_active_duration": 0.5,
        "end_inactive_duration": 0.5,
        "min_rally_duration": 1.0,
        "max_rally_duration": 20.0,
        "pre_roll": 0.0,
        "post_roll": 0.0,
        "merge_gap": 0.0,
    }
    defaults.update(overrides)
    return DetectionSettings(**defaults)


def _signal(*active_ranges: tuple[float, float], duration: float = 12.0):
    timestamps = np.arange(0.0, duration, 0.25)
    scores = np.full(timestamps.size, 0.1)
    for start, end in active_ranges:
        scores[(timestamps >= start) & (timestamps < end)] = 0.85
    return timestamps, scores


def test_hysteresis_keeps_a_short_dip_inside_one_rally() -> None:
    timestamps, scores = _signal((2.0, 6.0))
    scores[(timestamps >= 3.0) & (timestamps < 3.25)] = 0.1

    rallies = segment_rallies(
        timestamps, scores, _settings(), video_duration=12.0
    )

    assert len(rallies) == 1
    assert rallies[0].start_time == pytest.approx(2.0)
    assert rallies[0].end_time == pytest.approx(6.0)
    assert 0.0 <= rallies[0].confidence <= 1.0


def test_short_detection_is_removed_before_roll_is_added() -> None:
    timestamps, scores = _signal((3.0, 3.75))

    rallies = segment_rallies(
        timestamps,
        scores,
        _settings(min_rally_duration=1.0, pre_roll=2.0, post_roll=2.0),
        video_duration=12.0,
    )

    assert rallies == []


def test_pre_and_post_roll_are_added_and_clamped_to_video() -> None:
    timestamps, scores = _signal((0.25, 2.0), (9.5, 11.75))

    rallies = segment_rallies(
        timestamps,
        scores,
        _settings(pre_roll=0.75, post_roll=1.25),
        video_duration=12.0,
    )

    assert [(r.start_time, r.end_time) for r in rallies] == pytest.approx(
        [(0.0, 3.25), (8.75, 12.0)]
    )


def test_nearby_rallies_merge_but_larger_gap_does_not() -> None:
    timestamps, scores = _signal((1.0, 3.0), (3.75, 5.5), (7.0, 9.0))

    rallies = segment_rallies(
        timestamps,
        scores,
        _settings(merge_gap=1.0),
        video_duration=12.0,
    )

    assert [(r.start_time, r.end_time) for r in rallies] == pytest.approx(
        [(1.0, 5.5), (7.0, 9.0)]
    )


def test_continuous_activity_is_capped_and_requires_inactivity_to_rearm() -> None:
    timestamps, scores = _signal((1.0, 10.0))

    rallies = segment_rallies(
        timestamps,
        scores,
        _settings(max_rally_duration=3.0, min_rally_duration=0.5),
        video_duration=12.0,
    )

    assert len(rallies) == 1
    assert rallies[0].start_time == pytest.approx(1.0)
    assert rallies[0].end_time == pytest.approx(4.0)


def test_merge_rallies_uses_duration_weighted_confidence() -> None:
    merged = merge_rallies(
        [Rally(1.0, 2.0, 0.2), Rally(2.5, 4.5, 0.8)], max_gap=0.5
    )

    assert len(merged) == 1
    assert merged[0].start_time == 1.0
    assert merged[0].end_time == 4.5
    assert merged[0].confidence == pytest.approx(0.6)


def test_rally_serialization_accepts_annotation_keys() -> None:
    rally = Rally.from_dict({"start": 2.5, "end": 4.0})

    assert rally.duration == 1.5
    assert rally.to_annotation_dict() == {"start": 2.5, "end": 4.0}
    assert Rally.from_dict(rally.to_dict()) == rally


@pytest.mark.parametrize(
    "start,end,confidence",
    [(-1.0, 2.0, 0.5), (2.0, 2.0, 0.5), (1.0, 2.0, 1.1)],
)
def test_rally_rejects_invalid_values(
    start: float, end: float, confidence: float
) -> None:
    with pytest.raises(ValueError):
        Rally(start, end, confidence)

