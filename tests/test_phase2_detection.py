from __future__ import annotations

import numpy as np
import pytest

from config import DetectionSettings
from detection.motion_detector import _motion_spread_metric
from detection.post_processing import merge_rallies, segment_rallies
from detection.rally_scoring import RallyScorer
from detection.serve_detector import ServeDetector
from models import Rally


def _segmentation_settings(**overrides: object) -> DetectionSettings:
    values: dict[str, object] = {
        "analysis_fps": 4.0,
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
    values.update(overrides)
    return DetectionSettings(**values)


def _candidate_signals() -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(0.0, 8.0, 0.25)
    scores = np.full(timestamps.size, 0.1)
    scores[(timestamps >= 2.0) & (timestamps < 5.0)] = 0.85
    return timestamps, scores


def test_motion_spread_is_translation_invariant_for_one_player() -> None:
    left = np.zeros((360, 640), dtype=np.uint8)
    boundary = np.zeros_like(left)
    left[100:250, 50:105] = 255
    # This location straddled a cell boundary and scored more than twice as
    # highly with the original fixed-grid implementation.
    boundary[100:250, 180:235] = 255

    assert _motion_spread_metric(left, 12) == pytest.approx(
        _motion_spread_metric(boundary, 12), abs=1e-12
    )


def test_motion_spread_rewards_separated_players_over_one_silhouette() -> None:
    compact = np.zeros((360, 640), dtype=np.uint8)
    compact[100:250, 180:235] = 255
    distributed = np.zeros_like(compact)
    for y, x in ((60, 80), (60, 500), (250, 80), (250, 500)):
        distributed[y : y + 35, x : x + 25] = 255

    compact_score = _motion_spread_metric(compact, 12)
    distributed_score = _motion_spread_metric(distributed, 12)

    assert compact_score < 0.30
    assert distributed_score > 0.80
    assert distributed_score > compact_score * 3


def test_visual_serve_cue_does_not_require_audio() -> None:
    timestamps = np.arange(0.0, 4.0, 0.125)
    roi = np.zeros(timestamps.size)
    spread = np.zeros(timestamps.size)
    audio = np.zeros(timestamps.size)
    onset = int(1.5 / 0.125)
    roi[onset : onset + 9] = 0.9
    spread[onset : onset + 9] = 0.75

    result = ServeDetector().score(timestamps, roi, spread, audio)

    assert float(np.max(result.likelihood)) > 0.60
    assert float(np.max(result.context)) == pytest.approx(
        float(np.max(result.likelihood))
    )


def test_audio_spike_alone_cannot_create_a_strong_serve_cue() -> None:
    timestamps = np.arange(0.0, 4.0, 0.125)
    zeros = np.zeros(timestamps.size)
    audio = zeros.copy()
    audio[len(audio) // 2] = 1.0

    result = ServeDetector().score(timestamps, zeros, zeros, audio)

    assert float(np.max(result.likelihood)) <= 0.12 + 1e-12


def test_compact_motion_onset_does_not_confirm_itself_as_a_serve() -> None:
    settings = DetectionSettings()
    timestamps = np.arange(0.0, 4.0, 0.125)
    roi = np.zeros(timestamps.size)
    spread = np.zeros(timestamps.size)
    audio = np.zeros(timestamps.size)
    onset = int(1.5 / 0.125)
    roi[onset : onset + 9] = 0.95
    spread[onset : onset + 9] = 0.20
    audio[onset] = 0.50

    result = ServeDetector(settings).score(timestamps, roi, spread, audio)

    assert float(np.max(result.likelihood)) < settings.serve_confirmation_threshold


def test_rally_scorer_combines_bounded_spread_and_serve_without_renormalizing() -> None:
    settings = DetectionSettings(
        motion_weight=0.0,
        roi_motion_weight=0.0,
        motion_spread_weight=0.5,
        audio_weight=0.0,
        temporal_weight=0.0,
        serve_weight=0.5,
        smoothing_window=0.01,
    )
    timestamps = np.asarray([0.0, 0.5])
    zeros = np.zeros(2)

    result = RallyScorer(settings).score(
        timestamps,
        zeros,
        zeros,
        zeros,
        motion_spread=np.asarray([0.2, 0.8]),
        serve=np.asarray([0.4, 0.6]),
    )

    assert result.motion_spread.tolist() == pytest.approx([0.2, 0.8])
    assert result.serve.tolist() == pytest.approx([0.4, 0.6])
    assert result.combined.tolist() == pytest.approx([0.3, 0.7])


def test_compact_candidate_without_serve_is_disabled_not_deleted() -> None:
    timestamps, scores = _candidate_signals()
    spread = np.zeros_like(scores)
    spread[(timestamps >= 2.0) & (timestamps < 5.0)] = 0.12

    rallies = segment_rallies(
        timestamps,
        scores,
        _segmentation_settings(),
        video_duration=8.0,
        motion_spread_scores=spread,
        serve_scores=np.zeros_like(scores),
    )

    assert len(rallies) == 1
    assert rallies[0].enabled is False
    assert rallies[0].serve_confidence == 0.0


def test_visual_serve_keeps_otherwise_compact_candidate_enabled() -> None:
    timestamps, scores = _candidate_signals()
    spread = np.zeros_like(scores)
    spread[(timestamps >= 2.0) & (timestamps < 5.0)] = 0.12
    serve = np.zeros_like(scores)
    serve[np.where(timestamps == 2.0)[0][0]] = 0.72

    rallies = segment_rallies(
        timestamps,
        scores,
        _segmentation_settings(),
        video_duration=8.0,
        motion_spread_scores=spread,
        serve_scores=serve,
    )

    assert len(rallies) == 1
    assert rallies[0].enabled is True
    assert rallies[0].serve_confidence == pytest.approx(0.72)


def test_distributed_play_overrides_missing_serve_and_audio() -> None:
    timestamps, scores = _candidate_signals()
    spread = np.zeros_like(scores)
    active_indices = np.where((timestamps >= 2.0) & (timestamps < 5.0))[0]
    spread[active_indices] = 0.18
    spread[active_indices[len(active_indices) // 2]] = 0.8

    rallies = segment_rallies(
        timestamps,
        scores,
        _segmentation_settings(),
        video_duration=8.0,
        motion_spread_scores=spread,
        serve_scores=np.zeros_like(scores),
    )

    assert len(rallies) == 1
    assert rallies[0].enabled is True


def test_one_frame_camera_bump_does_not_override_compact_retrieval() -> None:
    timestamps = np.arange(0.0, 8.0, 0.125)
    scores = np.full(timestamps.size, 0.1)
    active = np.where((timestamps >= 2.0) & (timestamps < 5.0))[0]
    scores[active] = 0.85
    spread = np.zeros_like(scores)
    spread[active] = 0.15
    spread[active[len(active) // 2]] = 0.9

    rallies = segment_rallies(
        timestamps,
        scores,
        _segmentation_settings(analysis_fps=8.0),
        video_duration=8.0,
        motion_spread_scores=spread,
        serve_scores=np.zeros_like(scores),
    )

    assert len(rallies) == 1
    assert rallies[0].enabled is False


def test_legacy_segmentation_without_context_remains_enabled() -> None:
    timestamps, scores = _candidate_signals()

    rallies = segment_rallies(
        timestamps, scores, _segmentation_settings(), video_duration=8.0
    )

    assert len(rallies) == 1
    assert rallies[0].enabled is True


def test_merge_preserves_strongest_serve_confidence() -> None:
    rallies = merge_rallies(
        [
            Rally(1.0, 2.0, 0.7, True, 0.35),
            Rally(2.2, 3.0, 0.8, True, 0.75),
        ],
        max_gap=0.25,
    )

    assert len(rallies) == 1
    assert rallies[0].serve_confidence == pytest.approx(0.75)
