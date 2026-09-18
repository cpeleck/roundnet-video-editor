from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from training.feedback import (
    ANNOTATION_ALL_RALLIES,
    ANNOTATION_HIGHLIGHTS_ONLY,
    FEEDBACK_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    LABEL_IGNORED,
    LABEL_NON_RALLY,
    LABEL_RALLY,
    dense_features_from_result,
    fingerprint_source,
    make_sample_labels,
    save_feedback,
)


@dataclass
class _SignalResult:
    timestamps: np.ndarray
    motion_scores: np.ndarray
    roi_motion_scores: np.ndarray
    audio_scores: np.ndarray
    temporal_scores: np.ndarray
    rally_scores: np.ndarray
    serve_likelihood_scores: np.ndarray


def _signals() -> _SignalResult:
    timestamps = np.arange(0.0, 6.0, 0.5, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, timestamps.size, dtype=np.float64)
    return _SignalResult(
        timestamps=timestamps,
        motion_scores=ramp,
        roi_motion_scores=ramp[::-1].copy(),
        audio_scores=np.square(ramp),
        temporal_scores=np.sqrt(ramp),
        rally_scores=np.clip(ramp * 1.2, 0.0, 1.0),
        serve_likelihood_scores=np.roll(ramp, 1),
    )


def test_dense_features_collects_stable_canonical_columns() -> None:
    dense = dense_features_from_result(_signals())

    assert dense.names == (
        "motion",
        "roi_motion",
        "audio",
        "temporal",
        "serve_likelihood",
        "rally_score",
    )
    assert dense.values.shape == (12, 6)
    assert dense.values.dtype == np.float32
    np.testing.assert_allclose(dense.timestamps, np.arange(0.0, 6.0, 0.5))


def test_dense_features_accepts_serialized_signal_mapping() -> None:
    dense = dense_features_from_result(
        {
            "signals": {
                "timestamps": [0.0, 0.5],
                "motion": [0.1, 0.2],
                "motion_spread": [0.2, 0.3],
                "serve": [0.8, 0.1],
                "rally_score": [0.4, 0.7],
            }
        }
    )

    assert dense.names == ("motion", "motion_spread", "serve_likelihood", "rally_score")
    assert dense.values.shape == (2, 4)


def test_sample_labels_distinguish_disabled_from_explicit_rejection() -> None:
    timestamps = np.arange(0.0, 6.0, 0.5)
    labels = make_sample_labels(
        timestamps,
        [
            {"start": 1.0, "end": 3.0, "enabled": True},
            {"start": 4.0, "end": 5.0, "enabled": False},
        ],
        rejected_detections=[{"start": 0.0, "end": 0.75}],
        annotation_completeness=ANNOTATION_ALL_RALLIES,
        boundary_ignore_seconds=0.25,
    )

    assert labels.tolist() == [
        LABEL_NON_RALLY,
        LABEL_NON_RALLY,
        LABEL_IGNORED,
        LABEL_RALLY,
        LABEL_RALLY,
        LABEL_RALLY,
        LABEL_IGNORED,
        LABEL_NON_RALLY,
        LABEL_IGNORED,
        LABEL_IGNORED,
        LABEL_NON_RALLY,
        LABEL_NON_RALLY,
    ]


def test_partial_annotations_only_make_explicit_rejections_negative() -> None:
    timestamps = np.arange(0.0, 5.0, 1.0)
    labels = make_sample_labels(
        timestamps,
        [{"start": 2.0, "end": 4.0, "enabled": True}],
        rejected_detections=[{"start": 0.0, "end": 1.0}],
        annotation_completeness=ANNOTATION_HIGHLIGHTS_ONLY,
        boundary_ignore_seconds=0.0,
    )

    assert labels.tolist() == [
        LABEL_NON_RALLY,
        LABEL_IGNORED,
        LABEL_RALLY,
        LABEL_RALLY,
        LABEL_IGNORED,
    ]


def test_source_fingerprint_is_content_based(tmp_path: Path) -> None:
    first = tmp_path / "first.mov"
    second = tmp_path / "renamed.mov"
    payload = (b"roundnet-video-bytes" * 200) + b"tail"
    first.write_bytes(payload)
    second.write_bytes(payload)

    first_fingerprint = fingerprint_source(first, sample_size=64)
    second_fingerprint = fingerprint_source(second, sample_size=64)

    assert first_fingerprint["digest"] == second_fingerprint["digest"]
    assert first_fingerprint["strategy"] == "head_tail"
    assert first_fingerprint["size_bytes"] == len(payload)


def test_save_feedback_writes_versioned_json_and_compressed_dense_arrays(
    tmp_path: Path,
) -> None:
    source = tmp_path / "game.mov"
    source.write_bytes(b"synthetic source" * 128)
    metadata_path = tmp_path / "game.labels.json"
    signals = _signals()

    result = save_feedback(
        metadata_path,
        source_path=source,
        duration_seconds=6.0,
        roi=(0.1, 0.2, 0.7, 0.6),
        initial_predictions=[
            {"start_time": 1.0, "end_time": 3.0, "confidence": 0.8},
            {"start_time": 4.0, "end_time": 5.0, "confidence": 0.6},
        ],
        final_intervals=[
            {"start": 1.1, "end": 2.9, "enabled": True},
            {"start": 4.0, "end": 5.0, "enabled": False},
        ],
        rejected_detections=[{"start": 4.0, "end": 5.0, "reason": "user_deleted"}],
        annotation_completeness=ANNOTATION_ALL_RALLIES,
        signal_data=signals,
        detection_settings={"analysis_fps": 2.0, "debug_mode": False},
        boundary_ignore_seconds=0.0,
        created_utc="2026-09-15T12:00:00+00:00",
    )

    assert result.metadata_path == metadata_path
    assert result.features_path == tmp_path / "game.labels.npz"
    assert result.sample_count == signals.timestamps.size
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == FEEDBACK_SCHEMA_VERSION
    assert payload["annotation_completeness"] == ANNOTATION_ALL_RALLIES
    assert payload["source"]["id"] == result.source_id
    assert payload["source"]["fingerprint"]["digest"] == result.source_id
    assert payload["initial_predictions"][0]["prediction_id"] == "prediction-0001"
    assert payload["final_enabled_intervals"] == [{"end": 2.9, "start": 1.1}]
    assert payload["final_disabled_intervals"] == [{"end": 5.0, "start": 4.0}]
    assert payload["rejected_detections"][0]["source_prediction_id"] == "prediction-0002"
    assert "not asserted play-core boundaries" in payload["interval_semantics"][
        "final_enabled_intervals"
    ]
    assert payload["rallies"] == [{"end": 2.9, "start": 1.1}]

    assert result.features_path is not None
    with np.load(result.features_path, allow_pickle=False) as artifact:
        assert int(artifact["schema_version"]) == FEATURE_SCHEMA_VERSION
        assert str(artifact["source_id"]) == result.source_id
        assert artifact["features"].shape == (12, 6)
        assert artifact["features"].dtype == np.float32
        assert artifact["labels"].dtype == np.int8
        assert set(np.unique(artifact["labels"])).issubset({-1, 0, 1})
        assert artifact["feature_names"].tolist() == payload["features"]["feature_names"]


def test_save_feedback_accepts_a_reviewed_zero_rally_video(tmp_path: Path) -> None:
    source = tmp_path / "no-rallies.mp4"
    source.write_bytes(b"no rallies here")
    metadata_path = tmp_path / "no-rallies.json"

    result = save_feedback(
        metadata_path,
        source_path=source,
        duration_seconds=6.0,
        roi=None,
        initial_predictions=[],
        final_intervals=[],
        annotation_completeness=ANNOTATION_ALL_RALLIES,
        signal_data=_signals(),
        boundary_ignore_seconds=0.25,
    )

    assert result.positive_count == 0
    assert result.negative_count == 12
    assert result.ignored_count == 0
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["rallies"] == []
    assert payload["final_enabled_intervals"] == []
    assert payload["features"]["label_counts"] == {
        "ignored": 0,
        "non_rally": 12,
        "rally": 0,
    }


def test_feedback_rejects_misaligned_signals(tmp_path: Path) -> None:
    result = _signals()
    result.audio_scores = result.audio_scores[:-1]

    with pytest.raises(ValueError, match="one item per timestamp"):
        dense_features_from_result(result)


def test_pending_rallies_need_review_before_becoming_positive_labels() -> None:
    from models import Rally

    times = np.arange(6, dtype=float)
    pending = Rally(1, 3)
    reviewed = Rally(4, 6, reviewed=True)
    labels = make_sample_labels(times, [pending, reviewed], boundary_ignore_seconds=0)
    assert labels.tolist() == [-1, -1, -1, -1, 1, 1]
    complete = make_sample_labels(times, [pending, reviewed], boundary_ignore_seconds=0,
                                  annotation_completeness=ANNOTATION_ALL_RALLIES)
    assert complete.tolist() == [0, 1, 1, 0, 1, 1]


def test_feedback_protects_original_video_and_json_from_sidecar_collision(tmp_path: Path) -> None:
    source = tmp_path / "original.mp4"
    source.write_bytes(b"important source footage")
    kwargs = dict(source_path=source, duration_seconds=6, roi=None,
                  initial_predictions=[], final_intervals=[], signal_data=_signals())
    with pytest.raises(ValueError, match="separately from the source"):
        save_feedback(source, **kwargs)
    metadata = tmp_path / "labels.json"
    for conflict in (source, metadata):
        with pytest.raises(ValueError, match="features separately"):
            save_feedback(metadata, features_path=conflict, **kwargs)
    assert source.read_bytes() == b"important source footage"
    assert not metadata.exists()


def test_feature_sidecars_are_saved_next_to_metadata(tmp_path: Path) -> None:
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    separate = tmp_path / "other"
    separate.mkdir()
    with pytest.raises(ValueError, match="beside"):
        save_feedback(tmp_path / "labels.json", source_path=source, duration_seconds=6,
                      roi=None, initial_predictions=[], final_intervals=[], signal_data=_signals(),
                      features_path=separate / "features.npz")


def test_feedback_preserves_stable_prediction_identity_and_review_state(tmp_path: Path) -> None:
    from models import Rally

    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    rally = Rally(1, 3, reviewed=True)
    pending = Rally(4, 5)
    result = save_feedback(tmp_path / "labels.json", source_path=source, duration_seconds=6,
                           roi=None, initial_predictions=[rally], final_intervals=[rally, pending],
                           rejected_detections=[rally], signal_data=_signals())
    payload = json.loads(result.metadata_path.read_text())
    assert payload["initial_predictions"][0]["prediction_id"] == rally.rally_id
    assert payload["rejected_detections"][0]["source_prediction_id"] == rally.rally_id
    assert payload["reviewed_intervals"][0]["reviewed"] is True
    assert payload["reviewed_intervals"][1]["reviewed"] is False
