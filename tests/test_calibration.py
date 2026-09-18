"""Behavioral checks for local learning, held-out evaluation and provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.calibration import (
    evaluate_intervals, import_source_intervals, load_profile,
    predict_profile, train_from_feedback,
)
from training.feedback import ANNOTATION_ALL_RALLIES, make_sample_labels, save_feedback


def _recording(tmp_path: Path, index: int, *, complete: bool = True, duplicate_source: bool = False) -> tuple[Path, dict]:
    source = tmp_path / f"game-{index}.mp4"
    source.write_bytes(b"same source" if duplicate_source else f"original video {index}".encode())
    times = np.arange(0, 20, 0.125)
    playing = ((times >= 3) & (times < 7)) | ((times >= 11) & (times < 15))
    retrieving = (times >= 17) & (times < 19)
    rng = np.random.default_rng(index)
    motion = np.clip(0.08 + playing * 0.75 + retrieving * 0.7 + rng.normal(0, 0.02, times.size), 0, 1)
    spread = np.clip(0.02 + playing * 0.85 + rng.normal(0, 0.01, times.size), 0, 1)
    signals = {
        "timestamps": times, "motion_scores": motion,
        "motion_spread_scores": spread,
        "serve_scores": np.where((times >= 3) & (times < 3.5), 0.8, 0.0),
        # Intentionally false-positive-prone initial heuristic to compare.
        "rally_scores": np.where(playing | retrieving, 0.9, 0.1),
    }
    path = tmp_path / f"game-{index}.json"
    save_feedback(
        path, source_path=source, duration_seconds=20, roi=None,
        initial_predictions=[{"start": 3, "end": 7, "serve_confidence": 0.8}],
        final_intervals=[{"start": 3, "end": 7}, {"start": 11, "end": 15}],
        rejected_detections=[{"start": 17, "end": 19, "serve_confidence": 0.1}],
        annotation_completeness=ANNOTATION_ALL_RALLIES if complete else "unknown",
        signal_data=signals,
        detection_settings={"pre_roll": 0, "post_roll": 0, "retrieval_filter_enabled": False,
                            "min_rally_duration": 0.5, "merge_gap": 0.1},
        boundary_ignore_seconds=0.125,
    )
    return path, signals


def test_local_profile_uses_disjoint_recording_folds_and_reduces_retrieval_false_hits(tmp_path: Path) -> None:
    recordings = [_recording(tmp_path, i) for i in range(3)]
    destination = tmp_path / "profile.json"
    report = train_from_feedback([p for p, _ in recordings], destination)
    assert report["validation_method"] == "leave_one_source_recording_out"
    assert report["recording_count"] == 3
    for fold in report["folds"]:
        assert fold["held_out_source_id"] not in fold["training_source_ids"]
        assert len(fold["training_source_ids"]) == 2
    assert report["learned_events"]["false_positives"] == 0
    assert report["heuristic_events"]["false_positives"] == 3
    assert report["learned_events"]["recall"] == 1
    assert report["learned_samples"]["balanced_accuracy"] > report["heuristic_samples"]["balanced_accuracy"]
    model = load_profile(destination)
    scores = predict_profile(model, recordings[0][1])
    assert scores.shape == recordings[0][1]["timestamps"].shape
    assert np.all((scores >= 0) & (scores <= 1))
    assert model["coefficients"][model["feature_names"].index("rally_score")] == 0
    assert "pickle" not in destination.read_text()


def test_duplicate_originals_do_not_count_as_independent_games(tmp_path: Path) -> None:
    paths = [_recording(tmp_path, i, duplicate_source=True)[0] for i in range(3)]
    with pytest.raises(ValueError, match="distinct original recordings"):
        train_from_feedback(paths, tmp_path / "profile.json")
    assert not (tmp_path / "profile.json").exists()


def test_feature_mismatch_and_non_finite_profile_are_rejected(tmp_path: Path) -> None:
    recordings = [_recording(tmp_path, i) for i in range(3)]
    destination = tmp_path / "profile.json"
    train_from_feedback([p for p, _ in recordings], destination)
    model = load_profile(destination)
    signals = dict(recordings[0][1], audio_scores=np.zeros(160))
    with pytest.raises(ValueError, match="feature mismatch"):
        predict_profile(model, signals)
    model["scale"][0] = 0
    destination.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="scale must be positive"):
        load_profile(destination)


def test_unconfirmed_legacy_background_cannot_become_training_negatives(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        path, _ = _recording(tmp_path, i)
        payload = json.loads(path.read_text())
        payload.pop("annotation_completeness_confirmed")
        payload["rejected_detections"] = []
        path.write_text(json.dumps(payload))
        paths.append(path)
    with pytest.raises(ValueError, match="reviewed samples per class"):
        train_from_feedback(paths, tmp_path / "profile.json")


def test_partial_annotations_train_but_do_not_claim_whole_video_event_accuracy(tmp_path: Path) -> None:
    paths = [_recording(tmp_path, i, complete=False)[0] for i in range(3)]
    report = train_from_feedback(paths, tmp_path / "profile.json")
    assert report["learned_events"] is None
    assert report["learned_samples"]["balanced_accuracy"] > 0.95
    assert any("Event metrics unavailable" in value for value in report["warnings"])


def test_cancellation_does_not_replace_existing_profile(tmp_path: Path) -> None:
    paths = [_recording(tmp_path, i)[0] for i in range(3)]
    destination = tmp_path / "profile.json"
    destination.write_text("existing profile")
    with pytest.raises(RuntimeError, match="cancelled"):
        train_from_feedback(paths, destination, cancelled=lambda: True)
    assert destination.read_text() == "existing profile"


def test_default_labels_ignore_unreviewed_background_and_configured_padding() -> None:
    times = np.arange(0, 10, 0.5)
    labels = make_sample_labels(times, [{"start": 2, "end": 8}], boundary_ignore_seconds=0,
                                pre_roll_seconds=1, post_roll_seconds=2)
    assert np.all(labels[times < 3] == -1)
    assert np.all(labels[(times >= 3) & (times < 6)] == 1)
    assert np.all(labels[times >= 6] == -1)


def test_serve_confidence_survives_all_feedback_provenance(tmp_path: Path) -> None:
    path, _ = _recording(tmp_path, 0)
    payload = json.loads(path.read_text())
    assert payload["initial_predictions"][0]["serve_confidence"] == 0.8
    assert payload["rejected_detections"][0]["serve_confidence"] == 0.1
    assert "serve_confidence" in payload["reviewed_intervals"][0]


def test_event_matching_is_one_to_one_and_reports_time_error() -> None:
    report = evaluate_intervals(
        [{"start": 1, "end": 4}, {"start": 1, "end": 4}, {"start": 9, "end": 11}],
        [{"start": 1.25, "end": 4.5}, {"start": 15, "end": 17}], 20,
    )
    assert report["true_positives"] == 1
    assert report["false_positives"] == 2
    assert report["false_negatives"] == 1
    assert report["false_positives_per_hour"] == 360
    assert report["mean_start_error_seconds"] == 0.25
    assert report["mean_end_error_seconds"] == 0.5


def test_source_edl_uses_source_in_out_not_montage_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "manual.edl"
    path.write_text("TITLE: CUT\nFCM: NON-DROP FRAME\n001 AX V C 00:00:10:00 00:00:14:15 00:00:00:00 00:00:04:15\n")
    assert import_source_intervals(path, fps=30) == [{"start": 10, "end": 14.5}]
    with pytest.raises(ValueError, match="frame rate"):
        import_source_intervals(path)
    path.write_text("FCM: DROP FRAME\n001 AX V C 00:00:10:00 00:00:14:15 00:00:00:00 00:00:04:15\n")
    with pytest.raises(ValueError, match="Drop-frame"):
        import_source_intervals(path, fps=29.97)


def test_training_rejects_mismatched_npz_source(tmp_path: Path) -> None:
    paths = [_recording(tmp_path, i)[0] for i in range(3)]
    payload = json.loads(paths[0].read_text())
    payload["source"]["id"] = "different-original"
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint differs"):
        train_from_feedback(paths, tmp_path / "profile.json")


def test_exported_source_decisions_roundtrip_to_manual_reference(tmp_path: Path) -> None:
    from models import Rally
    from training.evaluation import load_reference_intervals
    from video.edit_decisions import save_edit_decisions

    source = tmp_path / "original.mov"
    source.write_bytes(b"video placeholder")
    rallies = [Rally(10, 14.5), Rally(22.25, 26.75), Rally(30, 33, enabled=False)]
    expected = [{"start": 10, "end": 14.5}, {"start": 22.25, "end": 26.75}]
    for extension in (".edl", ".json"):
        path = tmp_path / f"edit{extension}"
        save_edit_decisions(path, source, rallies, fps=60, source_duration=40)
        assert load_reference_intervals(path, fps=60) == expected


def test_edl_roundtrip_fractional_frame_rate(tmp_path: Path) -> None:
    from models import Rally
    from video.edit_decisions import save_edit_decisions

    fps = 30000 / 1001
    path = tmp_path / "edit.edl"
    save_edit_decisions(path, tmp_path / "game;day.mov", [Rally(3601.25, 3608.5)], fps=fps)
    result = import_source_intervals(path, fps=fps)
    assert result[0]["start"] == pytest.approx(3601.25, abs=0.5 / fps)
    assert result[0]["end"] == pytest.approx(3608.5, abs=0.5 / fps)


def test_import_rejects_ambiguous_edit_json_and_multi_clip_edl(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps({"edits": [{"output_start": 0, "output_end": 10}]}))
    with pytest.raises(ValueError, match="Unsupported edit-decision"):
        import_source_intervals(path)
    path = tmp_path / "multi.edl"
    path.write_text("FCM: NON-DROP FRAME\n"
                    "001 AX V C 00:00:01:00 00:00:03:00 00:00:00:00 00:00:02:00\n"
                    "* FROM CLIP NAME: game1.mov\n"
                    "002 AX V C 00:00:05:00 00:00:07:00 00:00:02:00 00:00:04:00\n"
                    "* FROM CLIP NAME: game2.mov\n")
    with pytest.raises(ValueError, match="multiple source"):
        import_source_intervals(path, fps=30)


def test_training_never_overwrites_correction_input(tmp_path: Path) -> None:
    paths = [_recording(tmp_path, i)[0] for i in range(3)]
    original = paths[0].read_bytes()
    with pytest.raises(ValueError, match="separately from correction"):
        train_from_feedback(paths, paths[0])
    assert paths[0].read_bytes() == original


def test_pending_candidates_do_not_become_positive_training_samples(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path, _ = _recording(tmp_path, index, complete=False)
        payload = json.loads(path.read_text())
        for interval in payload["reviewed_intervals"]:
            interval["reviewed"] = False
        # Deliberately retain stale positive labels in NPZ. Loader must rebuild
        # from explicit review provenance, not trust the previous dense labels.
        path.write_text(json.dumps(payload))
        paths.append(path)
    with pytest.raises(ValueError, match="reviewed samples per class"):
        train_from_feedback(paths, tmp_path / "profile.json")


def test_unlabeled_video_does_not_count_as_validation_recording(tmp_path: Path) -> None:
    paths = [_recording(tmp_path, i, complete=False)[0] for i in range(3)]
    payload = json.loads(paths[0].read_text())
    payload["rejected_detections"] = []
    for interval in payload["reviewed_intervals"]:
        interval["reviewed"] = False
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="distinct original recordings; found 2"):
        train_from_feedback(paths, tmp_path / "profile.json")
