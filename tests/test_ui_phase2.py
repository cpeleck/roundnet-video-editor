from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from config import DetectionSettings
from ui.main_window import MainWindow, _new_rally, _serve_confidence
from ui.settings_dialog import DEFAULT_SETTINGS


def test_ui_phase2_defaults_match_detector_defaults() -> None:
    detector = DetectionSettings()

    for name in (
        "motion_sensitivity",
        "audio_sensitivity",
        "serve_sensitivity",
        "motion_weight",
        "roi_motion_weight",
        "motion_spread_weight",
        "audio_weight",
        "temporal_weight",
        "serve_weight",
        "rally_threshold",
        "end_threshold",
    ):
        assert DEFAULT_SETTINGS[name] == pytest.approx(getattr(detector, name))

    assert DEFAULT_SETTINGS["retrieval_filter_enabled"] is detector.retrieval_filter_enabled


def test_ui_rally_helper_preserves_serve_confidence() -> None:
    rally = _new_rally(2.0, 5.0, 0.8, True, 0.65)

    assert _serve_confidence(rally) == pytest.approx(0.65)


def test_main_window_label_writer_uses_training_feedback_format(tmp_path) -> None:
    source = tmp_path / "game.mp4"
    source.write_bytes(b"local-video-placeholder")
    destination = tmp_path / "game.labels.json"
    rally = _new_rally(0.5, 1.5, 0.8, True, 0.6)
    stub = SimpleNamespace(
        video_path=str(source),
        video_duration=2.0,
        roi=(0.1, 0.1, 0.8, 0.8),
        initial_detected_rallies=[
            {
                "start_time": 0.5,
                "end_time": 1.5,
                "confidence": 0.8,
                "enabled": True,
            }
        ],
        rallies=[rally],
        rejected_detections=[],
        detection_result={
            "timestamps": np.asarray([0.0, 0.5, 1.0, 1.5]),
            "motion_scores": np.asarray([0.0, 0.8, 0.9, 0.1]),
            "rally_scores": np.asarray([0.0, 0.7, 0.8, 0.1]),
        },
        detection_settings=DetectionSettings().to_dict(),
    )

    result = MainWindow._write_labels(stub, str(destination))

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["final_enabled_intervals"] == [{"end": 1.5, "start": 0.5}]
    assert result.features_path == destination.with_suffix(".npz")
    assert result.features_path.is_file()
