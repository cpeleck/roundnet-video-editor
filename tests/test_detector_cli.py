from __future__ import annotations

import json

import numpy as np

from detection import detector
from detection.detector import DetectionResult
from models import Rally


def _result() -> DetectionResult:
    samples = np.asarray([0.0, 0.5, 1.0])
    return DetectionResult(
        rallies=[Rally(0.25, 1.25, 0.9)],
        timestamps=samples,
        motion_scores=np.asarray([0.0, 0.8, 0.2]),
        roi_motion_scores=np.asarray([0.0, 0.9, 0.2]),
        audio_scores=np.asarray([0.0, 0.7, 0.0]),
        temporal_scores=np.asarray([0.1, 0.6, 0.3]),
        rally_scores=np.asarray([0.0, 0.75, 0.2]),
        duration=2.0,
    )


def test_cli_emits_machine_readable_json(monkeypatch, tmp_path, capsys) -> None:
    video = tmp_path / "game.mp4"
    video.write_bytes(b"placeholder")
    monkeypatch.setattr(
        detector.RoundnetDetector,
        "analyze",
        lambda self, video_path, roi=None, progress_callback=None: _result(),
    )

    exit_code = detector.main([str(video), "--quiet"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rally_count"] == 1
    assert payload["rallies"][0]["start_time"] == 0.25
    assert "signals" not in payload


def test_cli_can_include_debug_signals(monkeypatch, tmp_path) -> None:
    video = tmp_path / "game.mov"
    video.write_bytes(b"placeholder")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        detector.RoundnetDetector,
        "analyze",
        lambda self, video_path, roi=None, progress_callback=None: _result(),
    )

    exit_code = detector.main(
        [
            str(video),
            "--quiet",
            "--include-signals",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["signals"]["timestamps"] == [0.0, 0.5, 1.0]


def test_cli_reports_missing_video(capsys, tmp_path) -> None:
    exit_code = detector.main([str(tmp_path / "missing.mp4"), "--quiet"])

    assert exit_code == 1
    assert "does not exist" in capsys.readouterr().err

