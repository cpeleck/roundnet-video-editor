from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from video.edit_decisions import build_edit_decisions, save_edit_decisions
from video.exporter import FFmpegExporter, generate_export_segments
from video.metadata import probe_video_metadata
from video.presentation import crop_dimensions, crop_filter, match_statistics, prepare_export_options, scores_before_rallies


def test_score_counts_disabled_points_but_not_rejected_detections() -> None:
    rallies = [
        {"start": 20, "end": 22, "winner": "A", "starred": True},
        {"start": 0, "end": 2, "winner": "B", "enabled": False},
        {"start": 5, "end": 8, "winner": "A", "rejected": True},
        {"start": 10, "end": 12, "winner": "A", "starred": False},
    ]
    options = prepare_export_options({"initial_score_a": 3, "initial_score_b": 2})
    assert scores_before_rallies(rallies, options)[0] == (4, 3)
    stats = match_statistics(rallies, options)
    assert (stats["score_a"], stats["score_b"], stats["tagged_points"]) == (5, 3, 3)
    assert [segment.source_index for segment in generate_export_segments(rallies, highlights_only=True)] == [0]


def test_edit_decisions_preserve_exact_source_and_output_times(tmp_path: Path) -> None:
    rallies = [{"start": 9.1234, "end": 10.4321, "starred": True, "note": "hello\nTITLE: bad"},
               {"start": 3, "end": 4, "enabled": False}, {"start": 0, "end": 1.5, "starred": True}]
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    path = save_edit_decisions(tmp_path / "cuts.json", source, rallies, fps=29.97)
    data = json.loads(path.read_text())
    assert data["edits"][1]["source_start"] == 9.1234
    assert data["edits"][1]["output_start"] == 1.5
    assert data["output_duration"] == pytest.approx(2.8087)
    edl = save_edit_decisions(tmp_path / "cuts.edl", source, rallies, fps=29.97).read_text()
    assert "FCM: NON-DROP FRAME" in edl
    assert "* COMMENT: hello TITLE: bad" in edl
    assert "\nTITLE: bad" not in edl
    with pytest.raises(ValueError, match="separately"):
        save_edit_decisions(source, source, rallies)


def test_crop_has_exact_even_ratio_and_validates_keyframes() -> None:
    assert crop_dimensions(1920, 1080, "9:16") == (594, 1056)
    assert crop_dimensions(1920, 1080, "1:1") == (1080, 1080)
    graph = crop_filter(1920, 1080, "9:16", [{"time": 10, "x": 0, "y": .5}, {"time": 12, "x": 1, "y": .5}], 10)
    assert "clip(" in graph and "3-2*" in graph
    with pytest.raises(ValueError, match="normalized"):
        crop_filter(1920, 1080, "9:16", [{"time": 10, "x": 2, "y": .5}], 10)


@pytest.mark.parametrize("has_audio,ratio", [(True, "9:16"), (False, "1:1")])
def test_real_export_renders_score_crop_overlay_and_summary(tmp_path: Path, has_audio: bool, ratio: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg not available")
    source = tmp_path / "source.mov"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=20:duration=3"]
    if has_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3", "-c:a", "aac"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)]
    subprocess.run(command, check=True, capture_output=True)
    logo = tmp_path / "logo ' [safe].png"
    cv2.imwrite(str(logo), np.full((20, 40, 4), (0, 0, 255, 220), dtype=np.uint8))
    rallies = [
        {"start": 0, "end": .6, "enabled": False, "winner": "A"},
        {"start": .7, "end": 1.8, "starred": True, "winner": "B", "note": "Nice shot: [x]; ' ",
         "crop_keyframes": [{"time": .7, "x": 0, "y": .5}, {"time": 1.8, "x": 1, "y": .5}]},
        {"start": 2, "end": 2.8, "rejected": True, "winner": "A"},
    ]
    output = tmp_path / "highlights.mp4"
    result = FFmpegExporter().export(source, output, rallies, prefer_hardware=False,
        export_options={"aspect_ratio": ratio, "scoreboard": True, "highlights_only": True,
                        "team_a": "A':;[x]", "team_b": "B", "include_stats": True,
                        "include_notes": True, "overlay_path": str(logo)})
    metadata = probe_video_metadata(output)
    assert metadata.display_resolution == crop_dimensions(320, 180, ratio)
    assert metadata.duration == pytest.approx(4.1, abs=.15)
    assert metadata.has_audio == has_audio
    assert result.segment_count == 1
    assert not list(tmp_path.glob(".roundnet-export-*"))
    capture = cv2.VideoCapture(str(output))
    capture.set(cv2.CAP_PROP_POS_MSEC, 2200)
    success, frame = capture.read()
    capture.release()
    assert success
    # The appended summary has a dark slate background, not the test pattern.
    assert tuple(frame[0, 0].tolist()) == pytest.approx((35, 25, 17), abs=8)
