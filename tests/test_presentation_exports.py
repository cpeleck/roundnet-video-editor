from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from video.edit_decisions import build_edit_decisions, save_edit_decisions
from video.exporter import FFmpegExporter, _prepare_presentation, generate_export_segments
from video.metadata import probe_video_metadata
from video.presentation import crop_dimensions, crop_filter, export_dimensions, match_statistics, prepare_export_options, render_overlay_card, scorecard_lines, scores_before_rallies


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


def test_classified_outcomes_supply_export_scores_without_cached_winners() -> None:
    from models import Rally

    rallies = [
        Rally(0, 2, classification={"version": 1, "kind": "ace"}),
        Rally(3, 5, classification={"version": 1, "kind": "redo"}),
        Rally(6, 8, classification={"version": 1, "kind": "double_fault"}),
    ]
    options = prepare_export_options()
    assert scores_before_rallies(rallies, options) == {0: (0, 0), 1: (1, 0), 2: (1, 0)}
    summary = match_statistics(rallies, options)
    assert (summary["score_a"], summary["score_b"], summary["tagged_points"]) == (2, 0, 2)
    assert summary["outcomes"] == {"Ace": 1, "Double Fault": 1}


def test_end_card_marks_incomplete_scoring_as_recorded(monkeypatch) -> None:
    from models import Rally
    import video.presentation as presentation

    summary = match_statistics([Rally(0, 1), Rally(2, 3, classification={"version": 1, "kind": "ace"})])
    drawn = []
    original = presentation.cv2.putText

    def capture(frame, value, *args, **kwargs):
        drawn.append(value)
        return original(frame, value, *args, **kwargs)

    monkeypatch.setattr(presentation.cv2, "putText", capture)
    presentation.render_player_end_card_image(1600, 900, summary)
    assert "RECORDED SCORE" in drawn
    assert "FINAL SCORE" not in drawn
    assert "KNOWN PLAYER STATISTICS" in drawn


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


def test_overlay_box_fits_text_instead_of_using_fixed_video_width(tmp_path):
    short, long = tmp_path / "short.png", tmp_path / "long.png"
    render_overlay_card(short, 864, 22, ["A 0 | B 0"])
    render_overlay_card(long, 864, 22, ["North 0 | South 0"])
    small, large = cv2.imread(str(short), cv2.IMREAD_UNCHANGED), cv2.imread(str(long), cv2.IMREAD_UNCHANGED)
    assert small.shape[1] < large.shape[1] < 450
    assert small.shape[0] == large.shape[0] < 55
    assert small.shape[2] == 4
    oversized = tmp_path / "oversized.png"
    render_overlay_card(oversized, 200, 22, ["A very long team name " * 20])
    assert cv2.imread(str(oversized)).shape[1] <= 200


def test_scorecard_includes_omitted_points_and_resets_for_reordered_highlights(tmp_path, monkeypatch):
    from models.match_flow import match_timeline
    from video.metadata import VideoMetadata
    rallies = [{"rally_id": str(i), "start": i, "end": i + .5,
                "classification": {"version": 1, "kind": "ace"}, "enabled": i != 1}
               for i in range(4)]
    options = prepare_export_options({"scoreboard": True})
    segments = generate_export_segments(rallies, segment_order=["3", "0"])
    metadata = VideoMetadata(path=tmp_path / "source.mp4", duration_seconds=4,
                             width=1920, height=1080, fps=30, frame_count=120, has_audio=False)
    rendered = []
    monkeypatch.setattr("video.exporter.render_overlay_card", lambda path, width, font_size, lines: rendered.append(lines))
    _prepare_presentation(tmp_path, rallies, segments, metadata, options)
    assert rendered == [
        ["Team A 1 | Team B 2"],
        ["Team A 0 | Team B 0"],
    ]
    last = match_timeline(rallies, options)[-1]
    assert scorecard_lines(last, options, completed=True) == [
        "Team A 2 | Team B 2"]


def test_real_export_updates_team_score_at_each_cut(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required")
    source, output = tmp_path / "source.mp4", tmp_path / "scorecards.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "color=c=white:s=640x360:r=30:d=1.5", "-c:v", "libx264", str(source)],
        check=True, capture_output=True)
    rallies = [{"start": i * .5, "end": (i + 1) * .5,
                "classification": {"version": 1, "kind": "ace"}} for i in range(3)]
    FFmpegExporter().export(source, output, rallies, prefer_hardware=False,
        export_options={"scoreboard": True})
    expected = [
        ["Team A 0 | Team B 0"],
        ["Team A 1 | Team B 0"],
        ["Team A 1 | Team B 1"],
    ]
    capture = cv2.VideoCapture(str(output))
    try:
        for index, lines in enumerate(expected):
            expected_path = tmp_path / f"expected-{index}.png"
            render_overlay_card(expected_path, 288, 10, lines)
            image = cv2.imread(str(expected_path), cv2.IMREAD_UNCHANGED).astype(float)
            alpha = image[:, :, 3:4] / 255
            composite = image[:, :, :3] * alpha + 255 * (1 - alpha)
            # Check the very first frame after each cut: no one-frame lag.
            capture.set(cv2.CAP_PROP_POS_FRAMES, index * 15)
            ok, frame = capture.read()
            assert ok
            actual = frame[16:16 + image.shape[0], 16:16 + image.shape[1]].astype(float)
            assert np.abs(actual - composite).mean() < 5
    finally:
        capture.release()


@pytest.mark.parametrize("aspect,expected", [
    ("source", (3840, 2160)), ("16:9", (3840, 2160)),
    ("9:16", (2160, 3840)), ("1:1", (2160, 2160)),
])
def test_4k_dimensions_are_independent_of_source_size(aspect, expected):
    assert export_dimensions(1920, 1080, aspect, "2160p") == expected
    assert export_dimensions(7680, 4320, aspect, "2160p") == expected
    assert export_dimensions(7680, 4320, "source") == (7680, 4320)


def test_resolution_preserves_original_framing_and_even_pixels():
    assert export_dimensions(640, 480, "source", "2160p") == (2880, 2160)
    assert export_dimensions(480, 640, "source", "2160p") == (2160, 2880)
    assert export_dimensions(1920, 1080, "16:9", "720p") == (1280, 720)
    assert export_dimensions(1920, 1080, "16:9", "1080p") == (1920, 1080)
    assert export_dimensions(1920, 1080, "16:9", "1440p") == (2560, 1440)


@pytest.mark.parametrize("options", [
    {"resolution": "8k"}, {"frame_rate": 120}, {"frame_rate": 0},
    {"frame_rate": True}, {"frame_rate": "60"}, {"frame_rate": float("nan")},
])
def test_export_quality_settings_reject_invalid_values(options):
    with pytest.raises(ValueError):
        prepare_export_options(options)


@pytest.mark.parametrize("with_presentation", [False, True])
def test_real_4k_60fps_export(tmp_path, with_presentation):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe required")
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "color=c=red:s=320x180:r=30:d=1", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=1", "-c:v", "libx264",
        "-c:a", "aac", str(source)], check=True, capture_output=True)
    output = tmp_path / "4k60.mp4"
    logo = tmp_path / "logo.png"
    cv2.imwrite(str(logo), np.full((20, 40, 4), (0, 0, 255, 220), dtype=np.uint8))
    FFmpegExporter().export(source, output,
        [{"start": .033, "end": .2}, {"start": .433, "end": .6}],
        prefer_hardware=False, export_options={"resolution": "2160p", "frame_rate": 60,
            "scoreboard": with_presentation, "include_stats": with_presentation,
            "overlay_path": str(logo) if with_presentation else None, "stats_duration": 1})
    metadata = probe_video_metadata(output)
    assert metadata.display_resolution == (3840, 2160)
    assert metadata.fps == pytest.approx(60)
    assert metadata.has_audio
    assert metadata.duration == pytest.approx(.334 + int(with_presentation), abs=.06)
    frames = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(output)],
        check=True, capture_output=True, text=True).stdout)["frames"]
    timestamps = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    assert np.diff(timestamps) == pytest.approx(np.full(len(timestamps) - 1, 1 / 60), abs=.000002)


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
                        "stats_duration": 4.0,
                        "include_notes": True, "overlay_path": str(logo)})
    metadata = probe_video_metadata(output)
    assert metadata.display_resolution == crop_dimensions(320, 180, ratio)
    assert metadata.duration == pytest.approx(5.1, abs=.15)
    assert metadata.has_audio == has_audio
    assert result.segment_count == 1
    assert not list(tmp_path.glob(".roundnet-export-*"))
    capture = cv2.VideoCapture(str(output))
    capture.set(cv2.CAP_PROP_POS_MSEC, 2200)
    success, frame = capture.read()
    capture.release()
    assert success
    # The appended player card has its own dark background, not the source frame.
    assert tuple(frame[0, 0].tolist()) == pytest.approx((61, 42, 32), abs=8)
