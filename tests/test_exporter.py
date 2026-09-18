from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from video.errors import ExportCancelledError, ExportError, FFmpegNotFoundError
from video.exporter import (
    ExportSegment,
    FFmpegExporter,
    build_ffmpeg_command,
    build_filter_complex,
    generate_export_segments,
)
from video.metadata import VideoMetadata


@dataclass
class RallyStub:
    start_time: float
    end_time: float
    confidence: float = 0.8
    enabled: bool = True


def metadata_for(path: Path, *, has_audio: bool = True) -> VideoMetadata:
    return VideoMetadata(
        path=path,
        duration_seconds=60.0,
        width=1920,
        height=1080,
        fps=29.97,
        frame_count=1798,
        has_audio=has_audio,
        video_codec="h264",
        audio_codec="aac" if has_audio else None,
        format_name="mov,mp4",
    )


def test_generate_segments_ignores_disabled_and_sorts_chronologically() -> None:
    segments = generate_export_segments(
        [
            RallyStub(10.25, 13.5),
            RallyStub(2.0, 4.0),
            RallyStub(7.0, 8.0, enabled=False),
        ],
        source_duration=20.0,
    )

    assert [(item.start_time, item.end_time) for item in segments] == [
        (2.0, 4.0),
        (10.25, 13.5),
    ]
    assert [item.source_index for item in segments] == [1, 0]


def test_generate_segments_supports_dicts_aliases_and_pairs() -> None:
    segments = generate_export_segments(
        [
            {"start": 5, "end": 6, "enabled": True},
            (1.25, 2.75),
            {"start_time": 8, "end_time": 9, "enabled": False},
        ]
    )

    assert [(item.start, item.end) for item in segments] == [(1.25, 2.75), (5, 6)]


@pytest.mark.parametrize(
    ("rally", "message"),
    [
        (RallyStub(-1, 2), "starts before the video"),
        (RallyStub(3, 3), "must end after it starts"),
        (RallyStub(4, 2), "must end after it starts"),
        (RallyStub(float("nan"), 2), "non-finite start time"),
    ],
)
def test_generate_segments_rejects_invalid_ranges(
    rally: RallyStub, message: str
) -> None:
    with pytest.raises(ExportError, match=message):
        generate_export_segments([rally], source_duration=10)


def test_generate_segments_rejects_end_beyond_source() -> None:
    with pytest.raises(ExportError, match="beyond the source duration"):
        generate_export_segments([RallyStub(9, 11)], source_duration=10)


def test_generate_segments_clamps_small_container_duration_disagreement() -> None:
    segments = generate_export_segments(
        [RallyStub(9, 10.1)], source_duration=10
    )

    assert segments[0].end_time == 10


def test_filter_graph_trims_video_and_audio_then_concatenates() -> None:
    graph = build_filter_complex(
        [ExportSegment(1.25, 3), ExportSegment(9, 10.5)], has_audio=True
    )

    assert "[0:v:0]trim=start=1.25:end=3,setpts=PTS-STARTPTS[v0]" in graph
    assert "[0:a:0]atrim=start=9:end=10.5,asetpts=PTS-STARTPTS[a1]" in graph
    assert graph.endswith("[v0][a0][v1][a1]concat=n=2:v=1:a=1[vout][aout]")


def test_filter_graph_supports_silent_video() -> None:
    graph = build_filter_complex([ExportSegment(0, 2.5)], has_audio=False)

    assert "[0:a" not in graph
    assert graph.endswith("[v0]concat=n=1:v=1:a=0[vout]")


def test_software_command_preserves_source_dimensions_and_fps() -> None:
    command = build_ffmpeg_command(
        "source movie.mov",
        "edited.mp4",
        [ExportSegment(2, 4)],
        has_audio=True,
    )

    assert command[0] == "ffmpeg"
    assert command[command.index("-i") + 1] == "source movie.mov"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-crf") + 1] == "18"
    assert command[command.index("-c:a") + 1] == "aac"
    assert "-r" not in command
    assert "-s" not in command
    assert "-vf" not in command
    assert command[-1] == "edited.mp4"


def test_videotoolbox_command_requests_quality_mode() -> None:
    command = build_ffmpeg_command(
        "in.mov",
        "out.mp4",
        [ExportSegment(0, 2)],
        has_audio=False,
        video_encoder="h264_videotoolbox",
    )

    assert command[command.index("-c:v") + 1] == "h264_videotoolbox"
    assert command[command.index("-q:v") + 1] == "65"
    assert "-c:a" not in command


def test_missing_ffmpeg_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("video.exporter.shutil.which", lambda _name: None)

    with pytest.raises(FFmpegNotFoundError, match="brew install ffmpeg"):
        _ = FFmpegExporter().ffmpeg_path


def test_export_writes_temp_then_atomically_installs_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "game.mov"
    source.write_bytes(b"source")
    destination = tmp_path / "game_roundnet_rallies.mp4"
    exporter = FFmpegExporter(ffmpeg_path=sys.executable)
    monkeypatch.setattr(exporter, "hardware_encoder_available", lambda: False)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        callback = kwargs["progress_callback"]
        assert callable(callback)
        callback(0.5)
        Path(command[-1]).write_bytes(b"finished mp4")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(exporter, "_run_ffmpeg", fake_run)
    progress: list[float] = []
    result = exporter.export(
        source,
        destination,
        [RallyStub(20, 22), RallyStub(4, 7), RallyStub(10, 12, enabled=False)],
        metadata=metadata_for(source),
        progress_callback=progress.append,
    )

    assert destination.read_bytes() == b"finished mp4"
    assert result.output_path == destination.resolve()
    assert result.segment_count == 2
    assert result.output_duration_seconds == pytest.approx(5)
    assert result.video_encoder == "libx264"
    assert progress == [0.0, 0.5, 1.0]
    filter_graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert filter_graph.index("trim=start=4:end=7") < filter_graph.index(
        "trim=start=20:end=22"
    )
    assert not list(tmp_path.glob("*.partial-*.mp4"))


def test_hardware_failure_retries_with_software(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "game.mov"
    source.write_bytes(b"source")
    destination = tmp_path / "highlights.mp4"
    exporter = FFmpegExporter(ffmpeg_path=sys.executable)
    monkeypatch.setattr(exporter, "hardware_encoder_available", lambda: True)
    attempted_encoders: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        encoder = command[command.index("-c:v") + 1]
        attempted_encoders.append(encoder)
        if encoder == "h264_videotoolbox":
            Path(command[-1]).write_bytes(b"incomplete")
            return SimpleNamespace(returncode=1, stderr="hardware encoder failed")
        Path(command[-1]).write_bytes(b"software output")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(exporter, "_run_ffmpeg", fake_run)
    result = exporter.export(
        source,
        destination,
        [RallyStub(1, 3)],
        metadata=metadata_for(source),
    )

    assert attempted_encoders == ["h264_videotoolbox", "libx264"]
    assert destination.read_bytes() == b"software output"
    assert result.video_encoder == "libx264"
    assert result.used_hardware_acceleration is False


def test_failed_export_preserves_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "game.mov"
    source.write_bytes(b"source")
    destination = tmp_path / "highlights.mp4"
    destination.write_bytes(b"previous good export")
    exporter = FFmpegExporter(ffmpeg_path=sys.executable)
    monkeypatch.setattr(exporter, "hardware_encoder_available", lambda: False)

    def fake_failure(command: list[str], **_kwargs: object) -> SimpleNamespace:
        Path(command[-1]).write_bytes(b"partial corrupt output")
        return SimpleNamespace(returncode=1, stderr="simulated encoder error")

    monkeypatch.setattr(exporter, "_run_ffmpeg", fake_failure)

    with pytest.raises(ExportError, match="simulated encoder error"):
        exporter.export(
            source,
            destination,
            [RallyStub(1, 2)],
            metadata=metadata_for(source),
            overwrite=True,
        )

    assert destination.read_bytes() == b"previous good export"
    assert not list(tmp_path.glob(".*.partial-*.mp4"))


def test_cancel_before_export_leaves_no_output(tmp_path: Path) -> None:
    source = tmp_path / "game.mov"
    source.write_bytes(b"source")
    destination = tmp_path / "highlights.mp4"

    with pytest.raises(ExportCancelledError, match="before it started"):
        FFmpegExporter(ffmpeg_path=sys.executable).export(
            source,
            destination,
            [RallyStub(1, 2)],
            metadata=metadata_for(source),
            cancel_callback=lambda: True,
        )

    assert not destination.exists()


def test_export_rejects_empty_enabled_selection(tmp_path: Path) -> None:
    source = tmp_path / "game.mov"
    source.write_bytes(b"source")

    with pytest.raises(ExportError, match="No enabled rallies"):
        FFmpegExporter(ffmpeg_path=sys.executable).export(
            source,
            tmp_path / "out.mp4",
            [RallyStub(1, 2, enabled=False)],
            metadata=metadata_for(source),
        )
