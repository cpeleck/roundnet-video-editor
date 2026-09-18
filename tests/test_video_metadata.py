from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video.errors import FFprobeNotFoundError, MetadataError, VideoFileError
from video.metadata import probe_video_metadata


def test_ffprobe_metadata_parses_fractional_fps_audio_and_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "phone.mov"
    video.write_bytes(b"not decoded in this unit test")
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 3840,
                "height": 2160,
                "avg_frame_rate": "30000/1001",
                "duration": "12.5",
                "nb_frames": "375",
                "side_data_list": [{"rotation": -90}],
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "12.6",
            "bit_rate": "14000000",
        },
    }
    monkeypatch.setattr(
        "video.metadata._resolve_executable", lambda _path, _name: "/fake/ffprobe"
    )
    monkeypatch.setattr(
        "video.metadata.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    metadata = probe_video_metadata(video, allow_opencv_fallback=False)

    assert metadata.path == video.resolve()
    assert metadata.duration_seconds == pytest.approx(12.5)
    assert metadata.fps == pytest.approx(29.97002997)
    assert metadata.resolution == (3840, 2160)
    assert metadata.display_resolution == (2160, 3840)
    assert metadata.rotation_degrees == 270
    assert metadata.has_audio is True
    assert metadata.video_codec == "hevc"
    assert metadata.audio_codec == "aac"
    assert metadata.bit_rate == 14_000_000
    assert metadata.probe_backend == "ffprobe"


def test_duration_falls_back_to_format_and_frame_count_is_estimated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "game.mp4"
    video.touch()
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "25/1",
            }
        ],
        "format": {"duration": "4.2"},
    }
    monkeypatch.setattr(
        "video.metadata._resolve_executable", lambda _path, _name: "/fake/ffprobe"
    )
    monkeypatch.setattr(
        "video.metadata.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    metadata = probe_video_metadata(video, allow_opencv_fallback=False)

    assert metadata.duration == pytest.approx(4.2)
    assert metadata.frame_count == 105
    assert metadata.has_audio is False


def test_missing_ffprobe_can_be_reported_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "game.mp4"
    video.touch()
    monkeypatch.setattr(
        "video.metadata._resolve_executable", lambda _path, _name: None
    )

    with pytest.raises(FFprobeNotFoundError, match="brew install ffmpeg"):
        probe_video_metadata(video, allow_opencv_fallback=False)


def test_ffprobe_failure_includes_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "broken.mp4"
    video.touch()
    monkeypatch.setattr(
        "video.metadata._resolve_executable", lambda _path, _name: "/fake/ffprobe"
    )
    monkeypatch.setattr(
        "video.metadata.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="moov atom not found"
        ),
    )

    with pytest.raises(MetadataError, match="moov atom not found"):
        probe_video_metadata(video, allow_opencv_fallback=False)


def test_missing_input_has_exact_path_in_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mov"

    with pytest.raises(VideoFileError, match=str(missing)):
        probe_video_metadata(missing)
