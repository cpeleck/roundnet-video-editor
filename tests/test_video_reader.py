from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from video.errors import FrameExtractionCancelledError
from video.metadata import probe_video_metadata
from video.video_reader import VideoReader, extract_frame, extract_thumbnail


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (80, 60),
    )
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot create an MJPG test video")
    for frame_index in range(20):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[:, :, 0] = frame_index * 8  # blue in BGR
        frame[:, :, 1] = 40
        frame[:, :, 2] = 240 - frame_index * 4  # red in BGR
        writer.write(frame)
    writer.release()
    return path


def test_opencv_metadata_fallback_reads_generated_video(
    sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "video.metadata._resolve_executable", lambda _path, _name: None
    )

    metadata = probe_video_metadata(sample_video)

    assert metadata.probe_backend == "opencv"
    assert metadata.resolution == (80, 60)
    assert metadata.fps == pytest.approx(10.0)
    assert metadata.frame_count == 20
    assert metadata.duration_seconds == pytest.approx(2.0)
    assert metadata.has_audio is None


def test_extract_frame_rgb_crop_and_resize(sample_video: Path) -> None:
    with VideoReader(sample_video) as reader:
        rgb = reader.read_frame(
            0.0,
            rgb=True,
            roi=(10, 5, 40, 30),
            max_size=(20, 20),
        )

    assert rgb.shape == (15, 20, 3)
    # The source's red channel is intentionally much brighter than blue.
    assert float(rgb[:, :, 0].mean()) > float(rgb[:, :, 2].mean())


def test_normalized_roi_and_bounds_validation(sample_video: Path) -> None:
    frame = extract_frame(
        sample_video,
        0.25,
        roi=(0.25, 0.25, 0.5, 0.5),
        normalized_roi=True,
    )
    assert frame.shape == (30, 40, 3)

    with pytest.raises(ValueError, match="outside the decoded frame bounds"):
        extract_frame(sample_video, 0, roi=(70, 10, 20, 20))


def test_thumbnail_defaults_to_display_rgb_and_fits_bounds(sample_video: Path) -> None:
    thumbnail = extract_thumbnail(sample_video, max_size=(32, 32))

    assert thumbnail.shape == (24, 32, 3)
    assert float(thumbnail[:, :, 0].mean()) > float(thumbnail[:, :, 2].mean())


def test_sampled_iteration_reports_source_timestamps_and_progress(
    sample_video: Path,
) -> None:
    progress: list[float] = []
    with VideoReader(sample_video) as reader:
        samples = list(reader.iter_frames(2.0, progress_callback=progress.append))

    assert [sample.timestamp_seconds for sample in samples] == pytest.approx(
        [0.0, 0.5, 1.0, 1.5]
    )
    assert [sample.source_frame_index for sample in samples] == [0, 5, 10, 15]
    assert progress[0] == 0.0
    assert progress[-1] == 1.0
    assert progress == sorted(progress)


def test_sampled_iteration_can_be_cancelled(sample_video: Path) -> None:
    checks = 0

    def cancel_after_first_frame() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    with VideoReader(sample_video) as reader:
        iterator = reader.iter_frames(2, cancel_callback=cancel_after_first_frame)
        first = next(iterator)
        assert first.timestamp_seconds == pytest.approx(0)
        with pytest.raises(FrameExtractionCancelledError, match="cancelled"):
            next(iterator)
