"""Audio transient extraction used as supporting evidence for active play."""

from __future__ import annotations

from dataclasses import dataclass
import math
from os import PathLike
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Sequence

import numpy as np

from config import DetectionSettings
from ._callbacks import (
    CancelCallback,
    DetectionCancelled,
    ProgressCallback,
    check_cancelled,
    report_progress,
)


@dataclass(frozen=True)
class AudioAnalysis:
    """Audio features aligned to the sampled video timestamps."""

    timestamps: np.ndarray
    transient_activity: np.ndarray
    rms_energy: np.ndarray
    warning: str | None = None

    @property
    def audio_scores(self) -> np.ndarray:
        """Compatibility alias for the transient signal."""

        return self.transient_activity


class AudioDetector:
    """Extract mono PCM with FFmpeg and measure short impact-like transients."""

    def __init__(self, settings: DetectionSettings | None = None) -> None:
        self.settings = settings or DetectionSettings()
        self.settings.validate()
        self.last_warning: str | None = None

    def analyze(
        self,
        video_path: str | PathLike[str],
        timestamps: Sequence[float] | np.ndarray,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> AudioAnalysis:
        """Return transient activity aligned to ``timestamps``.

        Missing audio, a missing FFmpeg installation, or a malformed audio
        stream degrades gracefully to a zero signal.  Motion-only analysis can
        therefore still complete, and the reason is exposed through ``warning``.
        """

        path = Path(video_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {path}")
        times = np.asarray(timestamps, dtype=np.float64)
        if times.ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if np.any(~np.isfinite(times)) or (times.size and times[0] < 0):
            raise ValueError("timestamps must be finite and non-negative")
        if times.size > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if not times.size:
            empty = np.empty(0, dtype=np.float64)
            return AudioAnalysis(times, empty, empty)

        check_cancelled(cancel_callback)
        report_progress(progress_callback, 0.0, "Extracting audio")
        samples, warning = self._extract_pcm(path, cancel_callback)
        self.last_warning = warning
        if warning is not None or samples.size == 0:
            zeros = np.zeros(times.size, dtype=np.float64)
            report_progress(progress_callback, 1.0, warning or "No audio stream found")
            return AudioAnalysis(times, zeros, zeros.copy(), warning or "No audio stream found")

        report_progress(progress_callback, 0.65, "Analyzing audio transients")
        transient, energy = self._measure_features(samples, times, cancel_callback)
        report_progress(progress_callback, 1.0, "Audio analysis complete")
        return AudioAnalysis(times, transient, energy)

    def _extract_pcm(
        self, path: Path, cancel_callback: CancelCallback | None
    ) -> tuple[np.ndarray, str | None]:
        temporary_path: Path | None = None
        process: subprocess.Popen[bytes] | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            command = [
                self.settings.ffmpeg_path,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-map",
                "0:a:0?",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(self.settings.audio_sample_rate),
                "-f",
                "s16le",
                str(temporary_path),
            ]
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except FileNotFoundError:
                return np.empty(0, dtype=np.float32), (
                    f"FFmpeg executable not found: {self.settings.ffmpeg_path}"
                )
            except OSError as exc:
                return np.empty(0, dtype=np.float32), (
                    f"Could not start FFmpeg ({self.settings.ffmpeg_path}): {exc}"
                )

            started = time.monotonic()
            stderr = b""
            while True:
                check_cancelled(cancel_callback)
                if time.monotonic() - started > self.settings.ffmpeg_timeout:
                    process.kill()
                    stderr = process.communicate()[1] or b""
                    return np.empty(0, dtype=np.float32), (
                        f"Audio extraction timed out after {self.settings.ffmpeg_timeout:g}s"
                    )
                try:
                    _, stderr = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue

            if process.returncode != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                if len(detail) > 300:
                    detail = detail[-300:]
                message = "FFmpeg could not extract an audio stream"
                if detail:
                    message = f"{message}: {detail}"
                return np.empty(0, dtype=np.float32), message

            sample_count = temporary_path.stat().st_size // np.dtype(np.int16).itemsize
            if sample_count <= 0:
                return np.empty(0, dtype=np.float32), "Video has no usable audio stream"
            pcm = np.fromfile(temporary_path, dtype="<i2", count=sample_count)
            return pcm.astype(np.float32) / 32768.0, None
        except DetectionCancelled:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
            raise
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _measure_features(
        self,
        samples: np.ndarray,
        timestamps: np.ndarray,
        cancel_callback: CancelCallback | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_rate = self.settings.audio_sample_rate
        if timestamps.size > 1:
            timestamp_period = float(np.median(np.diff(timestamps)))
        else:
            timestamp_period = 1.0 / self.settings.analysis_fps
        window_seconds = max(self.settings.audio_window_duration, timestamp_period)
        half_window = window_seconds / 2.0

        energy = np.zeros(timestamps.size, dtype=np.float64)
        peak = np.zeros(timestamps.size, dtype=np.float64)
        for index, timestamp in enumerate(timestamps):
            if index % 128 == 0:
                check_cancelled(cancel_callback)
            start = max(0, math.floor((float(timestamp) - half_window) * sample_rate))
            end = min(samples.size, math.ceil((float(timestamp) + half_window) * sample_rate))
            if end <= start:
                continue
            window = samples[start:end]
            energy[index] = math.sqrt(float(np.mean(np.square(window, dtype=np.float64))))
            # A high percentile is more robust than a single maximum in windy
            # phone recordings but still emphasizes short ball/hand impacts.
            peak[index] = float(np.percentile(np.abs(window), 99.5))

        log_energy = np.log1p(50.0 * energy)
        onset = np.maximum(0.0, np.diff(log_energy, prepend=log_energy[0]))
        transient = 0.55 * peak + 0.25 * energy + 0.20 * onset
        return transient, energy


def analyze_audio(
    video_path: str | PathLike[str],
    timestamps: Sequence[float] | np.ndarray,
    settings: DetectionSettings | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> AudioAnalysis:
    """Functional convenience wrapper around :class:`AudioDetector`."""

    return AudioDetector(settings).analyze(
        video_path,
        timestamps,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


__all__ = ["AudioAnalysis", "AudioDetector", "analyze_audio"]
