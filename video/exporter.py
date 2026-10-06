"""Reliable FFmpeg export of enabled rally time ranges."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import platform
from queue import Empty, Queue
import shutil
import subprocess
from threading import Thread
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeAlias
from uuid import uuid4

from .errors import (
    ExportCancelledError,
    ExportError,
    FFmpegNotFoundError,
    MetadataError,
    VideoFileError,
)
from .metadata import VideoMetadata, probe_video_metadata
from .presentation import (
    crop_dimensions, crop_filter, field, match_statistics,
    prepare_export_options, render_card, render_player_end_card, scores_before_rallies,
)


LOGGER = logging.getLogger(__name__)
_BOUNDARY_TOLERANCE_SECONDS = 0.25

ProgressCallback: TypeAlias = Callable[[float], None]
CancelCallback: TypeAlias = Callable[[], bool]


@dataclass(frozen=True, slots=True, order=True)
class ExportSegment:
    """A validated source interval selected for the finished video."""

    start_time: float
    end_time: float
    source_index: int = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def start(self) -> float:
        return self.start_time

    @property
    def end(self) -> float:
        return self.end_time


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Summary returned after the final MP4 has been atomically installed."""

    output_path: Path
    segment_count: int
    output_duration_seconds: float
    video_encoder: str
    used_hardware_acceleration: bool

    @property
    def duration_seconds(self) -> float:
        return self.output_duration_seconds


@dataclass(frozen=True, slots=True)
class _ProcessOutcome:
    returncode: int
    stderr: str


def _field(item: object, names: Sequence[str], default: object = ...) -> object:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
    else:
        for name in names:
            if hasattr(item, name):
                return getattr(item, name)
    if default is ...:
        joined = " or ".join(repr(name) for name in names)
        raise ExportError(f"Rally is missing required field {joined}: {item!r}")
    return default


def _finite_time(value: object, field_name: str, rally_index: int) -> float:
    if isinstance(value, bool):
        raise ExportError(
            f"Rally {rally_index + 1} has a non-numeric {field_name}: {value!r}"
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ExportError(
            f"Rally {rally_index + 1} has a non-numeric {field_name}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ExportError(
            f"Rally {rally_index + 1} has a non-finite {field_name}: {value!r}"
        )
    return parsed


def generate_export_segments(
    rallies: Iterable[object],
    *,
    source_duration: float | None = None,
    highlights_only: bool = False,
) -> tuple[ExportSegment, ...]:
    """Convert enabled Rally-like objects into chronological source intervals.

    Objects may expose ``start_time``/``end_time`` attributes, ``start``/``end``
    attributes, equivalent mapping keys, or be simple ``(start, end)`` pairs.
    Disabled rallies are ignored before timestamp validation.
    """

    duration: float | None = None
    if source_duration is not None:
        try:
            duration = float(source_duration)
        except (TypeError, ValueError) as exc:
            raise ValueError("source_duration must be a number") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("source_duration must be a positive finite number")

    segments: list[ExportSegment] = []
    for index, rally in enumerate(rallies):
        enabled = _field(rally, ("enabled",), True)
        if not bool(enabled) or bool(_field(rally, ("rejected",), False)):
            continue
        if highlights_only and not bool(_field(rally, ("starred",), False)):
            continue
        if isinstance(rally, Sequence) and not isinstance(rally, (str, bytes)):
            if len(rally) != 2:
                raise ExportError(
                    f"Rally {index + 1} sequence must contain exactly (start, end)"
                )
            start_value, end_value = rally
        else:
            start_value = _field(rally, ("start_time", "start"))
            end_value = _field(rally, ("end_time", "end"))
        start = _finite_time(start_value, "start time", index)
        end = _finite_time(end_value, "end time", index)
        if start < 0:
            raise ExportError(
                f"Rally {index + 1} starts before the video at {start:.3f}s"
            )
        if end <= start:
            raise ExportError(
                f"Rally {index + 1} must end after it starts "
                f"({start:.3f}s -> {end:.3f}s)"
            )
        if duration is not None:
            if start >= duration:
                raise ExportError(
                    f"Rally {index + 1} starts at {start:.3f}s, at or beyond the "
                    f"source duration of {duration:.3f}s"
                )
            if end > duration + _BOUNDARY_TOLERANCE_SECONDS:
                raise ExportError(
                    f"Rally {index + 1} ends at {end:.3f}s, beyond the source "
                    f"duration of {duration:.3f}s"
                )
            # Container, OpenCV, and Qt duration estimates can differ by a frame.
            end = min(end, duration)
        segments.append(ExportSegment(start, end, index))

    return tuple(sorted(segments, key=lambda segment: (segment.start_time, segment.end_time)))


def _format_seconds(value: float) -> str:
    formatted = f"{value:.9f}".rstrip("0").rstrip(".")
    return formatted or "0"


def build_filter_complex(
    segments: Sequence[ExportSegment],
    *,
    has_audio: bool,
    video_filters: Mapping[int, str] | None = None,
    segment_overlays: Mapping[int, int] | None = None,
    custom_overlay_input: int | None = None,
    stats_input: int | None = None,
    stats_duration: float = 5.0,
    output_size: tuple[int, int] | None = None,
    fps: float = 30,
) -> str:
    """Build the trim/concat graph used for an exact, single-pass export."""

    if not segments:
        raise ValueError("At least one export segment is required")
    filters: list[str] = []
    concat_inputs: list[str] = []
    for index, segment in enumerate(segments):
        if segment.end_time <= segment.start_time or segment.start_time < 0:
            raise ValueError(f"Invalid export segment at index {index}: {segment!r}")
        start = _format_seconds(segment.start_time)
        end = _format_seconds(segment.end_time)
        extra = (video_filters or {}).get(index, "")
        base_label = f"base{index}" if index in (segment_overlays or {}) else f"v{index}"
        filters.append(f"[0:v:0]trim=start={start}:end={end},setpts=PTS-STARTPTS"
                       + (f",{extra}" if extra else "") + f"[{base_label}]")
        if index in (segment_overlays or {}):
            overlay_input = segment_overlays[index]
            padding = max(4, int((output_size or (1920, 1080))[0] * .025))
            filters.append(f"[{base_label}][{overlay_input}:v:0]overlay=x={padding}:y={padding}:eof_action=repeat:shortest=0[v{index}]")
        concat_inputs.append(f"[v{index}]")
        if has_audio:
            filters.append(
                f"[0:a:0]atrim=start={start}:end={end},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
            concat_inputs.append(f"[a{index}]")
    video_label = "vcut" if custom_overlay_input is not None or stats_input is not None else "vout"
    audio_label = "acut" if stats_input is not None else "aout"
    if has_audio:
        concat = (
            "".join(concat_inputs)
            + f"concat=n={len(segments)}:v=1:a=1[{video_label}][{audio_label}]"
        )
    else:
        concat = "".join(concat_inputs) + f"concat=n={len(segments)}:v=1:a=0[{video_label}]"
    filters.append(concat)
    if custom_overlay_input is not None:
        width, height = output_size or (1920, 1080)
        logo_width = max(2, int(width * .25))
        logo_height = max(2, int(height * .25))
        filters.append(f"[{custom_overlay_input}:v:0]scale={logo_width}:{logo_height}:force_original_aspect_ratio=decrease[logo]")
        next_label = "vbranded" if stats_input is not None else "vout"
        padding = max(4, int(width * .025))
        filters.append(f"[{video_label}][logo]overlay=x=W-w-{padding}:y=H-h-{padding}:eof_action=repeat:shortest=0[{next_label}]")
        video_label = next_label
    if stats_input is not None:
        filters.append(f"[{stats_input}:v:0]loop=loop=-1:size=1:start=0,setpts=N/(25*TB),trim=duration={stats_duration:.3f},fps={fps:.9f},setsar=1[statsv]")
        if has_audio:
            filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={stats_duration:.3f},asetpts=PTS-STARTPTS[statsa]")
            filters.append(f"[{video_label}][{audio_label}][statsv][statsa]concat=n=2:v=1:a=1[vout][aout]")
        else:
            filters.append(f"[{video_label}][statsv]concat=n=2:v=1:a=0[vout]")
    return ";".join(filters)


def build_ffmpeg_command(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    segments: Sequence[ExportSegment],
    *,
    has_audio: bool,
    video_encoder: str = "libx264",
    ffmpeg_path: str | os.PathLike[str] = "ffmpeg",
    overwrite: bool = True,
    include_progress: bool = True,
    image_inputs: Sequence[str | os.PathLike[str]] = (),
    filter_options: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return an argument-vector command without executing FFmpeg."""

    if video_encoder not in {"libx264", "h264_videotoolbox"}:
        raise ValueError(
            "video_encoder must be 'libx264' or 'h264_videotoolbox'"
        )
    filter_graph = build_filter_complex(segments, has_audio=has_audio, **(filter_options or {}))
    command = [
        os.fspath(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-i",
        os.fspath(input_path),
    ]
    for image_path in image_inputs:
        command.extend(["-i", os.fspath(image_path)])
    command.extend(["-filter_complex", filter_graph, "-map", "[vout]"])
    if has_audio:
        command.extend(["-map", "[aout]"])
    command.extend(["-map_metadata", "0", "-c:v", video_encoder])
    if video_encoder == "h264_videotoolbox":
        # A constant-quality request avoids guessing a bitrate from resolution.
        command.extend(["-b:v", "0", "-q:v", "65", "-allow_sw", "1"])
    else:
        command.extend(["-preset", "medium", "-crf", "18"])
    command.extend(["-pix_fmt", "yuv420p", "-fps_mode:v", "passthrough"])
    if has_audio:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.extend(["-sn", "-dn", "-movflags", "+faststart"])
    if include_progress:
        command.extend(["-progress", "pipe:1", "-nostats"])
    command.append(os.fspath(output_path))
    return command


def _short_error(message: str, limit: int = 1800) -> str:
    collapsed = " ".join(message.split())
    if not collapsed:
        return "FFmpeg returned no error details"
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


def _parse_clock(value: str) -> float | None:
    try:
        hours_text, minutes_text, seconds_text = value.split(":", 2)
        parsed = (
            int(hours_text) * 3600 + int(minutes_text) * 60 + float(seconds_text)
        )
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _progress_seconds(key: str, value: str) -> float | None:
    if key == "out_time_us":
        try:
            return max(0.0, float(value) / 1_000_000.0)
        except ValueError:
            return None
    if key == "out_time_ms":
        # Despite its historical name, FFmpeg reports this field in microseconds.
        try:
            return max(0.0, float(value) / 1_000_000.0)
        except ValueError:
            return None
    if key == "out_time":
        return _parse_clock(value)
    return None


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3.0)


def _prepare_presentation(
    directory: Path,
    rallies: Sequence[object],
    segments: Sequence[ExportSegment],
    metadata: VideoMetadata,
    options: Mapping[str, Any],
) -> tuple[list[Path], dict[str, Any]]:
    """Build optional artwork once so hardware retry reuses identical content."""
    active = options["aspect_ratio"] != "source" or any(options[key] for key in
        ("scoreboard", "include_notes", "include_stats", "overlay_path"))
    if not active:
        return [], {}
    source_width, source_height = metadata.display_resolution
    width, height = crop_dimensions(source_width, source_height, options["aspect_ratio"])
    images: list[Path] = []
    video_filters: dict[int, str] = {}
    overlays: dict[int, int] = {}
    scores = scores_before_rallies(rallies, options)
    for index, segment in enumerate(segments):
        rally = rallies[segment.source_index]
        video_filters[index] = crop_filter(source_width, source_height,
            options["aspect_ratio"], field(rally, "crop_keyframes", []), segment.start)
        lines: list[str] = []
        if options["scoreboard"]:
            a, b = scores[segment.source_index]
            lines.extend([f"{options['team_a']}  {a}", f"{options['team_b']}  {b}"])
        if options["include_notes"]:
            tag = " / ".join(str(field(rally, key, "")) for key in ("player", "outcome") if field(rally, key, ""))
            if tag:
                lines.append(tag)
            if field(rally, "note", ""):
                lines.append(str(field(rally, "note")))
        if lines:
            path = directory / f"rally-{index}.png"
            card_width = max(2, min(width - 8, int(width * .75)))
            card_height = max(20, int(min(height * .35, max(height * .055, width * .06) * len(lines))))
            render_card(path, card_width, card_height, lines, transparent=True)
            images.append(path)
            overlays[index] = len(images)
    filters: dict[str, Any] = {"video_filters": video_filters, "segment_overlays": overlays,
                              "output_size": (width, height), "fps": metadata.fps or 30}
    if options["overlay_path"]:
        images.append(Path(options["overlay_path"]))
        filters["custom_overlay_input"] = len(images)
    if options["include_stats"]:
        path = directory / "player-end-card.png"
        render_player_end_card(path, width, height, match_statistics(rallies, options))
        images.append(path)
        filters["stats_input"] = len(images)
        filters["stats_duration"] = options["stats_duration"]
    return images, filters


class FFmpegExporter:
    """Export rally clips with progress, cancellation, and codec fallback."""

    def __init__(
        self,
        ffmpeg_path: str | os.PathLike[str] | None = None,
        *,
        ffprobe_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._requested_ffmpeg_path = ffmpeg_path
        self._requested_ffprobe_path = ffprobe_path
        self._resolved_ffmpeg_path: str | None = None
        self._hardware_encoder_available: bool | None = None

    @property
    def ffmpeg_path(self) -> str:
        if self._resolved_ffmpeg_path is None:
            requested = self._requested_ffmpeg_path
            if requested is None:
                resolved = shutil.which("ffmpeg")
            else:
                supplied = os.fspath(requested)
                if os.sep not in supplied and (
                    os.altsep is None or os.altsep not in supplied
                ):
                    resolved = shutil.which(supplied)
                else:
                    candidate = Path(supplied).expanduser()
                    resolved = (
                        str(candidate.resolve())
                        if candidate.is_file() and os.access(candidate, os.X_OK)
                        else None
                    )
            if resolved is None:
                raise FFmpegNotFoundError(
                    "FFmpeg was not found. Install it with `brew install ffmpeg` "
                    "or pass ffmpeg_path explicitly."
                )
            self._resolved_ffmpeg_path = resolved
        return self._resolved_ffmpeg_path

    @property
    def ffprobe_path(self) -> str | os.PathLike[str] | None:
        if self._requested_ffprobe_path is not None:
            return self._requested_ffprobe_path
        sibling = Path(self.ffmpeg_path).with_name("ffprobe")
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return sibling
        return None

    def hardware_encoder_available(self) -> bool:
        """Return whether this macOS FFmpeg build exposes VideoToolbox H.264."""

        if self._hardware_encoder_available is not None:
            return self._hardware_encoder_available
        if platform.system() != "Darwin":
            self._hardware_encoder_available = False
            return False
        try:
            completed = subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            self._hardware_encoder_available = False
            return False
        encoder_listing = f"{completed.stdout}\n{completed.stderr}"
        self._hardware_encoder_available = (
            completed.returncode == 0 and "h264_videotoolbox" in encoder_listing
        )
        return self._hardware_encoder_available

    def _run_ffmpeg(
        self,
        command: Sequence[str],
        *,
        output_duration: float,
        progress_callback: ProgressCallback | None,
        cancel_callback: CancelCallback | None,
    ) -> _ProcessOutcome:
        try:
            process = subprocess.Popen(
                list(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise FFmpegNotFoundError(
                f"Could not start FFmpeg ({command[0]}): {exc}"
            ) from exc

        stdout_queue: Queue[str] = Queue()
        stderr_lines: list[str] = []

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                stdout_queue.put(line)
            process.stdout.close()

        def read_stderr() -> None:
            assert process.stderr is not None
            stderr_lines.extend(process.stderr.readlines())
            process.stderr.close()

        stdout_thread = Thread(target=read_stdout, name="ffmpeg-progress", daemon=True)
        stderr_thread = Thread(target=read_stderr, name="ffmpeg-errors", daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        last_progress = 0.0

        try:
            while process.poll() is None or stdout_thread.is_alive() or not stdout_queue.empty():
                if cancel_callback is not None:
                    try:
                        cancelled = bool(cancel_callback())
                    except Exception as exc:
                        _terminate_process(process)
                        raise ExportError(f"Export cancellation callback failed: {exc}") from exc
                    if cancelled:
                        _terminate_process(process)
                        raise ExportCancelledError("Video export was cancelled")
                try:
                    line = stdout_queue.get(timeout=0.1)
                except Empty:
                    continue
                key, separator, value = line.strip().partition("=")
                if not separator:
                    continue
                elapsed = _progress_seconds(key, value)
                if elapsed is not None and output_duration > 0:
                    progress = min(0.999, max(last_progress, elapsed / output_duration))
                    if progress_callback is not None and progress > last_progress:
                        try:
                            progress_callback(progress)
                        except Exception as exc:
                            _terminate_process(process)
                            raise ExportError(f"Export progress callback failed: {exc}") from exc
                    last_progress = progress
        finally:
            if process.poll() is None:
                _terminate_process(process)
            stdout_thread.join(timeout=2.0)
            stderr_thread.join(timeout=2.0)
        return _ProcessOutcome(process.returncode or 0, "".join(stderr_lines))

    def _probe_metadata(self, input_path: Path) -> VideoMetadata:
        try:
            return probe_video_metadata(
                input_path,
                ffprobe_path=self.ffprobe_path,
                allow_opencv_fallback=True,
            )
        except MetadataError as exc:
            raise ExportError(
                f"Cannot export because source metadata could not be read: {exc}"
            ) from exc

    def _detect_audio_stream(self, input_path: Path) -> bool:
        """Ask FFmpeg to decode one audio frame when OpenCV metadata is ambiguous."""

        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-frames:a",
            "1",
            "-f",
            "null",
            "-",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExportError(
                f"Timed out while checking the source audio stream: {input_path}"
            ) from exc
        except OSError as exc:
            raise FFmpegNotFoundError(
                f"Could not start FFmpeg ({self.ffmpeg_path}): {exc}"
            ) from exc
        if completed.returncode == 0:
            return True
        detail = _short_error(completed.stderr)
        if "matches no streams" in detail.lower() or "stream map" in detail.lower():
            return False
        raise ExportError(f"FFmpeg could not inspect the source audio stream: {detail}")

    def export(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        rallies: Iterable[object],
        *,
        prefer_hardware: bool = True,
        overwrite: bool = True,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
        metadata: VideoMetadata | None = None,
        has_audio: bool | None = None,
        export_options: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        """Export enabled rallies to a high-quality, broadly compatible MP4."""

        source = Path(input_path).expanduser()
        if not source.exists():
            raise VideoFileError(f"Video file does not exist: {source}")
        if not source.is_file():
            raise VideoFileError(f"Video path is not a file: {source}")
        source = source.resolve()
        destination = Path(output_path).expanduser()
        if destination.suffix.lower() != ".mp4":
            raise VideoFileError(f"Output path must end in .mp4: {destination}")
        if not destination.parent.exists():
            raise VideoFileError(
                f"Output directory does not exist: {destination.parent}"
            )
        if not destination.parent.is_dir():
            raise VideoFileError(f"Output parent is not a directory: {destination.parent}")
        destination = destination.resolve()
        if source == destination:
            raise VideoFileError("Output path must be different from the source video")
        if destination.exists() and not overwrite:
            raise VideoFileError(
                f"Output file already exists and overwrite is disabled: {destination}"
            )
        if cancel_callback is not None:
            try:
                cancelled_before_start = bool(cancel_callback())
            except Exception as exc:
                raise ExportError(f"Export cancellation callback failed: {exc}") from exc
            if cancelled_before_start:
                raise ExportCancelledError("Video export was cancelled before it started")

        try:
            options = prepare_export_options(export_options)
        except (ValueError, TypeError) as exc:
            raise ExportError(f"Invalid export options: {exc}") from exc
        all_rallies = list(rallies)
        ffmpeg_path = self.ffmpeg_path
        source_metadata = metadata or self._probe_metadata(source)
        segments = generate_export_segments(
            all_rallies, source_duration=source_metadata.duration_seconds,
            highlights_only=options["highlights_only"],
        )
        if not segments:
            raise ExportError("No enabled rallies are available to export")
        audio_present = source_metadata.has_audio if has_audio is None else has_audio
        if audio_present is None:
            # OpenCV cannot report audio, so make a cheap one-frame query rather
            # than dropping a source track or building a graph for a missing one.
            audio_present = self._detect_audio_stream(source)
        output_duration = sum(segment.duration for segment in segments) + (options["stats_duration"] if options["include_stats"] else 0.0)
        temporary = destination.with_name(
            f".{destination.stem}.partial-{uuid4().hex}.mp4"
        )
        hardware_requested = prefer_hardware and self.hardware_encoder_available()
        encoders = (
            ("h264_videotoolbox", "libx264")
            if hardware_requested
            else ("libx264",)
        )
        hardware_error: str | None = None
        last_reported_progress = 0.0

        def report_progress(progress: float) -> None:
            nonlocal last_reported_progress
            clamped = min(1.0, max(last_reported_progress, progress))
            if progress_callback is not None and clamped > last_reported_progress:
                progress_callback(clamped)
            last_reported_progress = clamped

        if progress_callback is not None:
            try:
                progress_callback(0.0)
            except Exception as exc:
                raise ExportError(f"Export progress callback failed: {exc}") from exc
        assets_directory = TemporaryDirectory(prefix=".roundnet-export-", dir=destination.parent)
        try:
            try:
                image_inputs, filter_options = _prepare_presentation(
                    Path(assets_directory.name), all_rallies, segments, source_metadata, options)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                raise ExportError(f"Could not prepare export presentation: {exc}") from exc
            for encoder in encoders:
                temporary.unlink(missing_ok=True)
                command = build_ffmpeg_command(
                    source,
                    temporary,
                    segments,
                    has_audio=bool(audio_present),
                    video_encoder=encoder,
                    ffmpeg_path=ffmpeg_path,
                    overwrite=True,
                    include_progress=True,
                    image_inputs=image_inputs,
                    filter_options=filter_options,
                )
                outcome = self._run_ffmpeg(
                    command,
                    output_duration=output_duration,
                    progress_callback=report_progress,
                    cancel_callback=cancel_callback,
                )
                if (
                    outcome.returncode == 0
                    and temporary.is_file()
                    and temporary.stat().st_size > 0
                ):
                    try:
                        report_progress(1.0)
                    except Exception as exc:
                        raise ExportError(
                            f"Export progress callback failed: {exc}"
                        ) from exc
                    # Install only after callbacks have accepted completion, so
                    # a callback failure cannot overwrite a previous good file.
                    os.replace(temporary, destination)
                    return ExportResult(
                        output_path=destination,
                        segment_count=len(segments),
                        output_duration_seconds=output_duration,
                        video_encoder=encoder,
                        used_hardware_acceleration=encoder == "h264_videotoolbox",
                    )
                if outcome.returncode == 0:
                    detail = (
                        "FFmpeg reported success but did not create a non-empty "
                        "temporary MP4"
                    )
                else:
                    detail = _short_error(outcome.stderr)
                if encoder == "h264_videotoolbox":
                    hardware_error = detail
                    LOGGER.warning(
                        "VideoToolbox export failed; retrying with libx264: %s", detail
                    )
                    continue
                combined_detail = detail
                if hardware_error is not None:
                    combined_detail = (
                        f"VideoToolbox attempt: {hardware_error} "
                        f"Software fallback: {detail}"
                    )
                raise ExportError(f"FFmpeg failed to export rally video: {combined_detail}")
        finally:
            temporary.unlink(missing_ok=True)
            assets_directory.cleanup()
        raise ExportError("FFmpeg failed to export rally video for an unknown reason")


def export_rallies(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    rallies: Iterable[object],
    *,
    prefer_hardware: bool = True,
    overwrite: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
    ffmpeg_path: str | os.PathLike[str] | None = None,
    ffprobe_path: str | os.PathLike[str] | None = None,
    export_options: Mapping[str, Any] | None = None,
) -> ExportResult:
    """Functional wrapper around :class:`FFmpegExporter`."""

    return FFmpegExporter(
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
    ).export(
        input_path,
        output_path,
        rallies,
        prefer_hardware=prefer_hardware,
        overwrite=overwrite,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        export_options=export_options,
    )


def is_ffmpeg_available(
    ffmpeg_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Return whether an executable FFmpeg can be resolved without running it."""

    try:
        _ = FFmpegExporter(ffmpeg_path=ffmpeg_path).ffmpeg_path
    except FFmpegNotFoundError:
        return False
    return True


def require_ffmpeg(
    ffmpeg_path: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve FFmpeg or raise :class:`FFmpegNotFoundError` with install help."""

    return FFmpegExporter(ffmpeg_path=ffmpeg_path).ffmpeg_path


# Friendly aliases for adapters/tests that use alternate terminology.
segments_from_rallies = generate_export_segments
build_export_command = build_ffmpeg_command
check_ffmpeg_available = is_ffmpeg_available
VideoExporter = FFmpegExporter


__all__ = [
    "CancelCallback",
    "ExportResult",
    "ExportSegment",
    "FFmpegExporter",
    "ProgressCallback",
    "build_export_command",
    "build_ffmpeg_command",
    "build_filter_complex",
    "check_ffmpeg_available",
    "export_rallies",
    "generate_export_segments",
    "is_ffmpeg_available",
    "require_ffmpeg",
    "segments_from_rallies",
    "VideoExporter",
]
