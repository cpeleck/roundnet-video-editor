"""Exceptions raised by the video input/output layer."""

from __future__ import annotations


class VideoIOError(RuntimeError):
    """Base class for actionable media errors shown by the application."""


class VideoFileError(VideoIOError):
    """The requested input or output file is not usable."""


class FFmpegNotFoundError(VideoIOError):
    """FFmpeg (or its companion ffprobe executable) could not be found."""


class FFprobeNotFoundError(FFmpegNotFoundError):
    """The ffprobe executable could not be found."""


class MetadataError(VideoIOError):
    """Video metadata could not be read or was invalid."""


class FrameExtractionError(VideoIOError):
    """A preview or analysis frame could not be decoded."""


class FrameExtractionCancelledError(FrameExtractionError):
    """Frame iteration was cancelled by the caller."""


class ExportError(VideoIOError):
    """FFmpeg could not create the requested rally video."""


class ExportCancelledError(ExportError):
    """The caller cancelled an in-progress export."""
