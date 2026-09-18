"""Background workers used by the desktop interface.

The workers deliberately keep all expensive detection and FFmpeg work off the
GUI thread.  Backend imports are deferred until ``run`` so importing the UI can
still produce a useful error dialog when an optional runtime dependency is
missing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import re
import threading
from typing import Any, Mapping, Sequence

from PySide6.QtCore import QThread, Signal


def _coerce_detection_settings(values: Any) -> Any:
    """Return a DetectionSettings instance without assuming one config API.

    Keeping this adapter here lets the settings dialog expose UI-only options
    (debug display and hardware export) without passing those keys to the
    detector constructor.
    """

    from config import DetectionSettings

    if isinstance(values, DetectionSettings):
        return values
    raw = dict(values or {}) if isinstance(values, Mapping) else {}
    raw.pop("debug_mode", None)
    raw.pop("prefer_hardware", None)
    raw.pop("save_labels_after_export", None)

    allowed = {field.name for field in fields(DetectionSettings)}
    filtered = {key: value for key, value in raw.items() if key in allowed}
    return DetectionSettings.from_dict(filtered)


class DetectionWorker(QThread):
    """Run rally detection and report GUI-safe progress signals."""

    progress = Signal(int, str)
    rally_count = Signal(int)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        video_path: str,
        roi: tuple[float, float, float, float] | None,
        settings: Any,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.roi = roi
        self.settings = settings
        self.court_context = None
        self.profile_path = ""
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def _is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _on_progress(self, *args: Any) -> None:
        if not args:
            return
        fraction: float
        message = "Analyzing video…"
        try:
            fraction = float(args[0])
        except (TypeError, ValueError):
            fraction = 0.0
            message = str(args[0])
        if len(args) > 1 and args[1] is not None:
            message = str(args[1])
        # Accept either the documented 0..1 scale or a legacy 0..100 value.
        percent = round(fraction if fraction > 1.0 else fraction * 100.0)
        percent = max(0, min(100, percent))
        self.progress.emit(percent, message)
        match = re.search(r"\b(\d+)\s+rall(?:y|ies)\b", message, re.I)
        if match:
            self.rally_count.emit(int(match.group(1)))

    def run(self) -> None:
        try:
            from detection import RoundnetDetector

            detector = RoundnetDetector(_coerce_detection_settings(self.settings))
            result = detector.analyze(
                self.video_path,
                roi=self.roi,
                progress_callback=self._on_progress,
                cancel_callback=self._is_cancelled,
                court_context=self.court_context,
            )
            if self.profile_path and not self._is_cancelled():
                from training.calibration import load_profile, predict_profile
                from detection.post_processing import segment_rallies
                try:
                    profile = load_profile(self.profile_path)
                    learned = predict_profile(profile, result)
                    learned_settings = deepcopy(detector.settings)
                    learned_settings.rally_threshold = profile["start_threshold"]
                    learned_settings.end_threshold = profile["end_threshold"]
                    rallies = segment_rallies(result.timestamps, learned, learned_settings,
                                             video_duration=result.duration, serve_scores=result.serve_scores)
                    # Keep the original heuristic traces for honest later comparison.
                    result = replace(result, rallies=rallies)
                    object.__setattr__(result, "learned_scores", learned)
                    object.__setattr__(result, "scoring_source", str(self.profile_path))
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    result = replace(result, warnings=(*result.warnings, f"Local profile unavailable; default detector used: {exc}"))
            if self._is_cancelled():
                self.cancelled.emit()
                return
            rallies = list(getattr(result, "rallies", ()) or ())
            self.rally_count.emit(len(rallies))
            self.progress.emit(100, f"Detection complete — {len(rallies)} rallies")
            self.succeeded.emit(result)
        except Exception as exc:  # errors must cross the thread boundary as text
            if self._is_cancelled() or "cancel" in type(exc).__name__.lower():
                self.cancelled.emit()
            else:
                self.failed.emit(f"{type(exc).__name__}: {exc}")


class ExportWorker(QThread):
    """Export enabled rally ranges with FFmpeg in the background."""

    progress = Signal(int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        input_path: str,
        output_path: str,
        rallies: Sequence[Any],
        prefer_hardware: bool = True,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.output_path = output_path
        self.rallies = list(rallies)
        self.rallies = deepcopy(self.rallies)
        self.prefer_hardware = bool(prefer_hardware)
        self.export_options = None
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def _is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _on_progress(self, value: Any, *_: Any) -> None:
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            fraction = 0.0
        percent = round(fraction if fraction > 1.0 else fraction * 100.0)
        percent = max(0, min(100, percent))
        self.progress.emit(percent, f"Exporting rally video… {percent}%")

    def run(self) -> None:
        try:
            from video.exporter import export_rallies

            result = export_rallies(
                self.input_path,
                self.output_path,
                self.rallies,
                prefer_hardware=self.prefer_hardware,
                overwrite=True,
                progress_callback=self._on_progress,
                cancel_callback=self._is_cancelled,
                export_options=self.export_options,
            )
            if self._is_cancelled():
                self.cancelled.emit()
                return
            self.progress.emit(100, "Export complete")
            self.succeeded.emit(result)
        except Exception as exc:
            if self._is_cancelled() or "cancel" in type(exc).__name__.lower():
                self.cancelled.emit()
            else:
                self.failed.emit(f"{type(exc).__name__}: {exc}")
