"""Modular heuristic rally detection.

The end-to-end detector is loaded lazily so running
``python -m detection.detector`` does not import the target module twice.
"""

from __future__ import annotations

from typing import Any


_DETECTOR_EXPORTS = {
    "DetectionResult",
    "RallyDetector",
    "RoundnetDetector",
    "analyze_video",
    "detect_rallies",
}
_SERVE_EXPORTS = {"ServeDetector", "ServeLikelihoodSeries"}


def __getattr__(name: str) -> Any:
    if name == "DetectionCancelled":
        from ._callbacks import DetectionCancelled

        return DetectionCancelled
    if name in _DETECTOR_EXPORTS:
        from . import detector

        return getattr(detector, name)
    if name in _SERVE_EXPORTS:
        from . import serve_detector

        return getattr(serve_detector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals()) | _DETECTOR_EXPORTS | _SERVE_EXPORTS | {"DetectionCancelled"}
    )


__all__ = [
    "DetectionCancelled",
    "DetectionResult",
    "RallyDetector",
    "RoundnetDetector",
    "ServeDetector",
    "ServeLikelihoodSeries",
    "analyze_video",
    "detect_rallies",
]
