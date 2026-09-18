"""Shared callback helpers for long-running detection work."""

from __future__ import annotations

import inspect
from typing import Callable, TypeAlias


ProgressCallback: TypeAlias = Callable[..., None]
CancelCallback: TypeAlias = Callable[[], bool]


class DetectionCancelled(RuntimeError):
    """Raised when a caller cancels video analysis."""


def report_progress(
    callback: ProgressCallback | None, fraction: float, message: str
) -> None:
    """Invoke either a two-argument or legacy one-argument callback."""

    if callback is None:
        return
    value = min(1.0, max(0.0, float(fraction)))
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        callback(value, message)
        return
    try:
        signature.bind(value, message)
    except TypeError:
        callback(value)
    else:
        callback(value, message)


def check_cancelled(callback: CancelCallback | None) -> None:
    if callback is not None and callback():
        raise DetectionCancelled("rally detection was cancelled")


__all__ = [
    "CancelCallback",
    "DetectionCancelled",
    "ProgressCallback",
    "check_cancelled",
    "report_progress",
]

