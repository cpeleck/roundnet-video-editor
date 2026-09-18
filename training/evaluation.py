"""Public reference import and event evaluation entry points."""

from pathlib import Path

from .calibration import evaluate_intervals, import_source_intervals


def load_reference_intervals(path: str | Path, *, fps: float | None = None) -> list[dict[str, float]]:
    """Read edited source-time intervals; see importer for EDL limitations."""
    return import_source_intervals(path, fps=fps)


__all__ = ["evaluate_intervals", "load_reference_intervals"]
