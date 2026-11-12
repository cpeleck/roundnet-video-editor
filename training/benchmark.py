"""Benchmark operations, kept separate from accuracy measured on held-out data."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median

from .calibration import train_from_feedback


def correction_counts(state: dict) -> dict | None:
    """Net interval corrections, not clicks, scoring edits, or export choices."""
    if not state.get("signals"):
        return None
    if any(not item.get("rally_id") for item in state.get("initial_predictions", [])):
        return None  # Older proposals without stable identities cannot be counted reliably.
    original = {item["rally_id"]: item for item in state.get("initial_predictions", []) if item.get("enabled", True)}
    final = {item["rally_id"]: item for item in state["rallies"] if not item.get("rejected", False)}
    added, removed = len(final.keys() - original.keys()), len(original.keys() - final.keys())
    trimmed = sum(any(abs(final[key][field] - original[key][field]) > 0.001
                      for field in ("start_time", "end_time")) for key in original.keys() & final.keys())
    return {"added": added, "removed": removed, "trimmed": trimmed, "total": added + removed + trimmed}


def run_benchmark(paths, profile_path, *, boundary_tolerance_seconds=1.0, progress=None, cancelled=None):
    report = train_from_feedback(paths, profile_path, boundary_tolerance_seconds=boundary_tolerance_seconds,
                                 progress=progress, cancelled=cancelled)
    # train_from_feedback deduplicates by content fingerprint. Use only the
    # exact held-out sources/revisions that were evaluated, once per source.
    selected = {}
    for path in paths:
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        source_id = metadata["source"]["id"]
        if source_id not in selected or metadata["created_utc"] > selected[source_id]["created_utc"]:
            selected[source_id] = metadata
    rows, normalized_times, measured_times, corrections = [], [], [], []
    for fold in report["folds"]:
        metadata = selected[fold["held_out_source_id"]]
        benchmark = metadata.get("benchmark", {})
        runtime = benchmark.get("detection_run") or {}
        elapsed = runtime.get("detection_elapsed_seconds")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            elapsed = None
        duration = fold["duration_seconds"]
        if elapsed is not None:
            normalized_times.append(elapsed * 1500 / duration)
            if 1440 <= duration <= 1560:
                measured_times.append(elapsed)
        counts = benchmark.get("corrections")
        if counts is not None:
            corrections.append(counts["total"])
        rows.append({"source_id": fold["held_out_source_id"], "title": fold["source_title"],
                     "duration_seconds": duration, "detection_seconds": elapsed,
                     "corrections": counts, "detection_run": runtime,
                     "project_id": benchmark.get("project_id"), "revision": benchmark.get("revision")})
    report.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operations": {"recordings": rows,
                       "timed_recording_count": len(normalized_times),
                       "measured_25_minute_count": len(measured_times),
                       "median_measured_25_minute_seconds": median(measured_times) if measured_times else None,
                       "estimated_25_minute_seconds": median(normalized_times) if normalized_times else None,
                       "median_corrections_per_recording": median(corrections) if corrections else None,
                       "correction_recording_count": len(corrections)},
        "correction_definition": "Net interval changes from initially enabled proposals: added/restored + removed/rejected + trimmed (both boundaries count once). Splits/merges appear as removed and added intervals. Scoring, review clicks, and export exclusions do not count.",
        "runtime_definition": "Full detector analysis wall time, excluding worker startup/queueing. Measured 25-minute sources span 24–26 minutes. Estimate scales each timed recording to 1500 seconds, then takes the median; it is not a measured 25-minute run or learned-model runtime.",
    })
    if report["recording_count"] < 8:
        report["warnings"].append("Collect 8–10 varied, fully labeled recordings before using these numbers as your project benchmark.")
    return report
