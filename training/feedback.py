"""Persist corrected rally decisions as local, training-ready feedback.

The JSON sidecar is intentionally human-readable and keeps the source-time
intervals useful without NumPy.  When detector signals are supplied, a second
compressed NPZ file stores a dense feature matrix and aligned {-1, 0, 1}
labels.  Neither artifact contains video frames or audio samples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from os import PathLike
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


FEEDBACK_SCHEMA_VERSION = 2
FEATURE_SCHEMA_VERSION = 1
DEFAULT_BOUNDARY_IGNORE_SECONDS = 0.25

ANNOTATION_ALL_RALLIES = "all_rallies"
ANNOTATION_HIGHLIGHTS_ONLY = "highlights_only"
ANNOTATION_UNKNOWN = "unknown"
ANNOTATION_COMPLETENESS_VALUES = frozenset(
    {
        ANNOTATION_ALL_RALLIES,
        ANNOTATION_HIGHLIGHTS_ONLY,
        ANNOTATION_UNKNOWN,
    }
)

LABEL_IGNORED = -1
LABEL_NON_RALLY = 0
LABEL_RALLY = 1


@dataclass(frozen=True)
class DenseFeatureSet:
    """Detector signals aligned into a single numeric matrix."""

    timestamps: np.ndarray
    values: np.ndarray
    names: tuple[str, ...]

    @property
    def sample_count(self) -> int:
        return int(self.timestamps.size)


@dataclass(frozen=True)
class FeedbackSaveResult:
    """Paths and label counts produced by :func:`save_feedback`."""

    metadata_path: Path
    features_path: Path | None
    source_id: str
    sample_count: int
    positive_count: int
    negative_count: int
    ignored_count: int


# Canonical feature names remain stable even when DetectionResult attribute
# names evolve.  Missing optional signals are omitted and the exact ordered
# feature list is recorded in both artifacts.
_FEATURE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("motion", ("motion_scores", "motion")),
    ("roi_motion", ("roi_motion_scores", "roi_motion")),
    (
        "motion_spread",
        ("motion_spread_scores", "spatial_motion_scores", "motion_spread"),
    ),
    ("audio", ("audio_scores", "audio")),
    ("temporal", ("temporal_scores", "temporal")),
    (
        "serve_likelihood",
        ("serve_likelihood_scores", "serve_scores", "serve_likelihood", "serve"),
    ),
    ("serve_context", ("serve_context_scores", "serve_context")),
    ("player_count", ("player_count_scores", "player_count")),
    ("readiness", ("readiness_scores", "readiness")),
    ("player_motion", ("player_motion_scores", "player_motion")),
    ("retrieval", ("retrieval_scores", "retrieval")),
    ("pose_serve", ("pose_serve_scores", "pose_serve")),
    ("rally_score", ("rally_scores", "rally_score", "combined", "scores")),
)

_MISSING = object()


def fingerprint_source(
    source_path: str | PathLike[str],
    *,
    sample_size: int = 1024 * 1024,
) -> dict[str, Any]:
    """Return a stable, inexpensive content fingerprint for a source video.

    Large videos are identified by their byte length plus the first and last
    ``sample_size`` bytes.  Small files are read in full.  This is deliberately
    much cheaper than hashing a multi-gigabyte recording on the GUI thread, but
    still survives file moves and renames.
    """

    if isinstance(sample_size, bool) or int(sample_size) != sample_size or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source video does not exist: {path}")

    before = path.stat()
    size = int(before.st_size)
    chunk_size = int(sample_size)
    digest = hashlib.sha256()
    digest.update(b"roundnet-source-fingerprint-v1\0")
    digest.update(size.to_bytes(16, byteorder="big", signed=False))

    with path.open("rb") as source:
        if size <= chunk_size * 2:
            digest.update(b"full\0")
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
            strategy = "full"
        else:
            digest.update(b"head-tail\0")
            digest.update(source.read(chunk_size))
            source.seek(size - chunk_size)
            digest.update(source.read(chunk_size))
            strategy = "head_tail"

    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError("source video changed while its fingerprint was being computed")
    return {
        "algorithm": "sha256-head-tail-v1",
        "digest": digest.hexdigest(),
        "size_bytes": size,
        "sample_size_bytes": chunk_size,
        "strategy": strategy,
    }


def dense_features_from_result(signal_data: Any) -> DenseFeatureSet:
    """Collect known aligned signals from a detector result or signal mapping."""

    if signal_data is None:
        raise TypeError("signal_data cannot be None")
    source = signal_data
    nested = _read_value(signal_data, ("signals",), default=None)
    if isinstance(nested, Mapping):
        source = nested

    raw_timestamps = _read_value(source, ("timestamps",), default=_MISSING)
    if raw_timestamps is _MISSING:
        raise ValueError("signal_data must contain timestamps")
    timestamps = _one_dimensional_float("timestamps", raw_timestamps)
    if np.any(~np.isfinite(timestamps)) or (timestamps.size and timestamps[0] < 0):
        raise ValueError("timestamps must be finite and non-negative")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    columns: list[np.ndarray] = []
    names: list[str] = []
    for canonical_name, aliases in _FEATURE_FIELDS:
        raw_values = _read_value(source, aliases, default=_MISSING)
        if raw_values is _MISSING or raw_values is None:
            continue
        values = _one_dimensional_float(canonical_name, raw_values)
        if values.size != timestamps.size:
            raise ValueError(
                f"{canonical_name} must have one item per timestamp "
                f"({values.size} != {timestamps.size})"
            )
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{canonical_name} contains non-finite values")
        columns.append(values.astype(np.float32, copy=False))
        names.append(canonical_name)

    if not columns:
        raise ValueError("signal_data does not contain any recognized feature signals")
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return DenseFeatureSet(
        timestamps=timestamps.astype(np.float64, copy=False),
        values=matrix,
        names=tuple(names),
    )


def make_sample_labels(
    timestamps: Sequence[float] | np.ndarray,
    final_intervals: Sequence[Any],
    *,
    rejected_detections: Sequence[Any] = (),
    annotation_completeness: str = ANNOTATION_UNKNOWN,
    boundary_ignore_seconds: float = DEFAULT_BOUNDARY_IGNORE_SECONDS,
    pre_roll_seconds: float = 0.0,
    post_roll_seconds: float = 0.0,
) -> np.ndarray:
    """Create aligned labels where -1=ignore, 0=non-rally, and 1=rally.

    Enabled ranges are positive export selections, not asserted play-core
    boundaries. Disabled ranges are always ignored: in the current editor they
    mean "omit from export", not "false positive". For a fully reviewed video,
    background outside all ranges is negative. For a highlight-only or unknown
    annotation, unreviewed background is ignored and only explicitly rejected
    detector ranges become negative. A narrow band around enabled range edges
    is ignored because review/export cuts commonly contain pre/post-roll rather
    than exact contact times.
    """

    times = _one_dimensional_float("timestamps", timestamps)
    if np.any(~np.isfinite(times)) or (times.size and times[0] < 0):
        raise ValueError("timestamps must be finite and non-negative")
    if times.size > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    completeness = _normalize_completeness(annotation_completeness)
    margin = _finite_non_negative("boundary_ignore_seconds", boundary_ignore_seconds)
    pre_roll = _finite_non_negative("pre_roll_seconds", pre_roll_seconds)
    post_roll = _finite_non_negative("post_roll_seconds", post_roll_seconds)

    base = LABEL_NON_RALLY if completeness == ANNOTATION_ALL_RALLIES else LABEL_IGNORED
    labels = np.full(times.size, base, dtype=np.int8)
    rejected = _normalize_plain_intervals(rejected_detections)
    final = _normalize_review_intervals(final_intervals, duration=None)

    # Disabled is an export preference rather than a negative annotation.
    # Pending detector proposals are not supervised examples until explicitly
    # reviewed (or the user confirms reviewing the complete recording).
    for interval in final:
        if not interval["enabled"] or (
            not interval["reviewed"] and completeness != ANNOTATION_ALL_RALLIES
        ):
            labels[_interval_mask(times, interval["start"], interval["end"])] = LABEL_IGNORED

    # Explicit user rejections are trustworthy negative examples even when the
    # rest of a video contains only highlight annotations.
    for interval in rejected:
        labels[_interval_mask(times, interval["start"], interval["end"])] = LABEL_NON_RALLY

    enabled_final = [
        interval for interval in final
        if interval["enabled"] and (interval["reviewed"] or completeness == ANNOTATION_ALL_RALLIES)
    ]
    for interval in enabled_final:
        start = interval["start"]
        end = interval["end"]
        labels[_interval_mask(times, start, end)] = LABEL_RALLY

    if margin > 0:
        for interval in enabled_final:
            start = interval["start"]
            end = interval["end"]
            # Keep a usable interior even for unusually short manual ranges.
            local_margin = min(margin, (end - start) * 0.20)
            if local_margin <= 0:
                continue
            boundary_mask = (
                ((times >= start - local_margin) & (times < start + local_margin))
                | ((times >= end - local_margin) & (times <= end + local_margin))
            )
            labels[boundary_mask] = LABEL_IGNORED

    # Export ranges contain viewing padding, which must never teach the model
    # that waiting before or after a point is active play.  If padding consumes
    # a short clip entirely, leave it unlabeled rather than inventing a core.
    if pre_roll > 0 or post_roll > 0:
        for interval in enabled_final:
            start, end = interval["start"], interval["end"]
            labels[_interval_mask(times, start, min(end, start + pre_roll + margin))] = LABEL_IGNORED
            labels[_interval_mask(times, max(start, end - post_roll - margin), end)] = LABEL_IGNORED

    return labels


def save_feedback(
    metadata_path: str | PathLike[str],
    *,
    source_path: str | PathLike[str],
    duration_seconds: float,
    roi: Sequence[float] | None,
    initial_predictions: Sequence[Any],
    final_intervals: Sequence[Any],
    rejected_detections: Sequence[Any] = (),
    annotation_completeness: str = ANNOTATION_UNKNOWN,
    signal_data: Any | None = None,
    detection_settings: Any | None = None,
    features_path: str | PathLike[str] | None = None,
    boundary_ignore_seconds: float = DEFAULT_BOUNDARY_IGNORE_SECONDS,
    created_utc: str | datetime | None = None,
) -> FeedbackSaveResult:
    """Save corrected intervals and optional aligned detector features.

    Args:
        metadata_path: Destination JSON sidecar.
        source_path: Original, unedited video used for source-time labels.
        duration_seconds: Source duration in seconds.
        roi: Optional four-number playing-area rectangle.
        initial_predictions: Immutable detector output before user edits.
        final_intervals: Final reviewed export ranges, including enabled flags.
            Disabled ranges are saved as ignored, never presumed false hits.
        rejected_detections: Detector ranges explicitly deleted as false hits.
        annotation_completeness: ``all_rallies``, ``highlights_only``, or
            ``unknown``. Only ``all_rallies`` makes all unlabeled time negative.
        signal_data: A DetectionResult-like object or ``signals`` mapping.
        detection_settings: JSON-compatible detector settings for provenance.
        features_path: Optional NPZ destination; defaults beside the JSON.
        boundary_ignore_seconds: Uncertain band around positive interval edges.
        created_utc: Optional ISO string/datetime, primarily for deterministic tests.

    Returns:
        A :class:`FeedbackSaveResult` with paths and class counts.
    """

    destination = Path(metadata_path).expanduser()
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"feedback destination folder does not exist: {destination.parent}")
    source = Path(source_path).expanduser().resolve()
    if destination.resolve() == source:
        raise ValueError("Save correction labels separately from the source video.")
    duration = _finite_positive("duration_seconds", duration_seconds)
    completeness = _normalize_completeness(annotation_completeness)
    margin = _finite_non_negative("boundary_ignore_seconds", boundary_ignore_seconds)
    settings_payload = _json_compatible(detection_settings)
    padding_settings = settings_payload if isinstance(settings_payload, Mapping) else {}
    pre_roll = _finite_non_negative("pre_roll", padding_settings.get("pre_roll", 0.0))
    post_roll = _finite_non_negative("post_roll", padding_settings.get("post_roll", 0.0))
    normalized_roi = _normalize_roi(roi)
    fingerprint = fingerprint_source(source)

    predictions = _normalize_predictions(initial_predictions, duration)
    reviewed = _normalize_review_intervals(final_intervals, duration)
    final = [
        {"start": item["start"], "end": item["end"]}
        for item in reviewed
        if item["enabled"]
    ]
    disabled = [
        {"start": item["start"], "end": item["end"]}
        for item in reviewed
        if not item["enabled"]
    ]
    rejected = _normalize_rejections(rejected_detections, predictions, duration)

    dense: DenseFeatureSet | None = None
    labels = np.empty(0, dtype=np.int8)
    npz_destination: Path | None = None
    if signal_data is not None:
        dense = dense_features_from_result(signal_data)
        if dense.timestamps.size and dense.timestamps[-1] >= duration + 1e-6:
            raise ValueError("Detector timestamps extend beyond the source duration.")
        labels = make_sample_labels(
            dense.timestamps,
            reviewed,
            rejected_detections=rejected,
            annotation_completeness=completeness,
            boundary_ignore_seconds=margin,
            pre_roll_seconds=pre_roll,
            post_roll_seconds=post_roll,
        )
        npz_destination = (
            Path(features_path).expanduser()
            if features_path is not None
            else destination.with_suffix(".npz")
        )
        if not npz_destination.parent.is_dir():
            raise FileNotFoundError(
                f"feature destination folder does not exist: {npz_destination.parent}"
            )
        if npz_destination.resolve() in {destination.resolve(), source}:
            raise ValueError("Save detector features separately from the video and correction JSON.")
        if npz_destination.parent.resolve() != destination.parent.resolve():
            raise ValueError("Save the detector feature sidecar beside its correction JSON.")
    elif features_path is not None:
        raise ValueError("features_path requires signal_data")

    source_id = fingerprint["digest"]
    timestamp = _normalize_created_utc(created_utc)
    label_counts = _label_counts(labels)
    feature_metadata: dict[str, Any] | None = None
    if dense is not None and npz_destination is not None:
        feature_metadata = {
            "format": "numpy-npz",
            "compressed": True,
            "file": npz_destination.name,
            "schema_version": FEATURE_SCHEMA_VERSION,
            "sample_count": dense.sample_count,
            "feature_names": list(dense.names),
            "matrix_shape": [dense.sample_count, len(dense.names)],
            "label_values": {
                "ignored": LABEL_IGNORED,
                "non_rally": LABEL_NON_RALLY,
                "rally": LABEL_RALLY,
            },
            "label_counts": label_counts,
            "boundary_ignore_seconds": margin,
            "pre_roll_ignore_seconds": pre_roll,
            "post_roll_ignore_seconds": post_roll,
        }

    source_metadata = {
        "id": source_id,
        "filename": source.name,
        "path": str(source),
        "duration_seconds": duration,
        "fingerprint": fingerprint,
    }
    payload: dict[str, Any] = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "kind": "roundnet_training_feedback",
        "created_utc": timestamp,
        "annotation_completeness": completeness,
        "annotation_completeness_confirmed": completeness == ANNOTATION_ALL_RALLIES,
        "source": source_metadata,
        "roi": normalized_roi,
        "detection_settings": settings_payload,
        "initial_predictions": predictions,
        "final_enabled_intervals": final,
        "final_disabled_intervals": disabled,
        "reviewed_intervals": reviewed,
        "rejected_detections": rejected,
        "features": feature_metadata,
        "interval_semantics": {
            "initial_predictions": (
                "Detector-produced export ranges; these may include configured pre/post-roll."
            ),
            "final_enabled_intervals": (
                "User-reviewed export selections; boundaries may include viewing padding and "
                "are not asserted play-core boundaries."
            ),
            "final_disabled_intervals": (
                "Ranges omitted from export. Disabled does not mean false detection, so their "
                "dense labels are ignored."
            ),
            "rejected_detections": (
                "Ranges explicitly deleted by the user as detector false positives."
            ),
        },
        # Compatibility aliases preserve the original v1 label consumer shape.
        "video": source.name,
        "video_path": str(source),
        "duration_seconds": duration,
        "rallies": [{"start": item["start"], "end": item["end"]} for item in final],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"

    if dense is not None and npz_destination is not None:
        _atomic_write_npz(
            npz_destination,
            timestamps=dense.timestamps,
            features=dense.values,
            labels=labels,
            feature_names=dense.names,
            source_id=source_id,
        )
    _atomic_write_text(destination, serialized)

    return FeedbackSaveResult(
        metadata_path=destination,
        features_path=npz_destination,
        source_id=source_id,
        sample_count=int(labels.size),
        positive_count=label_counts["rally"],
        negative_count=label_counts["non_rally"],
        ignored_count=label_counts["ignored"],
    )


def _normalize_predictions(items: Sequence[Any], duration: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, item in enumerate(items):
        interval = _normalize_interval(item, duration)
        raw_id = _read_value(item, ("prediction_id", "id", "rally_id"), default=None)
        prediction_id = str(raw_id).strip() if raw_id is not None else ""
        if not prediction_id:
            prediction_id = f"prediction-{index + 1:04d}"
        if prediction_id in used_ids:
            raise ValueError(f"duplicate initial prediction id: {prediction_id}")
        used_ids.add(prediction_id)
        confidence = _read_value(item, ("confidence",), default=1.0)
        confidence_value = _finite_unit_interval("confidence", confidence)
        enabled = bool(_read_value(item, ("enabled",), default=True))
        result.append(
            {
                "prediction_id": prediction_id,
                "start": interval["start"],
                "end": interval["end"],
                "confidence": confidence_value,
                "serve_confidence": _finite_unit_interval(
                    "serve_confidence", _read_value(item, ("serve_confidence",), default=0.0)
                ),
                "enabled": enabled,
            }
        )
    result.sort(key=lambda value: (value["start"], value["end"], value["prediction_id"]))
    return result


def _normalize_review_intervals(
    items: Sequence[Any], duration: float | None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items:
        interval: dict[str, Any] = _normalize_interval(item, duration)
        interval["enabled"] = bool(_read_value(item, ("enabled",), default=True))
        # Legacy caller-provided ranges were explicit manual annotations and
        # have no review flag. Modern Rally objects carry their actual state.
        interval["reviewed"] = bool(_read_value(item, ("reviewed",), default=True))
        interval["serve_confidence"] = _finite_unit_interval(
            "serve_confidence", _read_value(item, ("serve_confidence",), default=0.0)
        )
        result.append(interval)
    result.sort(key=lambda value: (value["start"], value["end"], not value["enabled"]))
    return result


def _normalize_rejections(
    items: Sequence[Any],
    predictions: Sequence[Mapping[str, Any]],
    duration: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        interval = _normalize_interval(item, duration)
        raw_id = _read_value(item, ("rejection_id", "id"), default=None)
        rejection_id = str(raw_id).strip() if raw_id is not None else ""
        if not rejection_id:
            rejection_id = f"rejection-{index + 1:04d}"
        source_prediction_id = _read_value(
            item, ("source_prediction_id", "prediction_id", "rally_id"), default=None
        )
        if source_prediction_id is None:
            source_prediction_id, matched_iou = _best_prediction_match(interval, predictions)
        else:
            source_prediction_id = str(source_prediction_id)
            matched_iou = _matching_iou(interval, source_prediction_id, predictions)
        reason = _read_value(item, ("reason",), default="user_deleted")
        entry: dict[str, Any] = {
            "rejection_id": rejection_id,
            "start": interval["start"],
            "end": interval["end"],
            "reason": str(reason or "user_deleted"),
            "source_prediction_id": source_prediction_id,
            "serve_confidence": _finite_unit_interval(
                "serve_confidence", _read_value(item, ("serve_confidence",), default=0.0)
            ),
        }
        if matched_iou is not None:
            entry["matched_iou"] = round(float(matched_iou), 6)
        result.append(entry)
    result.sort(key=lambda value: (value["start"], value["end"], value["rejection_id"]))
    return result


def _normalize_plain_intervals(
    items: Sequence[Any], *, enabled_only: bool = False
) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for item in items:
        if enabled_only and not bool(_read_value(item, ("enabled",), default=True)):
            continue
        result.append(_normalize_interval(item, duration=None))
    return result


def _normalize_interval(item: Any, duration: float | None) -> dict[str, float]:
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray, Mapping)):
        if len(item) != 2:
            raise ValueError("interval sequences must contain exactly start and end")
        raw_start, raw_end = item
    else:
        raw_start = _read_value(item, ("start", "start_time"), default=_MISSING)
        raw_end = _read_value(item, ("end", "end_time"), default=_MISSING)
        if raw_start is _MISSING or raw_end is _MISSING:
            raise ValueError("interval must contain start/end or start_time/end_time")
    start = _finite_non_negative("interval start", raw_start)
    end = _finite_positive("interval end", raw_end)
    if duration is not None:
        if start >= duration:
            raise ValueError("interval starts at or after the source duration")
        end = min(end, duration)
    if end <= start:
        raise ValueError("interval end must be greater than start")
    return {"start": float(start), "end": float(end)}


def _best_prediction_match(
    interval: Mapping[str, float], predictions: Sequence[Mapping[str, Any]]
) -> tuple[str | None, float | None]:
    best_id: str | None = None
    best_iou = 0.0
    for prediction in predictions:
        score = _interval_iou(interval, prediction)
        if score > best_iou:
            best_iou = score
            best_id = str(prediction["prediction_id"])
    return (best_id, best_iou) if best_id is not None and best_iou > 0 else (None, None)


def _matching_iou(
    interval: Mapping[str, float],
    prediction_id: str,
    predictions: Sequence[Mapping[str, Any]],
) -> float | None:
    for prediction in predictions:
        if str(prediction.get("prediction_id")) == prediction_id:
            return _interval_iou(interval, prediction)
    return None


def _interval_iou(first: Mapping[str, float], second: Mapping[str, Any]) -> float:
    intersection = max(
        0.0,
        min(float(first["end"]), float(second["end"]))
        - max(float(first["start"]), float(second["start"])),
    )
    union = max(float(first["end"]), float(second["end"])) - min(
        float(first["start"]), float(second["start"])
    )
    return intersection / union if union > 0 else 0.0


def _interval_mask(timestamps: np.ndarray, start: float, end: float) -> np.ndarray:
    return (timestamps >= start) & (timestamps < end)


def _normalize_roi(roi: Sequence[float] | None) -> list[float] | None:
    if roi is None:
        return None
    if isinstance(roi, (str, bytes, bytearray)) or len(roi) != 4:
        raise ValueError("roi must contain x, y, width, and height")
    values = [_finite_number("roi value", value) for value in roi]
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("roi width and height must be positive")
    return values


def _normalize_completeness(value: str) -> str:
    aliases = {"complete": ANNOTATION_ALL_RALLIES, "partial": ANNOTATION_HIGHLIGHTS_ONLY}
    normalized = aliases.get(str(value), str(value))
    if normalized not in ANNOTATION_COMPLETENESS_VALUES:
        choices = ", ".join(sorted(ANNOTATION_COMPLETENESS_VALUES))
        raise ValueError(f"annotation_completeness must be one of: {choices}")
    return normalized


def _normalize_created_utc(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        moment = value
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat()
    text = str(value).strip()
    if not text:
        raise ValueError("created_utc cannot be empty")
    return text


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("settings cannot contain non-finite numbers")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_compatible(value.to_dict())
    if is_dataclass(value):
        return _json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _read_value(source: Any, names: Sequence[str], *, default: Any) -> Any:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        try:
            value = getattr(source, name)
        except (AttributeError, TypeError):
            continue
        if value is not None:
            return value
    return default


def _one_dimensional_float(name: str, values: Any) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _finite_non_negative(name: str, value: Any) -> float:
    result = _finite_number(name, value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_positive(name: str, value: Any) -> float:
    result = _finite_number(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_unit_interval(name: str, value: Any) -> float:
    result = _finite_number(name, value)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "ignored": int(np.count_nonzero(labels == LABEL_IGNORED)),
        "non_rally": int(np.count_nonzero(labels == LABEL_NON_RALLY)),
        "rally": int(np.count_nonzero(labels == LABEL_RALLY)),
    }


def _atomic_write_text(destination: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_npz(
    destination: Path,
    *,
    timestamps: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    source_id: str,
) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype=np.int16),
                timestamps=np.asarray(timestamps, dtype=np.float64),
                features=np.asarray(features, dtype=np.float32),
                labels=np.asarray(labels, dtype=np.int8),
                feature_names=np.asarray(tuple(feature_names), dtype=np.str_),
                source_id=np.asarray(source_id, dtype=np.str_),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "ANNOTATION_ALL_RALLIES",
    "ANNOTATION_COMPLETENESS_VALUES",
    "ANNOTATION_HIGHLIGHTS_ONLY",
    "ANNOTATION_UNKNOWN",
    "DEFAULT_BOUNDARY_IGNORE_SECONDS",
    "DenseFeatureSet",
    "FEEDBACK_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "FeedbackSaveResult",
    "LABEL_IGNORED",
    "LABEL_NON_RALLY",
    "LABEL_RALLY",
    "dense_features_from_result",
    "fingerprint_source",
    "make_sample_labels",
    "save_feedback",
]
