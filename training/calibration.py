"""Explicit local supervised learning with recording-group validation.

Profiles are numeric JSON, never executable model pickles. The held-out
recording is excluded from scaling, fitting, and all threshold selection.
The final profile is fitted on all supplied recordings only after evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from config import DetectionSettings
from detection.post_processing import segment_rallies
from .feedback import (
    ANNOTATION_ALL_RALLIES, ANNOTATION_UNKNOWN, FEEDBACK_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION, _atomic_write_text, dense_features_from_result,
    make_sample_labels,
)

PROFILE_SCHEMA_VERSION = 1
MIN_RECORDINGS = 3
MIN_CLASS_SAMPLES = 20


@dataclass(frozen=True)
class _Recording:
    source_id: str
    path: Path
    metadata: dict[str, Any]
    timestamps: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    names: tuple[str, ...]
    duration: float
    complete: bool


def load_profile(path: str | Path) -> dict[str, Any]:
    """Read and strictly validate a portable numeric model profile."""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_profile(profile)
    return profile


def predict_profile(profile: Mapping[str, Any], signal_data: Any) -> np.ndarray:
    """Return one learned activity score per input timestamp.

    Missing or different detector features are errors, not silently filled
    columns. A new feature schema requires explicitly retraining the profile.
    """
    _validate_profile(profile)
    dense = dense_features_from_result(signal_data)
    expected = tuple(profile["feature_names"])
    if dense.names != expected:
        raise ValueError(
            "Profile feature mismatch. Re-analyze feedback with this detector and retrain. "
            f"Expected {expected}; received {dense.names}."
        )
    return _predict_matrix(profile, dense.values)


def train_from_feedback(
    paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    regularization: float = 0.05,
    progress: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate by whole recording, fit a final model, and save JSON.

    At least three unique source fingerprints and twenty reviewed samples of
    each class are required. Multiple corrections of one source are deduplicated
    to the latest saved revision. Event metrics use only explicitly complete
    recordings with no intentionally omitted rally intervals.
    """
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    destination = Path(output_path).expanduser()
    if not destination.parent.is_dir():
        raise ValueError("The profile destination folder does not exist.")
    if destination.suffix.lower() != ".json":
        raise ValueError("Save a local model profile with a .json filename.")
    if destination.resolve() in {Path(path).expanduser().resolve() for path in paths}:
        raise ValueError("Save the profile separately from correction files.")
    warnings: list[str] = []
    recordings = _load_recordings(paths, warnings)
    labeled_recordings = [item for item in recordings if np.any(item.labels >= 0)]
    if len(labeled_recordings) != len(recordings):
        warnings.append("Recordings with no explicitly reviewed sample interiors were excluded from training and validation.")
    recordings = labeled_recordings
    for recording in recordings:
        protected = [recording.path.parent / recording.metadata["features"]["file"]]
        source_path = recording.metadata.get("source", {}).get("path")
        if isinstance(source_path, str) and source_path:
            protected.append(Path(source_path).expanduser())
        if destination.resolve() in {path.resolve() for path in protected}:
            raise ValueError("Save the profile separately from source videos and detector features.")
    if len(recordings) < MIN_RECORDINGS:
        raise ValueError(
            f"Training needs at least {MIN_RECORDINGS} distinct original recordings; "
            f"found {len(recordings)}. More edits of the same video do not add a validation group."
        )
    names = recordings[0].names
    if any(item.names != names for item in recordings):
        raise ValueError("Feedback feature schemas differ. Re-analyze all recordings with the same detector.")
    if not any(name != "rally_score" for name in names):
        raise ValueError("Feedback must include original activity features, not only the final rally score.")
    all_labels = np.concatenate([item.labels[item.labels >= 0] for item in recordings])
    counts = {str(value): int(np.count_nonzero(all_labels == value)) for value in (0, 1)}
    if min(counts.values()) < MIN_CLASS_SAMPLES:
        raise ValueError(
            f"Training needs at least {MIN_CLASS_SAMPLES} reviewed samples per class; "
            f"found {counts['1']} rally and {counts['0']} non-rally samples. "
            "Reject false detections or explicitly confirm a fully reviewed recording."
        )
    if len(recordings) < 5:
        warnings.append("Fewer than five recordings: validation is preliminary and camera diversity may be limited.")
    if sum(item.duration for item in recordings) < 600:
        warnings.append("Less than ten minutes of source footage: collect longer representative games before relying on this profile.")

    folds: list[dict[str, Any]] = []
    learned_sample = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    heuristic_sample = dict(learned_sample)
    learned_events: list[dict[str, Any]] = []
    heuristic_events: list[dict[str, Any]] = []
    for index, held_out in enumerate(recordings):
        _check_cancel(cancelled)
        if progress:
            progress(f"Validating recording {index + 1} of {len(recordings)}: {held_out.path.name}")
        training = [item for item in recordings if item.source_id != held_out.source_id]
        profile = _fit(training, names, regularization, cancelled)
        predicted = _predict_matrix(profile, held_out.features)
        baseline = _baseline_scores(held_out)
        learned_counts = _sample_counts(held_out.labels, predicted >= 0.5)
        config = _recording_settings(held_out.metadata)
        heuristic_counts = _sample_counts(held_out.labels, baseline >= config.rally_threshold)
        for key in learned_sample:
            learned_sample[key] += learned_counts[key]
            heuristic_sample[key] += heuristic_counts[key]
        fold: dict[str, Any] = {
            "held_out_source_id": held_out.source_id,
            "training_source_ids": [item.source_id for item in training],
            "source_file": held_out.path.name,
            "sample_count": int(np.count_nonzero(held_out.labels >= 0)),
            "learned_samples": _classification_metrics(learned_counts),
            "heuristic_samples": _classification_metrics(heuristic_counts),
        }
        if held_out.complete and not held_out.metadata.get("final_disabled_intervals"):
            truth = held_out.metadata.get("final_enabled_intervals", [])
            learned_config = _recording_settings(held_out.metadata)
            learned_config.rally_threshold = 0.5
            learned_config.end_threshold = 0.35
            learned = segment_rallies(
                held_out.timestamps, predicted, learned_config, video_duration=held_out.duration,
            )
            # Recompute baseline from the saved original heuristic signal, so
            # edited or learned-enabled initial clip lists cannot contaminate it.
            baseline_rallies = segment_rallies(
                held_out.timestamps, baseline, config, video_duration=held_out.duration,
                motion_spread_scores=_column(held_out, "motion_spread"),
                serve_scores=_column(held_out, "serve_likelihood"),
            )
            fold["learned_events"] = evaluate_intervals(learned, truth, held_out.duration)
            fold["heuristic_events"] = evaluate_intervals(baseline_rallies, truth, held_out.duration)
            learned_events.append(fold["learned_events"])
            heuristic_events.append(fold["heuristic_events"])
        else:
            fold["event_metrics_unavailable"] = "Requires explicit full-review confirmation and no excluded true rallies."
        folds.append(fold)

    if not learned_events:
        warnings.append("Event metrics unavailable: no fully reviewed recording without omitted rallies. Sample metrics use only explicit labels.")
    if progress:
        progress("Fitting the final profile using all reviewed recordings…")
    _check_cancel(cancelled)
    final_profile = _fit(recordings, names, regularization, cancelled)
    report: dict[str, Any] = {
        "validation_method": "leave_one_source_recording_out",
        "recording_count": len(recordings),
        "reviewed_sample_count": int(all_labels.size),
        "positive_samples": counts["1"],
        "negative_samples": counts["0"],
        "learned_samples": _classification_metrics(learned_sample),
        "heuristic_samples": _classification_metrics(heuristic_sample),
        "learned_events": _aggregate_events(learned_events),
        "heuristic_events": _aggregate_events(heuristic_events),
        "folds": folds,
        "warnings": warnings,
        "profile_path": str(destination.resolve()),
        "boundary_semantics": "Absolute error against reviewed export boundaries, including intended padding; not contact-time accuracy.",
        "thresholds": {"learned_start": 0.5, "learned_end": 0.35},
    }
    final_profile.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_source_ids": [item.source_id for item in recordings],
        "report": report,
    })
    _validate_profile(final_profile)
    _check_cancel(cancelled)
    _atomic_write_text(destination, json.dumps(final_profile, indent=2, allow_nan=False) + "\n")
    return report


def evaluate_intervals(
    predicted: Sequence[Any], expected: Sequence[Any], duration_seconds: float,
    *, minimum_iou: float = 0.3,
) -> dict[str, Any]:
    """One-to-one event matching, false hits/hour, and export-boundary error.

    Matching maximizes the number of valid overlapping pairs, so a long clip
    spanning several true rallies can count as at most one detection.
    """
    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    if not math.isfinite(minimum_iou) or not 0 < minimum_iou <= 1:
        raise ValueError("minimum_iou must be in (0, 1]")
    detections = _intervals(predicted, duration)
    truth = _intervals(expected, duration)
    adjacency = []
    for start, end in detections:
        options = []
        for j, (left, right) in enumerate(truth):
            intersection = max(0.0, min(end, right) - max(start, left))
            union = (end - start) + (right - left) - intersection
            iou = intersection / union
            if iou >= minimum_iou:
                options.append((iou, j))
        adjacency.append([j for _, j in sorted(options, reverse=True)])
    matched: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False

    for i in range(len(detections)):
        augment(i, set())
    errors = [
        (abs(detections[i][0] - truth[j][0]), abs(detections[i][1] - truth[j][1]))
        for j, i in matched.items()
    ]
    tp = len(matched)
    fp, fn = len(detections) - tp, len(truth) - tp
    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_positives_per_hour": fp * 3600.0 / duration,
        "mean_start_error_seconds": float(np.mean([e[0] for e in errors])) if errors else None,
        "mean_end_error_seconds": float(np.mean([e[1] for e in errors])) if errors else None,
        "evaluated_duration_seconds": duration,
        "matching_iou": minimum_iou,
    }


def import_source_intervals(path: str | Path, *, fps: float | None = None) -> list[dict[str, float]]:
    """Import source-time JSON or a single-reel, non-drop-frame CMX EDL.

    JSON accepts a list, ``rallies``, ``final_enabled_intervals``, or this app's
    ``roundnet_edit_decisions`` export. EDL uses
    source IN/OUT (not montage record IN/OUT); fps must be specified explicitly.
    Source timecode must start at zero. Nonzero reel offsets and multiple reels
    require a full timeline interchange importer and are rejected/unsupported.
    """
    source = Path(path)
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            if "edits" in data:
                if data.get("format") != "roundnet_edit_decisions" or data.get("schema_version") != 1 or data.get("time_unit") != "seconds":
                    raise ValueError("Unsupported edit-decision JSON; use this app's source-time export.")
                if not isinstance(data["edits"], list) or any(not isinstance(edit, Mapping) for edit in data["edits"]):
                    raise ValueError("Edit decisions must be a list of source-time ranges.")
                data = [{"start": edit.get("source_start"), "end": edit.get("source_end")}
                        for edit in data["edits"]]
            elif "final_enabled_intervals" in data:
                data = data["final_enabled_intervals"]
            elif "rallies" in data:
                data = data["rallies"]
            elif "intervals" in data:
                data = data["intervals"]
            else:
                raise ValueError("JSON must contain source-time rallies or intervals.")
        return [{"start": a, "end": b} for a, b in _intervals(data, None)]
    if source.suffix.lower() != ".edl":
        raise ValueError("Use source-time JSON or a CMX .edl file.")
    if fps is None or not math.isfinite(fps) or fps <= 0:
        raise ValueError("EDL import requires the original source frame rate.")
    content = source.read_text(encoding="utf-8-sig")
    if re.search(r"\d{2}:\d{2}:\d{2};\d{2}", content) or re.search(r"FCM:\s*DROP FRAME", content, re.IGNORECASE):
        raise ValueError("Drop-frame EDL is not supported; export source-time JSON instead.")
    events: list[dict[str, float]] = []
    reels: set[str] = set()
    clip_names: set[str] = set()
    source_files: set[str] = set()
    for line in content.splitlines():
        if match := re.match(r"\*\s*FROM CLIP NAME:\s*(.+)", line, re.IGNORECASE):
            clip_names.add(match.group(1).strip())
        if match := re.match(r"\*\s*SOURCE FILE:\s*(.+)", line, re.IGNORECASE):
            source_files.add(match.group(1).strip())
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) < 8:
            raise ValueError("Malformed EDL event.")
        if "V" not in parts[2].upper():
            continue
        if parts[3] != "C":
            raise ValueError("Only straight-cut EDL events are supported.")
        reels.add(parts[1])
        events.append({"start": _edl_time(parts[4], fps), "end": _edl_time(parts[5], fps)})
    if len(reels) > 1 or len(clip_names) > 1 or len(source_files) > 1:
        raise ValueError("EDL references multiple source reels; select a single-source edit.")
    if not events:
        raise ValueError("No video cut events found in EDL.")
    return [{"start": a, "end": b} for a, b in _intervals(events, None)]


def _load_recordings(paths: Sequence[str | Path], warnings: list[str]) -> list[_Recording]:
    selected: dict[str, _Recording] = {}
    for path_value in paths:
        path = Path(path_value).expanduser().resolve()
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping) or metadata.get("schema_version") != FEEDBACK_SCHEMA_VERSION or metadata.get("kind") != "roundnet_training_feedback":
            raise ValueError(f"{path.name}: expected schema-v2 feedback saved by the app.")
        feature_meta = metadata.get("features")
        if not isinstance(feature_meta, Mapping) or feature_meta.get("schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"{path.name}: missing or incompatible detector feature sidecar. Analyze then save corrections.")
        feature_file = feature_meta.get("file")
        if not isinstance(feature_file, str) or Path(feature_file).name != feature_file:
            raise ValueError(f"{path.name}: feature sidecar must be a filename next to its JSON.")
        with np.load(path.parent / feature_file, allow_pickle=False) as arrays:
            if int(arrays["schema_version"]) != FEATURE_SCHEMA_VERSION:
                raise ValueError(f"{path.name}: incompatible NPZ schema.")
            source_id = str(arrays["source_id"])
            names = tuple(str(item) for item in arrays["feature_names"].tolist())
            timestamps = np.array(arrays["timestamps"], dtype=np.float64)
            values = np.array(arrays["features"], dtype=np.float64)
        if source_id != metadata.get("source", {}).get("id") or not source_id:
            raise ValueError(f"{path.name}: source fingerprint differs between JSON and NPZ.")
        if names != tuple(feature_meta.get("feature_names", [])) or not names or len(set(names)) != len(names):
            raise ValueError(f"{path.name}: feature names differ between JSON and NPZ.")
        if timestamps.ndim != 1 or values.shape != (timestamps.size, len(names)):
            raise ValueError(f"{path.name}: feature matrix is not aligned to timestamps.")
        if timestamps.size < 2 or not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0) or timestamps[0] < 0:
            raise ValueError(f"{path.name}: invalid sample timestamps.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path.name}: features contain non-finite values.")
        duration = float(metadata["source"]["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0 or timestamps[-1] >= duration + 1e-6:
            raise ValueError(f"{path.name}: timestamps fall outside the source duration.")
        complete = metadata.get("annotation_completeness") == ANNOTATION_ALL_RALLIES and metadata.get("annotation_completeness_confirmed") is True
        if metadata.get("annotation_completeness") == ANNOTATION_ALL_RALLIES and not complete:
            warnings.append(f"{path.name}: old full-review claim was not explicitly confirmed; unlabeled background ignored. Re-save after confirming review to include it.")
        reviewed = metadata.get("reviewed_intervals")
        if reviewed is None:
            reviewed = [dict(item, enabled=True) for item in metadata.get("final_enabled_intervals", [])]
            reviewed.extend(dict(item, enabled=False) for item in metadata.get("final_disabled_intervals", []))
        # Rebuild labels from provenance instead of trusting stale arrays.
        settings = metadata.get("detection_settings") or {}
        labels = make_sample_labels(
            timestamps, reviewed, rejected_detections=metadata.get("rejected_detections", []),
            annotation_completeness=ANNOTATION_ALL_RALLIES if complete else ANNOTATION_UNKNOWN,
            boundary_ignore_seconds=float(feature_meta.get("boundary_ignore_seconds", 0.25)),
            pre_roll_seconds=float(feature_meta.get("pre_roll_ignore_seconds", settings.get("pre_roll", 0.0))),
            post_roll_seconds=float(feature_meta.get("post_roll_ignore_seconds", settings.get("post_roll", 0.0))),
        )
        recording = _Recording(source_id, path, metadata, timestamps, values, labels, names, duration, complete)
        old = selected.get(source_id)
        if old is not None:
            warnings.append(f"Duplicate original video: used the latest saved correction for {path.name}; counted as one recording.")
            if str(metadata.get("created_utc", "")) <= str(old.metadata.get("created_utc", "")):
                continue
        selected[source_id] = recording
    return sorted(selected.values(), key=lambda item: item.source_id)


def _fit(recordings: Sequence[_Recording], names: tuple[str, ...], regularization: float, cancelled: Callable[[], bool] | None) -> dict[str, Any]:
    matrices, targets, sample_weights = [], [], []
    for item in recordings:
        indices = np.flatnonzero(item.labels >= 0)
        # Deterministic time-ordered subsampling bounds memory/CPU for long
        # tournaments. Every recording still has equal total training weight.
        if indices.size > 12000:
            indices = indices[np.linspace(0, indices.size - 1, 12000, dtype=int)]
        if indices.size:
            matrices.append(item.features[indices])
            targets.append(item.labels[indices])
            sample_weights.append(np.full(indices.size, 1.0 / indices.size))
    if not targets:
        raise ValueError("No reviewed training samples remain after excluding boundaries.")
    x = np.concatenate(matrices)
    y = np.concatenate(targets).astype(np.float64)
    if np.unique(y).size < 2:
        raise ValueError("A held-out fold leaves only one training class. Add rally and rejected/non-rally examples across more recordings.")
    weights = np.concatenate(sample_weights)
    weights /= weights.sum()
    # Equal class mass makes minority non-rally retrieval examples influential.
    for value in (0, 1):
        mask = y == value
        weights[mask] *= 0.5 / weights[mask].sum()
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale = np.maximum(scale, 1e-4)
    z = np.clip((x - mean) / scale, -20, 20)
    excluded = [names.index("rally_score")] if "rally_score" in names else []
    if excluded:
        z[:, excluded] = 0.0
    design = np.column_stack([z, np.ones(z.shape[0])])
    beta = np.zeros(design.shape[1])
    penalty = np.ones(beta.size) * regularization
    penalty[-1] = 0.0
    # Damped Newton updates converge quickly with this small feature set.
    for iteration in range(80):
        _check_cancel(cancelled)
        logits = design @ beta
        p = _sigmoid(logits)
        gradient = design.T @ (weights * (p - y)) + penalty * beta
        hessian = (design.T * (weights * p * (1 - p))) @ design + np.diag(penalty + 1e-8)
        step = np.linalg.solve(hessian, gradient)
        old_loss = float(np.dot(weights, np.logaddexp(0, logits) - y * logits) + 0.5 * np.dot(penalty, beta ** 2))
        fraction = 1.0
        while fraction > 1e-6:
            candidate = beta - fraction * step
            new_logits = design @ candidate
            new_loss = float(np.dot(weights, np.logaddexp(0, new_logits) - y * new_logits) + 0.5 * np.dot(penalty, candidate ** 2))
            if new_loss <= old_loss:
                beta = candidate
                break
            fraction *= 0.5
        if np.linalg.norm(fraction * step) < 1e-7:
            break
    return {
        "kind": "roundnet_local_activity_profile",
        "schema_version": PROFILE_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(names),
        "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients": beta[:-1].tolist(), "intercept": float(beta[-1]),
        "regularization": regularization,
        "start_threshold": 0.5, "end_threshold": 0.35,
        "excluded_evidence_features": ["rally_score"] if excluded else [],
        "score_semantics": "Class-balanced learned activity score, not a calibrated probability of an entire rally.",
    }


def _validate_profile(profile: Any) -> None:
    if not isinstance(profile, Mapping) or profile.get("kind") != "roundnet_local_activity_profile":
        raise ValueError("Not a Roundnet local activity profile.")
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION or profile.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Unsupported profile schema; retrain using this app version.")
    names = profile.get("feature_names")
    if not isinstance(names, list) or not names or len(names) > 128 or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("Profile has invalid feature names.")
    for key in ("mean", "scale", "coefficients"):
        try:
            array = np.asarray(profile.get(key), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Profile {key} must be finite numbers.") from exc
        if array.shape != (len(names),) or not np.all(np.isfinite(array)):
            raise ValueError(f"Profile {key} must have one finite number per feature.")
        if key == "scale" and np.any(array <= 0):
            raise ValueError("Profile scale must be positive.")
    for key in ("intercept", "start_threshold", "end_threshold"):
        try:
            number = float(profile[key])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Profile {key} is invalid.") from exc
        if not math.isfinite(number):
            raise ValueError(f"Profile {key} must be finite.")
    if not 0 <= float(profile["end_threshold"]) < float(profile["start_threshold"]) <= 1:
        raise ValueError("Profile thresholds are inconsistent.")


def _predict_matrix(profile: Mapping[str, Any], values: np.ndarray) -> np.ndarray:
    z = np.clip((values - np.asarray(profile["mean"])) / np.asarray(profile["scale"]), -20, 20)
    return _sigmoid(z @ np.asarray(profile["coefficients"]) + float(profile["intercept"]))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _check_cancel(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise RuntimeError("Training cancelled; no profile was saved.")


def _baseline_scores(item: _Recording) -> np.ndarray:
    column = _column(item, "rally_score")
    if column is None:
        raise ValueError(f"{item.path.name}: missing original rally_score for baseline comparison.")
    return column


def _column(item: _Recording, name: str) -> np.ndarray | None:
    return item.features[:, item.names.index(name)] if name in item.names else None


def _recording_settings(metadata: Mapping[str, Any]) -> DetectionSettings:
    allowed = {field.name for field in fields(DetectionSettings)}
    config = DetectionSettings(**{key: value for key, value in (metadata.get("detection_settings") or {}).items() if key in allowed})
    config.validate()
    return config


def _sample_counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.count_nonzero((labels == 1) & predictions)),
        "fp": int(np.count_nonzero((labels == 0) & predictions)),
        "fn": int(np.count_nonzero((labels == 1) & ~predictions)),
        "tn": int(np.count_nonzero((labels == 0) & ~predictions)),
    }


def _classification_metrics(counts: Mapping[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    return dict(counts, precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None)


def _aggregate_events(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not metrics:
        return None
    tp = sum(m["true_positives"] for m in metrics)
    fp = sum(m["false_positives"] for m in metrics)
    fn = sum(m["false_negatives"] for m in metrics)
    duration = sum(m["evaluated_duration_seconds"] for m in metrics)
    result = {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
              "precision": tp / (tp + fp) if tp + fp else None,
              "recall": tp / (tp + fn) if tp + fn else None,
              "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
              "false_positives_per_hour": fp * 3600 / duration,
              "evaluated_duration_seconds": duration, "recording_count": len(metrics)}
    for key in ("mean_start_error_seconds", "mean_end_error_seconds"):
        result[key] = sum(m[key] * m["true_positives"] for m in metrics if m[key] is not None) / tp if tp else None
    return result


def _intervals(items: Sequence[Any], duration: float | None) -> list[tuple[float, float]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("Intervals must be a list of source-time start/end pairs.")
    result = []
    for item in items:
        if isinstance(item, Mapping):
            if not item.get("enabled", True):
                continue
            a, b = item.get("start", item.get("start_time")), item.get("end", item.get("end_time"))
        elif hasattr(item, "start_time"):
            if not getattr(item, "enabled", True):
                continue
            a, b = item.start_time, item.end_time
        else:
            a, b = item
        try:
            start, end = float(a), float(b)
        except (TypeError, ValueError) as exc:
            raise ValueError("Intervals must contain numeric source start/end times.") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("Invalid source interval: expected 0 <= start < end.")
        if duration is not None and end > duration + 1e-6:
            raise ValueError("An interval extends beyond the source duration.")
        result.append((start, end))
    return sorted(result)


def _edl_time(timecode: str, fps: float) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}):(\d{2})", timecode)
    if match is None:
        raise ValueError("Expected non-drop-frame HH:MM:SS:FF source timecode.")
    hours, minutes, seconds, frames = map(int, match.groups())
    nominal_fps = int(round(fps))
    if minutes >= 60 or seconds >= 60 or frames >= nominal_fps:
        raise ValueError("Invalid EDL source timecode.")
    return ((hours * 3600 + minutes * 60 + seconds) * nominal_fps + frames) / fps


__all__ = ["load_profile", "predict_profile", "train_from_feedback", "evaluate_intervals", "import_source_intervals"]
