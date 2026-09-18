"""End-to-end orchestration and command-line interface for rally detection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from os import PathLike
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

from config import DetectionSettings
from models import Rally
from ._callbacks import (
    CancelCallback,
    DetectionCancelled,
    ProgressCallback,
    check_cancelled,
    report_progress,
)
from .audio_detector import AudioDetector
from .motion_detector import MotionDetector, ROI
from .post_processing import segment_rallies
from .rally_scoring import RallyScorer, robust_normalize
from .serve_detector import ServeDetector
from .court_context import normalize_court_context


@dataclass(frozen=True)
class DetectionResult:
    """Rallies plus aligned debug signals for timeline visualization."""

    rallies: list[Rally]
    timestamps: np.ndarray
    motion_scores: np.ndarray
    roi_motion_scores: np.ndarray
    audio_scores: np.ndarray
    temporal_scores: np.ndarray
    rally_scores: np.ndarray
    duration: float
    warnings: tuple[str, ...] = ()
    motion_spread_scores: np.ndarray | None = None
    serve_scores: np.ndarray | None = None
    serve_context_scores: np.ndarray | None = None
    player_count_scores: np.ndarray | None = None
    readiness_scores: np.ndarray | None = None
    player_motion_scores: np.ndarray | None = None
    retrieval_scores: np.ndarray | None = None
    pose_serve_scores: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_count = self.timestamps.size
        for name in (
            "motion_scores",
            "roi_motion_scores",
            "audio_scores",
            "temporal_scores",
            "rally_scores",
        ):
            values = getattr(self, name)
            if values.size != sample_count:
                raise ValueError(f"{name} must have one value per timestamp")
        for name in (
            "motion_spread_scores",
            "serve_scores",
            "serve_context_scores",
            "player_count_scores", "readiness_scores", "player_motion_scores",
            "retrieval_scores", "pose_serve_scores",
        ):
            values = getattr(self, name)
            if values is None:
                object.__setattr__(
                    self, name, np.zeros(sample_count, dtype=np.float64)
                )
            elif values.size != sample_count:
                raise ValueError(f"{name} must have one value per timestamp")

    @property
    def total_motion_scores(self) -> np.ndarray:
        return self.motion_scores

    @property
    def scores(self) -> np.ndarray:
        """Short alias for the final combined rally scores."""

        return self.rally_scores

    @property
    def duration_seconds(self) -> float:
        return self.duration

    def to_dict(self, *, include_signals: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible result, optionally including debug arrays."""

        data: dict[str, Any] = {
            "duration": self.duration,
            "rally_count": len(self.rallies),
            "rallies": [rally.to_dict() for rally in self.rallies],
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }
        if include_signals:
            data["signals"] = {
                "timestamps": self.timestamps.tolist(),
                "motion": self.motion_scores.tolist(),
                "roi_motion": self.roi_motion_scores.tolist(),
                "audio": self.audio_scores.tolist(),
                "temporal": self.temporal_scores.tolist(),
                "motion_spread": self.motion_spread_scores.tolist(),
                "serve": self.serve_scores.tolist(),
                "serve_context": self.serve_context_scores.tolist(),
                "rally_score": self.rally_scores.tolist(),
                "player_count": self.player_count_scores.tolist(),
                "readiness": self.readiness_scores.tolist(),
                "player_motion": self.player_motion_scores.tolist(),
                "retrieval": self.retrieval_scores.tolist(),
                "pose_serve": self.pose_serve_scores.tolist(),
            }
        return data


class RoundnetDetector:
    """Coordinate feature extraction, scoring, and temporal segmentation."""

    def __init__(self, settings: DetectionSettings | None = None) -> None:
        self.settings = settings or DetectionSettings()
        self.settings.validate()

    def analyze(
        self,
        video_path: str | PathLike[str],
        roi: ROI | Any | None = None,
        *,
        court_context: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> DetectionResult:
        """Analyze ``video_path`` and return editable rallies and debug scores.

        Progress callbacks normally receive ``(fraction, message)``.  A legacy
        one-argument ``callback(fraction)`` is also supported.  Cancellation is
        cooperative: when ``cancel_callback`` returns true,
        :class:`DetectionCancelled` is raised.
        """

        path = Path(video_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {path}")
        check_cancelled(cancel_callback)
        court = normalize_court_context(court_context)
        report_progress(progress_callback, 0.0, "Starting rally detection")

        motion_detector = MotionDetector(self.settings)
        motion = motion_detector.analyze(
            path,
            roi,
            court_context=court,
            progress_callback=_scaled_progress(
                progress_callback, 0.0, 0.72, "Motion: "
            ),
            cancel_callback=cancel_callback,
        )

        check_cancelled(cancel_callback)
        audio_detector = AudioDetector(self.settings)
        audio = audio_detector.analyze(
            path,
            motion.timestamps,
            progress_callback=_scaled_progress(
                progress_callback, 0.72, 0.88, "Audio: "
            ),
            cancel_callback=cancel_callback,
        )

        check_cancelled(cancel_callback)
        report_progress(progress_callback, 0.90, "Combining activity signals")
        # Serve detection consumes normalized visual/audio evidence but remains
        # independent of the final rally score.  Audio is only a bonus inside
        # ServeDetector, so videos without a usable microphone still receive a
        # visual quiet-to-distributed-motion start cue.
        normalized_roi = np.clip(
            robust_normalize(motion.roi_motion) * self.settings.motion_sensitivity,
            0.0,
            1.0,
        )
        normalized_audio = np.clip(
            robust_normalize(audio.transient_activity)
            * self.settings.audio_sensitivity,
            0.0,
            1.0,
        )
        spread_values = np.asarray(motion.motion_spread, dtype=np.float64)
        serve_series = ServeDetector(self.settings).score(
            motion.timestamps,
            normalized_roi,
            spread_values,
            normalized_audio,
            readiness=motion.readiness,
            player_motion=motion.player_motion,
            retrieval=motion.retrieval,
            pose_serve=motion.pose_serve,
        )
        score_series = RallyScorer(self.settings).score(
            motion.timestamps,
            motion.total_motion,
            motion.roi_motion,
            audio.transient_activity,
            motion_spread=spread_values,
            serve=serve_series.context,
        )

        check_cancelled(cancel_callback)
        report_progress(progress_callback, 0.95, "Finding rally boundaries")
        # Unknown/occluded players contribute zeros and leave the legacy score
        # unchanged. Context has a bounded, conservative influence on evidence.
        combined = score_series.combined.copy()
        if court is not None:
            strength = self.settings.court_context_strength
            combined = np.clip(combined * (1 - strength * motion.retrieval * (1 - spread_values))
                               + strength * motion.readiness * motion.player_motion * (1 - combined), 0, 1)
        rallies = segment_rallies(
            score_series.timestamps,
            combined,
            self.settings,
            video_duration=motion.duration,
            motion_spread_scores=score_series.motion_spread,
            serve_scores=serve_series.likelihood,
        )
        warnings = ((audio.warning,) if audio.warning else ()) + motion.warnings
        result = DetectionResult(
            rallies=rallies,
            timestamps=score_series.timestamps,
            motion_scores=score_series.motion,
            roi_motion_scores=score_series.roi_motion,
            audio_scores=score_series.audio,
            temporal_scores=score_series.temporal,
            rally_scores=combined,
            duration=motion.duration,
            warnings=warnings,
            motion_spread_scores=score_series.motion_spread,
            serve_scores=serve_series.likelihood,
            serve_context_scores=serve_series.context,
            player_count_scores=motion.player_count,
            readiness_scores=motion.readiness,
            player_motion_scores=motion.player_motion,
            retrieval_scores=motion.retrieval,
            pose_serve_scores=motion.pose_serve,
            metadata={"court_context": court, "player_backend": motion.player_backend,
                      "player_tracks": list(motion.player_tracks),
                      "pose_notes": "Pose cues are observed joint motion, not a trained serve/action classifier."},
        )
        report_progress(
            progress_callback,
            1.0,
            f"Detection complete: {len(rallies)} rallies found",
        )
        return result

    # Familiar aliases make the detector easy to integrate with UI workers and
    # leave room for a future model-backed implementation of the same contract.
    detect = analyze

    def detect_rallies(
        self,
        video_path: str | PathLike[str],
        roi: ROI | Any | None = None,
        *,
        court_context: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> list[Rally]:
        return self.analyze(
            video_path,
            roi,
            court_context=court_context,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        ).rallies


# More general name for future callers; RoundnetDetector remains the descriptive
# primary API.
RallyDetector = RoundnetDetector


def detect_rallies(
    video_path: str | PathLike[str],
    roi: ROI | Any | None = None,
    settings: DetectionSettings | None = None,
    *,
    court_context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> list[Rally]:
    """Detect rallies without explicitly constructing a detector."""

    return RoundnetDetector(settings).detect_rallies(
        video_path,
        roi,
        court_context=court_context,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


def analyze_video(
    video_path: str | PathLike[str],
    roi: ROI | Any | None = None,
    settings: DetectionSettings | None = None,
    *,
    court_context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> DetectionResult:
    """Functional wrapper returning the complete :class:`DetectionResult`."""

    return RoundnetDetector(settings).analyze(
        video_path,
        roi,
        court_context=court_context,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m detection.detector",
        description="Detect active roundnet rallies in a local video.",
    )
    parser.add_argument("video", type=Path, help="source MP4/MOV video")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="playing-area rectangle in source pixels or normalized 0-1 values",
    )
    parser.add_argument("--analysis-fps", type=float, help="sample rate for analysis")
    parser.add_argument("--threshold", type=float, help="rally start threshold (0-1)")
    parser.add_argument("--end-threshold", type=float, help="rally end threshold (0-1)")
    parser.add_argument("--min-duration", type=float, help="minimum rally duration")
    parser.add_argument("--max-duration", type=float, help="maximum rally duration")
    parser.add_argument("--pre-roll", type=float, help="seconds retained before rallies")
    parser.add_argument("--post-roll", type=float, help="seconds retained after rallies")
    parser.add_argument("--merge-gap", type=float, help="maximum gap to merge in seconds")
    parser.add_argument(
        "--output-json",
        type=Path,
        metavar="PATH",
        help="write JSON to a file instead of stdout",
    )
    parser.add_argument(
        "--include-signals",
        action="store_true",
        help="include sampled debug signal arrays in JSON output",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="do not display progress on stderr"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m detection.detector VIDEO``."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    overrides = {
        "analysis_fps": args.analysis_fps,
        "rally_threshold": args.threshold,
        "end_threshold": args.end_threshold,
        "min_rally_duration": args.min_duration,
        "max_rally_duration": args.max_duration,
        "pre_roll": args.pre_roll,
        "post_roll": args.post_roll,
        "merge_gap": args.merge_gap,
    }
    try:
        settings = DetectionSettings().updated(
            **{name: value for name, value in overrides.items() if value is not None}
        )
        progress = None if args.quiet else _console_progress
        result = RoundnetDetector(settings).analyze(
            args.video,
            tuple(args.roi) if args.roi else None,
            progress_callback=progress,
        )
        payload = result.to_dict(include_signals=args.include_signals)
        payload["video"] = str(args.video)
        payload["settings"] = settings.to_dict()
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        if args.output_json:
            args.output_json.expanduser().write_text(serialized + "\n", encoding="utf-8")
            if not args.quiet:
                print(f"Wrote {args.output_json}", file=sys.stderr)
        else:
            print(serialized)
        return 0
    except DetectionCancelled:
        print("Detection cancelled.", file=sys.stderr)
        return 130
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _scaled_progress(
    callback: ProgressCallback | None,
    start: float,
    end: float,
    prefix: str,
) -> ProgressCallback | None:
    if callback is None:
        return None

    def update(fraction: float, message: str = "") -> None:
        report_progress(callback, start + (end - start) * fraction, prefix + message)

    return update


def _console_progress(fraction: float, message: str) -> None:
    print(f"[{fraction * 100:6.2f}%] {message}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())


__all__ = [
    "DetectionCancelled",
    "DetectionResult",
    "RallyDetector",
    "RoundnetDetector",
    "analyze_video",
    "build_argument_parser",
    "detect_rallies",
    "main",
]
