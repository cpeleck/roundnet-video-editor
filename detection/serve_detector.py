"""Estimate likely serve starts from quiet-to-active visual and audio transitions.

This is deliberately an evidence signal, not a hard gate.  A distant microphone
or an occluded server must not make an otherwise obvious rally disappear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from config import DetectionSettings


@dataclass(frozen=True)
class ServeLikelihoodSeries:
    """Raw serve likelihood and its short-lived start-confirmation context."""

    timestamps: np.ndarray
    likelihood: np.ndarray
    context: np.ndarray

    def __post_init__(self) -> None:
        expected = self.timestamps.size
        if self.likelihood.size != expected or self.context.size != expected:
            raise ValueError("serve signals must have one item per timestamp")


class ServeDetector:
    """Find quiet-to-impact/motion transitions that resemble a serve."""

    def __init__(self, settings: DetectionSettings | None = None) -> None:
        self.settings = settings or DetectionSettings()
        self.settings.validate()

    def score(
        self,
        timestamps: Sequence[float] | np.ndarray,
        roi_motion: Sequence[float] | np.ndarray,
        motion_spread: Sequence[float] | np.ndarray,
        audio: Sequence[float] | np.ndarray,
        *,
        readiness: Sequence[float] | np.ndarray | None = None,
        player_motion: Sequence[float] | np.ndarray | None = None,
        retrieval: Sequence[float] | np.ndarray | None = None,
        pose_serve: Sequence[float] | np.ndarray | None = None,
    ) -> ServeLikelihoodSeries:
        """Return normalized likelihood that each sample begins with a serve.

        The visual term rewards a quiet setup followed by rapid, distributed
        activity around the playing area.  A nearby audio transient strengthens
        the estimate, but visual evidence can still produce a useful score when
        the source has no usable audio.
        """

        times = _signal("timestamps", timestamps)
        roi = _signal("roi_motion", roi_motion, times.size)
        spread = _signal("motion_spread", motion_spread, times.size)
        audio_values = _signal("audio", audio, times.size)
        context_signals = {
            name: np.clip(np.nan_to_num(_signal(name, value, times.size)), 0, 1)
            if value is not None else np.zeros(times.size)
            for name, value in (("readiness", readiness), ("player_motion", player_motion),
                                ("retrieval", retrieval), ("pose_serve", pose_serve))
        }
        if times.size == 0:
            empty = np.empty(0, dtype=np.float64)
            return ServeLikelihoodSeries(times, empty, empty)
        if np.any(~np.isfinite(times)) or times[0] < 0:
            raise ValueError("timestamps must be finite and non-negative")
        if times.size > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("timestamps must be strictly increasing")

        roi = np.clip(np.nan_to_num(roi), 0.0, 1.0)
        spread = np.clip(np.nan_to_num(spread), 0.0, 1.0)
        audio_values = np.clip(np.nan_to_num(audio_values), 0.0, 1.0)
        sample_rate = _sample_rate(times, self.settings.analysis_fps)
        quiet_samples = max(1, round(self.settings.serve_quiet_window * sample_rate))
        post_samples = max(1, round(self.settings.serve_post_window * sample_rate))
        impact_before = max(0, round(0.20 * sample_rate))
        impact_after = max(1, round(0.35 * sample_rate))

        # Distributed activity matters here: one person slowly collecting a ball
        # can create strong local motion, while a serve usually releases several
        # players into motion around different sides of the net.
        visual_activity = np.clip(0.60 * roi + 0.40 * spread, 0.0, 1.0)
        likelihood = np.zeros(times.size, dtype=np.float64)
        for index in range(times.size):
            pre_start = max(0, index - quiet_samples)
            pre = visual_activity[pre_start:index]
            pre_level = float(np.mean(pre)) if pre.size else 0.0

            post_end = min(times.size, index + post_samples + 1)
            post_visual = visual_activity[index:post_end]
            post_spread = spread[index:post_end]
            post_level = float(np.mean(post_visual)) if post_visual.size else 0.0
            spread_level = float(np.mean(post_spread)) if post_spread.size else 0.0

            audio_start = max(0, index - impact_before)
            audio_end = min(times.size, index + impact_after + 1)
            impact = float(np.max(audio_values[audio_start:audio_end]))

            quiet_setup = float(np.clip(1.0 - pre_level, 0.0, 1.0))
            motion_rise = float(np.clip(post_level - pre_level, 0.0, 1.0))

            # Repositioning players make a perfectly quiet setup uncommon, so
            # quietness is a confidence multiplier with a non-zero floor rather
            # than a hard prerequisite.  Visual rise/follow-through can reach a
            # full-strength cue on silent footage.  Audio only adds a modest
            # bonus and, by itself, can never resemble a serve.
            visual_cue = 0.55 * motion_rise + 0.45 * spread_level
            visual_estimate = (0.45 + 0.55 * quiet_setup) * visual_cue
            estimate = visual_estimate + 0.12 * impact * (1.0 - visual_estimate)
            # Observed setup + wrist motion + receivers reacting strengthens
            # the candidate. A waving arm alone cannot produce a serve event.
            setup = float(np.max(context_signals["readiness"][pre_start:index + 1]))
            reaction = float(np.mean(context_signals["player_motion"][index:post_end]))
            arm = float(np.max(context_signals["pose_serve"][index:post_end]))
            collection = float(np.mean(context_signals["retrieval"][index:post_end]))
            strength = self.settings.court_context_strength
            support = setup * (0.35 * motion_rise + 0.35 * reaction + 0.30 * arm)
            estimate += strength * support * (1 - estimate)
            estimate *= 1 - strength * collection * (1 - spread_level)
            likelihood[index] = np.clip(
                estimate * self.settings.serve_sensitivity, 0.0, 1.0
            )

        hold_samples = max(1, round(self.settings.serve_hold_duration * sample_rate))
        context = trailing_maximum(likelihood, hold_samples)
        return ServeLikelihoodSeries(times, likelihood, context)


def trailing_maximum(values: Sequence[float] | np.ndarray, window_size: int) -> np.ndarray:
    """Hold recent evidence for ``window_size`` samples without looking ahead."""

    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if isinstance(window_size, bool) or int(window_size) != window_size or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    result = np.zeros_like(signal)
    size = int(window_size)
    for index in range(signal.size):
        result[index] = np.max(signal[max(0, index - size + 1) : index + 1])
    return result


def _signal(
    name: str,
    values: Sequence[float] | np.ndarray,
    expected_size: int | None = None,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected_size is not None and result.size != expected_size:
        raise ValueError(f"{name} must have one item per timestamp")
    return result


def _sample_rate(timestamps: np.ndarray, fallback: float) -> float:
    if timestamps.size < 2:
        return fallback
    period = float(np.median(np.diff(timestamps)))
    return 1.0 / period if period > 0 else fallback


__all__ = ["ServeDetector", "ServeLikelihoodSeries", "trailing_maximum"]
