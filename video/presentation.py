"""Match summaries and local raster artwork for optional video presentation.

Point totals include excluded clips: selection is an editorial decision and
must not silently change the match score. Rejected detections never earn points.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


def field(item: object, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def prepare_export_options(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate presentation options without changing the supplied dictionary."""
    result: dict[str, Any] = {
        "aspect_ratio": "source", "highlights_only": False, "scoreboard": False,
        "team_a": "Team A", "team_b": "Team B", "initial_score_a": 0,
        "initial_score_b": 0, "overlay_path": None, "include_stats": False,
        "include_notes": False,
    }
    if options:
        unknown = set(options) - set(result)
        if unknown:
            raise ValueError(f"Unknown export option: {sorted(unknown)[0]}")
        result.update(options)
    if result["aspect_ratio"] not in {"source", "16:9", "9:16", "1:1"}:
        raise ValueError("aspect_ratio must be source, 16:9, 9:16, or 1:1")
    for name in ("highlights_only", "scoreboard", "include_stats", "include_notes"):
        if not isinstance(result[name], bool):
            raise ValueError(f"{name} must be a boolean")
    for name in ("initial_score_a", "initial_score_b"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 999:
            raise ValueError(f"{name} must be an integer between 0 and 999")
    for name in ("team_a", "team_b"):
        value = result[name]
        if not isinstance(value, str) or not value.strip() or len(value) > 100:
            raise ValueError(f"{name} must contain 1 to 100 characters")
        result[name] = " ".join(value.split())
    if result["overlay_path"]:
        overlay = Path(result["overlay_path"]).expanduser().resolve()
        if overlay.suffix.lower() != ".png" or not overlay.is_file():
            raise ValueError("Choose an existing PNG for the custom overlay")
        if cv2.imread(str(overlay), cv2.IMREAD_UNCHANGED) is None:
            raise ValueError("The custom overlay could not be decoded as a PNG")
        result["overlay_path"] = str(overlay)
    return result


def scores_before_rallies(rallies: Sequence[object], options: Mapping[str, Any]) -> dict[int, tuple[int, int]]:
    """Map original list indices to the score immediately before each point."""
    a, b = options["initial_score_a"], options["initial_score_b"]
    scores: dict[int, tuple[int, int]] = {}
    def start(item: object) -> float:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return float(item[0])
        return float(field(item, "start_time", field(item, "start", 0)))
    for index, rally in sorted(enumerate(rallies), key=lambda pair: (start(pair[1]), pair[0])):
        scores[index] = (a, b)
        if field(rally, "rejected", False):
            continue
        winner = field(rally, "winner", "")
        a += int(winner == "A")
        b += int(winner == "B")
    return scores


def match_statistics(rallies: Iterable[object], options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Summarize tagged points, including points omitted from the exported cut."""
    opts = prepare_export_options(options)
    valid = [item for item in rallies if not field(item, "rejected", False)]
    outcomes = Counter(str(field(item, "outcome", "")) for item in valid if field(item, "outcome", ""))
    players = Counter(str(field(item, "player", "")) for item in valid if field(item, "player", ""))
    wins_a = sum(field(item, "winner", "") == "A" for item in valid)
    wins_b = sum(field(item, "winner", "") == "B" for item in valid)
    return {
        "rally_count": len(valid), "tagged_points": wins_a + wins_b,
        "untagged_points": len(valid) - wins_a - wins_b,
        "score_a": opts["initial_score_a"] + wins_a,
        "score_b": opts["initial_score_b"] + wins_b,
        "team_a": opts["team_a"], "team_b": opts["team_b"],
        "highlights": sum(bool(field(item, "starred", False)) for item in valid),
        "outcomes": dict(outcomes), "player_tags": dict(players),
    }


def crop_dimensions(width: int, height: int, aspect_ratio: str) -> tuple[int, int]:
    """Largest even-sized crop of the requested shape inside the source."""
    if width < 2 or height < 2:
        raise ValueError("Video dimensions must be at least two pixels")
    if aspect_ratio == "source":
        return width - width % 2, height - height % 2
    ratios = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1)}
    if aspect_ratio not in ratios:
        raise ValueError("Unsupported crop aspect ratio")
    rw, rh = ratios[aspect_ratio]
    # Multiples of two preserve both the exact requested ratio and yuv420p.
    scale = int(min(width / (2 * rw), height / (2 * rh)))
    if scale < 1:
        raise ValueError("Video is too small for the requested crop")
    return 2 * rw * scale, 2 * rh * scale


def _center_expression(keyframes: Sequence[Mapping[str, Any]], axis: str, start: float) -> str:
    points: dict[float, float] = {}
    for frame in keyframes:
        time, value = float(frame["time"]), float(frame[axis])
        if not math.isfinite(time) or not math.isfinite(value) or time < 0 or not 0 <= value <= 1:
            raise ValueError("Crop keyframes require finite source times and normalized centers")
        points[time - start] = value
    ordered = sorted(points.items())
    if not ordered:
        return "0.5"
    expression = f"{ordered[-1][1]:.8f}"
    for (left, lv), (right, rv) in reversed(list(zip(ordered, ordered[1:]))):
        amount = f"clip((t-({left:.8f}))/({right-left:.8f}),0,1)"
        eased = f"({amount})*({amount})*(3-2*({amount}))"
        value = f"({lv:.8f}+({rv-lv:.8f})*({eased}))"
        expression = f"if(lt(t,{right:.8f}),{value},{expression})"
    return f"if(lt(t,{ordered[0][0]:.8f}),{ordered[0][1]:.8f},{expression})"


def crop_filter(width: int, height: int, aspect_ratio: str, keyframes: Sequence[Mapping[str, Any]], start: float) -> str:
    """An FFmpeg crop with smoothly eased, boundary-clamped camera movement."""
    cw, ch = crop_dimensions(width, height, aspect_ratio)
    x = _center_expression(keyframes, "x", start)
    y = _center_expression(keyframes, "y", start)
    return f"crop={cw}:{ch}:x='clip(({x})*iw-ow/2,0,iw-ow)':y='clip(({y})*ih-oh/2,0,ih-oh)',setsar=1"


def render_card(path: Path, width: int, height: int, lines: Sequence[str], *, transparent: bool = False) -> None:
    """Render text as pixels, keeping arbitrary user text out of FFmpeg syntax.

    The desktop application uses Qt's Unicode fonts. Command-line exports use
    OpenCV's built-in font, transliterating Unicode where possible.
    """
    width, height = max(2, width), max(2, height)
    from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter
    if QGuiApplication.instance() is not None:
        from PySide6.QtCore import Qt, QRect
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        image.fill(QColor(17, 25, 35, 226 if transparent else 255))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        row = height / max(1, len(lines))
        font = QFont("Arial")
        font.setPixelSize(max(9, int(min(row * 0.49, width * 0.061))))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(238, 247, 244))
        padding = max(4, int(width * 0.035))
        for index, text in enumerate(lines):
            elided = painter.fontMetrics().elidedText(str(text), Qt.TextElideMode.ElideRight, width - 2 * padding)
            painter.drawText(QRect(padding, int(index * row), width - 2 * padding, int(row)), Qt.AlignmentFlag.AlignVCenter, elided)
        painter.end()
        if not image.save(str(path), "PNG"):
            raise OSError(f"Cannot write export artwork: {path}")
        return
    image = np.full((height, width, 4), (35, 25, 17, 226 if transparent else 255), dtype=np.uint8)
    row = height / max(1, len(lines))
    padding = max(4, int(width * 0.035))
    for index, text in enumerate(lines):
        ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "replace").decode("ascii")
        scale = max(0.2, min(row / 50, width / 550))
        while cv2.getTextSize(ascii_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width - 2 * padding and len(ascii_text) > 4:
            ascii_text = ascii_text[:-4] + "..."
        cv2.putText(image, ascii_text, (padding, int((index + 0.66) * row)), cv2.FONT_HERSHEY_SIMPLEX, scale, (244, 247, 238, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write export artwork: {path}")


def statistics_lines(stats: Mapping[str, Any]) -> list[str]:
    lines = ["MATCH SUMMARY", f"{stats['team_a']}  {stats['score_a']}   -   {stats['score_b']}  {stats['team_b']}",
             f"{stats['tagged_points']} tagged points  |  {stats['highlights']} highlights"]
    if stats["untagged_points"]:
        lines.append(f"{stats['untagged_points']} untagged rallies - score may be incomplete")
    lines.extend(f"{name}: {count}" for name, count in sorted(stats["outcomes"].items())[:5])
    if stats["player_tags"]:
        lines.append("PLAYER TAGS")
        lines.extend(f"{name}: {count}" for name, count in sorted(stats["player_tags"].items())[:4])
    return lines
