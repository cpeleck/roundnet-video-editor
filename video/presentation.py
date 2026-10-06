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
        "players": None, "stats_complete": False, "stats_duration": 5.0,
    }
    if options:
        unknown = set(options) - set(result)
        if unknown:
            raise ValueError(f"Unknown export option: {sorted(unknown)[0]}")
        result.update(options)
    if result["aspect_ratio"] not in {"source", "16:9", "9:16", "1:1"}:
        raise ValueError("aspect_ratio must be source, 16:9, 9:16, or 1:1")
    for name in ("highlights_only", "scoreboard", "include_stats", "include_notes", "stats_complete"):
        if not isinstance(result[name], bool):
            raise ValueError(f"{name} must be a boolean")
    duration = result["stats_duration"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 1 <= duration <= 30:
        raise ValueError("stats_duration must be between 1 and 30 seconds")
    result["stats_duration"] = float(duration)
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
    from models.point_stats import normalize_roster
    if result["players"] is not None:
        result["players"] = normalize_roster(result["players"])
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
        if field(rally, "rejected", False) or field(rally, "outcome", "") == "Replay / no point":
            continue
        winner = field(rally, "winner", "")
        a += int(winner == "A")
        b += int(winner == "B")
    return scores


def match_statistics(rallies: Iterable[object], options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Summarize tagged points, including points omitted from the exported cut."""
    opts = prepare_export_options(options)
    rallies = list(rallies)
    valid = [item for item in rallies if not field(item, "rejected", False)
             and field(item, "outcome", "") != "Replay / no point"]
    outcomes = Counter(str(field(item, "outcome", "")) for item in valid if field(item, "outcome", ""))
    players = Counter(str(field(item, "player", "")) for item in valid if field(item, "player", ""))
    wins_a = sum(field(item, "winner", "") == "A" for item in valid)
    wins_b = sum(field(item, "winner", "") == "B" for item in valid)
    from models.statistics import calculate_statistics
    detailed = calculate_statistics(rallies, opts)
    return {
        "rally_count": len(valid), "tagged_points": wins_a + wins_b,
        "untagged_points": len(valid) - wins_a - wins_b,
        "score_a": opts["initial_score_a"] + wins_a,
        "score_b": opts["initial_score_b"] + wins_b,
        "team_a": opts["team_a"], "team_b": opts["team_b"],
        "highlights": sum(bool(field(item, "starred", False)) for item in valid),
        "outcomes": dict(outcomes), "player_tags": dict(players),
        "player_statistics": detailed,
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
    detailed = stats.get("player_statistics", {})
    if detailed.get("coverage", {}).get("complete", 0) or detailed.get("coverage", {}).get("partial", 0):
        lines.append("PLAYER STATS - tagged points only")
        for p in detailed["players"]:
            lines.append(f"{p['name']}: {p['aces']} ace | {p['put_aways']} put-away | {p['defensive_gets']} get | {p['errors']} err")
    elif stats["player_tags"]:
        lines.append("PLAYER TAGS")
        lines.extend(f"{name}: {count}" for name, count in sorted(stats["player_tags"].items())[:4])
    return lines


def render_player_end_card_image(width: int, height: int, summary: Mapping[str, Any]) -> np.ndarray:
    """Build the same four-player slate used by preview and video export."""
    if width < 2 or height < 2:
        raise ValueError("End-card dimensions must be positive")
    canvas = np.full((height, width, 3), (28, 24, 20), dtype=np.uint8)
    portrait = width / height < .8
    scale = min(width / (900 if portrait else 1600), height / (1600 if portrait else 900))
    def rect(x1, y1, x2, y2, color, fill=-1):
        cv2.rectangle(canvas, (round(x1 * width), round(y1 * height)),
                      (round(x2 * width), round(y2 * height)), color, fill)
    def label(value, x, y, size=1, color=(239, 244, 248), weight=1, max_width=None):
        string = unicodedata.normalize("NFKD", str(value)).encode("ascii", "replace").decode("ascii")
        font_size = max(.18, size * scale)
        if max_width:
            while string and cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, font_size, weight)[0][0] > max_width * width:
                string = string[:-1]
            if string != str(value) and len(string) > 3:
                string = string[:-3] + "..."
        cv2.putText(canvas, string, (round(x * width), round(y * height)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, color, max(1, round(weight * scale)), cv2.LINE_AA)
    detail = summary["player_statistics"]
    players = detail["players"]
    rect(0, 0, 1, .18, (61, 42, 32))
    label("FINAL SCORE", .035, .052, .9, (130, 190, 255), 2)
    label(summary["team_a"], .035, .112, 1.45, max_width=.29)
    label(f"{summary['score_a']}  -  {summary['score_b']}", .395, .112, 1.75, (255, 255, 255), 3)
    label(summary["team_b"], .70, .112, 1.45, max_width=.27)
    left, top, right, bottom = .025, .205, .975, .94
    label("PLAYER STATISTICS", left, .235, .8, (130, 190, 255), 2)
    col_left = .31
    cell_width = (right - col_left) / 4
    labels = ["Serve %", "Aces : Aced", "Put-away %", "Defensive Gets",
              "Strong : Weak Sets", "Errors", "+/-  Ace : Rim", "Breaks : Broken",
              "Hitting", "Serving", "Defense", "Efficiency", "RPR"]
    def pct(value, numerator, denominator):
        return "--" if value is None else f"{value*100:.0f}% ({numerator}/{denominator})"
    def values(p):
        known_hits = p["put_aways"] + p["hits_returned"] + p["hit_errors"]
        rpr = p["rpr"]
        number = lambda key: "--" if rpr[key] is None else f"{rpr[key]:.1f}"
        return [pct(p["serve_pct"], p["serves_in"], p["serve_attempts"]),
                f"{p['aces']} : {p['aced']}", pct(p["put_away_pct"], p["put_aways"], known_hits),
                str(p["defensive_gets"]), f"{p['strong_sets']} : {p['weak_sets']}", str(p["errors"]),
                f"{p['aces']} : {p['rims']}", f"{p['breaks']} : {p['broken']}",
                number("hitting"), number("serving"), number("defense"), number("efficiency"), number("overall")]
    if portrait:
        player_values = [values(p) for p in players]
        for team_index, team in enumerate(("A", "B")):
            first = team_index * 2
            panel_top = .195 + team_index * .37
            row_height = .37 / (len(labels) + 1)
            rect(left, panel_top, right, panel_top + row_height,
                 (95, 70, 48) if team == "A" else (77, 63, 80))
            label("STATISTIC", left + .014, panel_top + row_height * .73, .69)
            for col in range(2):
                x = .45 + col * .26
                label(players[first + col]["name"], x, panel_top + row_height * .73,
                      .78, max_width=.24)
            for row, title in enumerate(labels):
                y = panel_top + (row + 1) * row_height
                rect(left, y, right, y + row_height,
                     (52, 48, 45) if row % 2 == 0 else (63, 57, 53))
                if row in (8, 12):
                    rect(left, y, right, y + .002, (135, 128, 117))
                label(title, left + .014, y + row_height * .73, .68,
                      (147, 206, 255) if row >= 8 else (235, 241, 246), max_width=.40)
                for col in range(2):
                    label(player_values[first + col][row], .45 + col * .26,
                          y + row_height * .73, .68, max_width=.24)
            rect(.445, panel_top, .448, panel_top + .37, (116, 111, 105))
            rect(.705, panel_top, .708, panel_top + .37, (116, 111, 105))
        coverage = detail["coverage"]
        label(f"{coverage['complete']}/{coverage['points']} points fully logged", left, .97,
              .66, (175, 189, 199), max_width=.95)
        return canvas
    table_top = .265
    row_height = (bottom - table_top) / (len(labels) + 1)
    rect(left, table_top, right, table_top + row_height, (73, 62, 54))
    for i, player in enumerate(players):
        x = col_left + i * cell_width
        rect(x, table_top, x + cell_width, table_top + row_height,
             (95, 70, 48) if player["team"] == "A" else (77, 63, 80))
        label(player["name"], x + .011, table_top + row_height * .7, .86, max_width=cell_width - .018)
    player_values = [values(p) for p in players]
    for row, title in enumerate(labels):
        y = table_top + (row + 1) * row_height
        rect(left, y, right, y + row_height, (52, 48, 45) if row % 2 == 0 else (63, 57, 53))
        if row in (8, 12):
            rect(left, y, right, y + .003, (135, 128, 117))
        label(title, left + .012, y + row_height * .7, .73 if row != 12 else .85,
              (147, 206, 255) if row >= 8 else (235, 241, 246), 2 if row == 12 else 1,
              max_width=col_left - left - .02)
        for col, p in enumerate(players):
            x = col_left + col * cell_width
            label(player_values[col][row], x + .011, y + row_height * .7,
                  .72 if row != 12 else .88, (255, 255, 255), 2 if row == 12 else 1,
                  max_width=cell_width - .018)
    for col in range(5):
        rect(col_left + col * cell_width - .0015, table_top, col_left + col * cell_width + .0015,
             bottom, (116, 111, 105))
    coverage = detail["coverage"]
    footer = f"{coverage['complete']}/{coverage['points']} points fully logged"
    if not detail["rpr_eligible"]:
        footer += "  |  RPR appears after full-match confirmation and complete touch logs"
    label(footer, left, .975, .58, (175, 189, 199), max_width=.95)
    return canvas


def render_player_end_card(path: Path, width: int, height: int, summary: Mapping[str, Any]) -> None:
    if not cv2.imwrite(str(path), render_player_end_card_image(width, height, summary)):
        raise OSError(f"Cannot write player end card: {path}")
