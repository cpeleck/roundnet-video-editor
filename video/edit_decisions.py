"""Exchange exact original-video cuts as JSON or a CMX3600 edit decision list."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .exporter import generate_export_segments
from .presentation import field


def build_edit_decisions(source_path: str | os.PathLike[str], rallies: Iterable[object], *,
                         fps: float = 30, highlights_only: bool = False,
                         source_duration: float | None = None) -> dict[str, Any]:
    """Preserve source seconds and output offsets; no rendered-video alignment."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    items = list(rallies)
    segments = generate_export_segments(items, source_duration=source_duration, highlights_only=highlights_only)
    cursor = 0.0
    edits: list[dict[str, Any]] = []
    for segment in segments:
        rally = items[segment.source_index]
        edits.append({
            "rally_id": str(field(rally, "rally_id", "")),
            "source_start": segment.start, "source_end": segment.end,
            "output_start": cursor, "output_end": cursor + segment.duration,
            "starred": bool(field(rally, "starred", False)),
            "winner": str(field(rally, "winner", "")),
            "outcome": str(field(rally, "outcome", "")),
            "player": str(field(rally, "player", "")),
            "note": str(field(rally, "note", "")),
            "crop_keyframes": field(rally, "crop_keyframes", []),
        })
        cursor += segment.duration
    return {"schema_version": 1, "format": "roundnet_edit_decisions", "time_unit": "seconds",
            "source_path": str(Path(source_path).expanduser().resolve()), "source_fps": fps,
            "output_duration": cursor, "edits": edits}


def _frame_timecode(frames: int, fps: float) -> str:
    nominal = max(1, round(fps))
    second, frame = divmod(frames, nominal)
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}:{frame:02d}"


def _safe_line(value: object) -> str:
    return " ".join(str(value).replace("\x00", "").split())


def render_edl(decisions: dict[str, Any]) -> str:
    """CMX3600 video cuts, non-drop timecodes; JSON retains VFR precision.

    Imported EDL frame boundaries are rounded to the source's nominal frame
    rate. For variable frame rate footage prefer the exact-seconds JSON.
    """
    fps = float(decisions["source_fps"])
    filename = _safe_line(Path(decisions["source_path"]).name)
    lines = [f"TITLE: {filename}", "FCM: NON-DROP FRAME", "",
             f"* SOURCE FILE: {_safe_line(decisions['source_path'])}",
             f"* SOURCE FRAME RATE: {fps:.9f}",
             "* VIDEO EDIT LIST; LINK ORIGINAL AUDIO IN YOUR EDITOR.", ""]
    record_frame = 0
    for index, edit in enumerate(decisions["edits"], 1):
        first = round(edit["source_start"] * fps)
        last = max(first + 1, round(edit["source_end"] * fps))
        source_in = _frame_timecode(first, fps)
        source_out = _frame_timecode(last, fps)
        record_in = _frame_timecode(record_frame, fps)
        record_frame += last - first
        record_out = _frame_timecode(record_frame, fps)
        lines.extend([f"{index:03d}  AX       V     C        {source_in} {source_out} {record_in} {record_out}",
                      f"* FROM CLIP NAME: {filename}", f"* SOURCE FILE: {_safe_line(decisions['source_path'])}"])
        if edit.get("note"):
            lines.append(f"* COMMENT: {_safe_line(edit['note'])}")
        lines.append("")
    return "\n".join(lines) + "\n"


def save_edit_decisions(path: str | os.PathLike[str], source_path: str | os.PathLike[str],
                        rallies: Iterable[object], *, fps: float = 30,
                        highlights_only: bool = False, source_duration: float | None = None) -> Path:
    """Atomically save a .json or .edl without overwriting the source video."""
    destination = Path(path).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    if destination == source:
        raise ValueError("Edit decisions must be saved separately from the source video")
    if destination.suffix.lower() not in {".json", ".edl"}:
        raise ValueError("Choose a .json or .edl filename")
    decisions = build_edit_decisions(source, rallies, fps=fps, highlights_only=highlights_only,
                                     source_duration=source_duration)
    content = render_edl(decisions) if destination.suffix.lower() == ".edl" else json.dumps(decisions, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
