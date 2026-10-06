"""Portable project snapshots, atomic recovery saves, and bounded undo history."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .rally import Rally


PROJECT_VERSION = 1


def recovery_path(source_path: str, directory: Path) -> Path:
    identity = hashlib.sha256(str(Path(source_path).resolve()).encode()).hexdigest()[:24]
    return directory / f"{identity}.roundnet.json"


def write_project(path: str | Path, state: dict[str, Any]) -> Path:
    validate_project(state)
    destination = Path(path).expanduser().resolve()
    source = Path(state["video_path"]).expanduser().resolve()
    if source == destination:
        raise ValueError("A project cannot overwrite the source video")
    # Validate before replacing an earlier good recovery snapshot.
    payload = deepcopy(state)
    payload.update(schema_version=PROJECT_VERSION, kind="roundnet_project",
                   saved_utc=datetime.now(timezone.utc).isoformat())
    if source.is_file():
        stat = source.stat()
        payload["source_stat"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    payload["relative_video_path"] = os.path.relpath(source, destination.parent)
    data = json.dumps(payload, indent=2, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=".roundnet-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def read_project(path: str | Path) -> dict[str, Any]:
    location = Path(path).expanduser().resolve()
    payload = json.loads(location.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != "roundnet_project" or payload.get("schema_version") != PROJECT_VERSION:
        raise ValueError("Unsupported Roundnet project version")
    validate_project(payload)
    source = Path(payload["video_path"]).expanduser()
    if not source.is_file() and payload.get("relative_video_path"):
        alternative = (location.parent / payload["relative_video_path"]).resolve()
        if alternative.is_file():
            payload["video_path"] = str(alternative)
    return payload


def source_matches(state: dict[str, Any], path: str | Path) -> bool:
    """Guard recovery against a different recording replacing the same filename."""
    source = Path(path)
    if not source.is_file():
        return False
    expected = state.get("source_stat", {})
    stat = source.stat()
    return (expected.get("size", stat.st_size) == stat.st_size
            and expected.get("mtime_ns", stat.st_mtime_ns) == stat.st_mtime_ns)


def validate_project(state: dict[str, Any]) -> None:
    if not isinstance(state, dict) or not isinstance(state.get("video_path"), str) or not state["video_path"]:
        raise ValueError("Project needs a source video path")
    duration = float(state.get("video_duration", 0))
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Invalid project duration")
    identities = set()
    if not isinstance(state.get("rallies", []), list):
        raise ValueError("Project rallies must be a list")
    for entry in state.get("rallies", []):
        rally = Rally.from_dict(entry)
        if rally.rally_id in identities:
            raise ValueError("Duplicate rally identity in project")
        identities.add(rally.rally_id)
        if duration > 0 and rally.end_time > duration + .05:
            raise ValueError("A saved rally extends beyond the source video")
    for name in ("settings", "match_settings"):
        if not isinstance(state.get(name, {}), dict):
            raise ValueError(f"Invalid project {name}")
    if not isinstance(state.get("source_stat", {}), dict):
        raise ValueError("Invalid source fingerprint")
    if not isinstance(state.get("signals", {}) or {}, dict):
        raise ValueError("Invalid saved analysis")
    match = state.get("match_settings", {})
    from .point_stats import normalize_roster
    normalize_roster(match.get("players"))
    from .point_stats import PLAYER_IDS
    server = match.get("starting_server", "A1")
    receiver = match.get("starting_receiver", "B1")
    if server not in PLAYER_IDS or receiver not in PLAYER_IDS or server[0] == receiver[0]:
        raise ValueError("Starting server and receiver must be on opposite teams")
    if not isinstance(match.get("stats_complete", False), bool):
        raise ValueError("Invalid statistics completeness flag")
    if not isinstance(match.get("setup_complete", True), bool):
        raise ValueError("Invalid match setup flag")
    target_score = match.get("target_score", 21)
    if isinstance(target_score, bool) or not isinstance(target_score, int) or not 2 <= target_score <= 99:
        raise ValueError("Invalid target score")
    for key in ("initial_score_a", "initial_score_b"):
        value = match.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 999:
            raise ValueError("Invalid initial match score")
    for key in ("team_a", "team_b"):
        value = match.get(key, "Team")
        if not isinstance(value, str) or not value.strip() or len(value) > 100:
            raise ValueError("Team names must contain 1 to 100 characters")


class EditHistory:
    def __init__(self, limit: int = 100):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("History limit must be a positive integer")
        self.limit = limit
        self.undo_stack: list[dict[str, Any]] = []
        self.redo_stack: list[dict[str, Any]] = []

    def push(self, state: dict[str, Any]) -> None:
        self.undo_stack.append(deepcopy(state))
        self.undo_stack = self.undo_stack[-self.limit:]
        self.redo_stack.clear()

    def undo(self, current: dict[str, Any]) -> dict[str, Any] | None:
        if not self.undo_stack:
            return None
        self.redo_stack.append(deepcopy(current))
        return self.undo_stack.pop()

    def redo(self, current: dict[str, Any]) -> dict[str, Any] | None:
        if not self.redo_stack:
            return None
        self.undo_stack.append(deepcopy(current))
        return self.redo_stack.pop()
