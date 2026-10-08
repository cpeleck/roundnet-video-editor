"""Canonical project commands, durable recovery, and derived match views.

This module has no Qt or web dependency. Scoring and statistics are always
derived by the existing domain models, including after undo and source relinks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

from config import DetectionSettings
from models.match_flow import CLASSIFICATIONS, match_timeline, normalize_classification
from models.point_stats import normalize_point_stats, normalize_roster
from models.project import EditHistory, read_project, source_matches, validate_project, write_project
from models.rally import Rally
from models.statistics import calculate_statistics
from video.metadata import probe_video_metadata


STAGES = ("setup", "find", "review", "statistics", "export")
DEFAULT_MATCH = dict(team_a="Team A", team_b="Team B", initial_score_a=0,
                     initial_score_b=0, starting_server="A1", starting_receiver="B1",
                     target_score=21, setup_complete=False, stats_complete=False)
SIGNAL_ALIASES = {key: f"{key}_scores" for key in (
    "motion", "roi_motion", "audio", "temporal", "motion_spread", "serve",
    "serve_context", "player_count", "readiness", "player_motion", "retrieval", "pose_serve")}
SIGNAL_ALIASES["rally_score"] = "rally_scores"
LEGACY_UI_SETTINGS = {"debug_mode", "prefer_hardware", "save_labels_after_export"}


def normalize_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Detector settings must be an object")
    preferences = {key: value for key, value in settings.items() if key in LEGACY_UI_SETTINGS}
    for key, value in preferences.items():
        _boolean(value, key)
    core = DetectionSettings.from_dict({key: value for key, value in settings.items() if key not in LEGACY_UI_SETTINGS})
    return {**core.to_dict(), **preferences}


def normalize_signals(signals):
    if signals is None:
        return None
    if not isinstance(signals, dict):
        raise ValueError("Saved detector signals must be an object")
    # Version-one Qt projects store DetectionResult field names. The detector's
    # CLI representation uses shorter keys, which remain accepted on input.
    return {SIGNAL_ALIASES.get(key, key): deepcopy(value) for key, value in signals.items()}


class ProjectError(ValueError):
    """An actionable application error that a frontend can display."""

    def __init__(self, message: str, *, code="invalid_command", status=422, **details):
        super().__init__(message)
        self.code, self.status, self.details = code, status, details

    def to_dict(self):
        return {"code": self.code, "message": str(self), **self.details}


@dataclass
class Project:
    state: dict[str, Any]
    history: EditHistory = field(default_factory=EditHistory)
    lock: Any = field(default_factory=RLock)
    busy_job_id: str | None = None


def _number(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _text(value, name, limit=100):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} needs 1 to {limit} characters")
    return value.strip()


def reconcile_classifications(rallies, settings):
    """Populate legacy presentation fields using the canonical match timeline."""
    from models.quick_stats import reconcile_quick_stats
    names = {p["player_id"]: p["name"] for p in normalize_roster(settings.get("players"))}
    for row in match_timeline(rallies, settings):
        rally = rallies[row["index"]]
        if not rally.classification:
            continue
        kind = rally.classification["kind"]
        actor = (rally.classification.get("player_id") if kind == "error" else
                 row["server_id"] if kind in ("ace", "double_fault", "service_break") else
                 row["receiver_id"] if kind == "sideout" else "")
        if row["provisional"] and kind != "error":
            actor = ""
        rallies[row["index"]] = replace(
            rally, winner=row["winner"],
            point_stats=reconcile_quick_stats(rally.point_stats, rally.classification, row,
                                              start=rally.start_time, end=rally.end_time),
            outcome="Replay / no point" if kind == "redo" else CLASSIFICATIONS[kind]["label"],
            player=names.get(actor, ""))
    return rallies


class ProjectService:
    def __init__(self, data_dir: str | Path | None = None, *, metadata_probe: Callable = probe_video_metadata):
        self.data_dir = Path(data_dir or Path(__file__).resolve().parents[1] / ".roundnet" / "web").resolve()
        self.directory = self.data_dir
        self.project_dir = self.data_dir / "projects"
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_probe = metadata_probe
        self._projects: dict[str, Project] = {}
        self._lock = RLock()
        self.recovery_errors: list[dict] = []
        for path in self.project_dir.glob("*.roundnet.json"):
            try:
                state = self._normalize(read_project(path))
                if path.name != f"{state['project_id']}.roundnet.json":
                    raise ValueError("Recovery project identity does not match its filename")
                self._projects[state["project_id"]] = Project(state)
            except (ValueError, OSError, TypeError, KeyError) as exc:
                self.recovery_errors.append({"path": str(path), "message": str(exc)})

    def _normalize(self, source):
        state = deepcopy(source)
        project_id = state.get("project_id") or uuid4().hex
        if not isinstance(project_id, str) or len(project_id) != 32 or any(c not in "0123456789abcdef" for c in project_id):
            raise ValueError("Invalid project identity")
        state["project_id"] = project_id
        state.setdefault("title", Path(state["video_path"]).stem)
        match = {**DEFAULT_MATCH, **state.get("match_settings", {})}
        match["setup_complete"] = state.get("match_settings", {}).get("setup_complete", True)
        match["players"] = normalize_roster(match.get("players"))
        state["match_settings"] = match
        state.setdefault("stage", "review" if match["setup_complete"] else "setup")
        state.setdefault("mode", "automatic")
        state.setdefault("selected", 0 if state.get("rallies") else -1)
        state.setdefault("position", 0)
        state.setdefault("revision", 0)
        state.setdefault("explanation", "")
        state.setdefault("highlight_queue", [])
        state.setdefault("export_settings", {})
        state.setdefault("complete", False)
        state.setdefault("complete_review_confirmed", state["complete"])
        state.setdefault("rejected_detections", [])
        state.setdefault("initial_predictions", [])
        state.setdefault("project_path", None)
        state.setdefault("settings", DetectionSettings().to_dict())
        state["settings"] = normalize_settings(state["settings"])
        if "prefer_hardware" in state["settings"]:
            state["export_settings"].setdefault("prefer_hardware", state["settings"]["prefer_hardware"])
        state.setdefault("save_status", "saved")
        state["signals"] = normalize_signals(state.get("signals"))
        self._validate(state)
        original_rallies = state["rallies"]
        state["rallies"] = [r.to_dict() for r in reconcile_classifications(
            [Rally.from_dict(r) for r in original_rallies], match)]
        if any(before.get("point_stats") != after.get("point_stats")
               for before, after in zip(original_rallies, state["rallies"])):
            match["stats_complete"] = False
        return state

    def _probe(self, source):
        if not isinstance(source, (str, Path)):
            raise ValueError("Video path must be a file path")
        path = Path(_text(str(source), "Video path", 4096)).expanduser().resolve()
        if not path.is_file():
            raise ProjectError("Locate the original recording to continue.", code="missing_source", status=409, video_path=str(path))
        metadata = self.metadata_probe(str(path))
        values = asdict(metadata) if is_dataclass(metadata) else dict(metadata)
        values = {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}
        duration = values.get("duration_seconds", values.get("duration", 0))
        duration = _number(duration, "Video duration", minimum=.001)
        return str(path), duration, values

    def create(self, video_path: str, *, title=None):
        path, duration, metadata = self._probe(video_path)
        state = self._normalize(dict(video_path=path, video_duration=duration,
                                    video_metadata=metadata, title=title or Path(path).stem,
                                    rallies=[], match_settings=deepcopy(DEFAULT_MATCH)))
        state = self._persist(state)
        with self._lock:
            self._projects[state["project_id"]] = Project(state)
        return self.snapshot(state["project_id"])

    def open(self, path: str, *, video_path=None, allow_source_mismatch=False):
        _boolean(allow_source_mismatch, "Use changed source")
        location = Path(path).expanduser().resolve()
        state = read_project(location)
        source = video_path or state["video_path"]
        if not Path(source).expanduser().is_file():
            raise ProjectError("The recording is missing. Locate the original video to reopen this project.",
                               code="missing_source", status=409, video_path=str(source), project_path=str(location))
        matches = source_matches(state, Path(source).expanduser())
        if not matches and not allow_source_mismatch:
            raise ProjectError("This recording has changed. Confirm using its saved cuts; outcomes and touch details will be cleared.",
                               code="source_mismatch", status=409, video_path=str(source), project_path=str(location))
        source, duration, metadata = self._probe(source)
        state.update(video_path=source, video_duration=duration, video_metadata=metadata)
        if not matches:
            self._clear_source_analysis(state)
        state = self._normalize(state)
        project_id = state["project_id"]
        state["project_path"] = str(location) if location.parent != self.project_dir else state.get("project_path")
        with self._lock:
            old = self._projects.get(project_id)
        if old is None:
            persisted = self._persist(state)
            candidate = Project(persisted)
            with self._lock:
                old = self._projects.setdefault(project_id, candidate)
            if old is candidate:
                return self.snapshot(project_id)
        with old.lock:
            if old.busy_job_id:
                raise ProjectError("Wait for the running job or cancel it before reopening.", code="project_busy", status=409)
            state["revision"] = max(old.state["revision"], state["revision"]) + 1
            old.state = self._persist(state)
            old.history = EditHistory()
        return self.snapshot(project_id)

    @staticmethod
    def _clear_source_analysis(state):
        for key in ("signals", "analysis_settings", "analysis_roi", "analysis_court_context"):
            state[key] = None
        state.update(initial_predictions=[], rejected_detections=[], complete=False, complete_review_confirmed=False, highlight_queue=[])
        state.setdefault("match_settings", {})["stats_complete"] = False
        state["rallies"] = [{**r, "reviewed": False, "point_stats": {}, "classification": {},
                              "winner": "", "outcome": "", "player": ""}
                             for r in state.get("rallies", [])]
        state.pop("source_stat", None)

    def get(self, project_id):
        with self._lock:
            project = self._projects.get(project_id)
        if project is None:
            raise ProjectError("Project not found", code="project_not_found", status=404)
        return project

    def get_state(self, project_id):
        project = self.get(project_id)
        with project.lock:
            return deepcopy(project.state)

    def revision(self, project_id):
        return self.get_state(project_id)["revision"]

    def require_source(self, project_id):
        state = self.get_state(project_id)
        source = Path(state["video_path"])
        if not source.is_file():
            raise ProjectError("Locate the original recording to continue.", code="missing_source", status=409, video_path=str(source))
        if not source_matches(state, source):
            raise ProjectError("The source recording changed. Relink it before editing or processing.", code="source_mismatch", status=409, video_path=str(source))
        return source

    def list(self):
        with self._lock:
            identities = list(self._projects)
        summaries = []
        for identity in identities:
            state = self.get_state(identity)
            source = Path(state["video_path"])
            summaries.append({key: state.get(key) for key in (
                "project_id", "title", "stage", "saved_utc", "project_path", "video_path", "source_stat", "revision")}
                | {"clip_count": len(state["rallies"]), "rally_count": len(state["rallies"]),
                   "thumbnail_url": f"/api/projects/{identity}/thumbnail?time=0",
                   "source_missing": not source.is_file(), "source_changed": source.is_file() and not source_matches(state, source),
                   "source_status": "missing" if not source.is_file() else "ready" if source_matches(state, source) else "changed"})
        return sorted(summaries, key=lambda state: state.get("saved_utc") or "", reverse=True)

    def snapshot(self, project_id):
        project = self.get(project_id)
        with project.lock:
            state = deepcopy(project.state)
            rallies = [Rally.from_dict(r) for r in state["rallies"]]
            timeline = match_timeline(rallies, state["match_settings"])
            missing = next((row["rally_id"] for row in timeline if not row["winner"] and row["kind"] != "redo"
                            and not rallies[row["index"]].rejected
                            and rallies[row["index"]].outcome != "Replay / no point"), None)
            source = Path(state["video_path"])
            state.update(timeline=timeline, statistics=calculate_statistics(rallies, state["match_settings"]),
                         can_undo=bool(project.history.undo_stack), can_redo=bool(project.history.redo_stack),
                         busy_job_id=project.busy_job_id, first_missing_outcome_id=missing,
                         source_missing=not source.is_file(), source_changed=source.is_file() and not source_matches(state, source),
                         source_status="missing" if not source.is_file() else
                         "ready" if source_matches(state, state["video_path"]) else "changed")
            return state

    def _validate(self, state):
        validate_project(state)
        _text(state["title"], "Match title")
        if state["stage"] not in STAGES:
            raise ValueError("Unknown workflow stage")
        if state["mode"] not in ("manual", "automatic"):
            raise ValueError("Choose automatic detection or manual clipping")
        if type(state["revision"]) is not int or state["revision"] < 0:
            raise ValueError("Invalid project revision")
        _number(state["position"], "Playback position")
        _boolean(state["complete"], "Full-recording review confirmation")
        _boolean(state["complete_review_confirmed"], "Full-recording review confirmation")
        normalize_settings(state["settings"])
        self._validate_queue(state)

    @staticmethod
    def _validate_queue(state):
        queue = state["highlight_queue"]
        if not isinstance(queue, list) or len(queue) > 1000:
            raise ValueError("Highlights need a list of at most 1000 clips")
        rallies = {r["rally_id"]: r for r in state["rallies"]}
        seen = set()
        for clip in queue:
            if not isinstance(clip, dict) or clip.get("rally_id") not in rallies:
                raise ValueError("A highlight must refer to a saved clip")
            identity = clip["rally_id"]
            if identity in seen:
                raise ValueError("Choose each highlight once")
            seen.add(identity)
            source = Rally.from_dict(rallies[identity])
            if source.rejected:
                raise ValueError("Restore a rejected clip before adding it to highlights")
            preview = replace(source, start_time=clip.get("start_time", source.start_time),
                              end_time=clip.get("end_time", source.end_time),
                              crop_keyframes=clip.get("crop_keyframes", source.crop_keyframes))
            _number(clip.get("start_time", source.start_time), "Highlight start")
            _number(clip.get("end_time", source.end_time), "Highlight end")
            if preview.end_time > state["video_duration"] + .05:
                raise ValueError("Highlight trim extends beyond the recording")

    def _persist(self, state):
        self._validate(state)
        state = deepcopy(state)
        state.update(save_status="saved", saved_utc=datetime.now(timezone.utc).isoformat())
        state.pop("save_error", None)
        source = Path(state["video_path"])
        if source.is_file():
            stat = source.stat()
            state["source_stat"] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        destination = self.project_dir / f"{state['project_id']}.roundnet.json"
        write_project(destination, state)
        if state.get("project_path") and Path(state["project_path"]).resolve() != destination:
            try:
                write_project(state["project_path"], state)
            except (OSError, ValueError) as exc:
                state.update(save_status="recovery_saved", save_error=f"Recovery saved; project file could not be updated: {exc}")
        return state

    @staticmethod
    def _check_revision(project, revision):
        if revision is not None and (type(revision) is not int or revision != project.state["revision"]):
            raise ProjectError("The project changed. Reload the latest edits and try again.",
                               code="revision_conflict", status=409, revision=project.state["revision"])

    def command(self, project_id, command):
        project = self.get(project_id)
        with project.lock:
            self._check_revision(project, command.get("revision"))
            if project.busy_job_id:
                raise ProjectError("Wait for the running job or cancel it before editing.", code="project_busy", status=409,
                                   job_id=project.busy_job_id)
            kind = command.get("type")
            if kind != "relink":
                self.require_source(project_id)
            if kind in ("undo", "redo"):
                stack = project.history.undo_stack if kind == "undo" else project.history.redo_stack
                if not stack:
                    return self.snapshot(project_id)
                state = deepcopy(stack[-1])
                state.update(revision=project.state["revision"] + 1,
                             project_path=project.state.get("project_path"),
                             explanation="Score, serving assignments, and saved edits restored.")
                if not source_matches(state, state["video_path"]):
                    raise ProjectError("The recording for this edit is unavailable or changed. Relink the original source before restoring it.",
                                       code="source_mismatch", status=409)
                persisted = self._persist(state)
                getattr(project.history, kind)(project.state)
                project.state = persisted
            else:
                state = deepcopy(project.state)
                affects_stats, history = self._apply(state, command)
                if affects_stats:
                    state["match_settings"]["stats_complete"] = False
                    state["complete"] = False
                    state["complete_review_confirmed"] = False
                state["rallies"] = [r.to_dict() for r in reconcile_classifications(
                    [Rally.from_dict(r) for r in state["rallies"]], state["match_settings"])]
                if state == project.state and kind != "save_as":
                    return self.snapshot(project_id)
                state["revision"] = project.state["revision"] + 1
                persisted = self._persist(state)
                if kind == "relink" and not source_matches(project.state, state["video_path"]):
                    project.history = EditHistory()
                elif history:
                    project.history.push(project.state)
                project.state = persisted
            return self.snapshot(project_id)

    def _apply(self, state, command):
        kind = command.get("type")
        affects_stats, history = False, True
        rallies = [Rally.from_dict(r) for r in state["rallies"]]
        identity = command.get("rally_id")
        index = next((i for i, rally in enumerate(rallies) if rally.rally_id == identity), -1)
        clip_kinds = {"classify", "clear_classification", "trim", "split", "star", "reject",
                      "export_enabled", "caption", "crop", "touches", "select"}
        if kind in clip_kinds and index < 0:
            raise ValueError("Select a saved clip")
        rally = rallies[index] if index >= 0 else None
        if kind in {"classify", "clear_classification", "touches"} and rally.rejected:
            raise ValueError("Restore this clip before classifying it")
        if kind not in {"setup", "stage", "select", "position", "relink", "save_as", "settings", "court", "rename"} and not state["match_settings"]["setup_complete"]:
            raise ValueError("Complete match setup before editing clips")
        if kind == "setup":
            old = state["match_settings"]
            match = {**old, **command["match_settings"], "setup_complete": True}
            match["players"] = normalize_roster(match.get("players"))
            state["match_settings"] = match
            state["mode"] = command.get("mode", state["mode"])
            state["stage"] = "find" if state["mode"] == "automatic" else "review"
            if "title" in command:
                state["title"] = _text(command["title"], "Match title")
            affects_stats = any(old.get(k) != match.get(k) for k in (
                "initial_score_a", "initial_score_b", "starting_server", "starting_receiver", "target_score"))
        elif kind == "stage":
            state["stage"] = command["stage"]
            history = False
        elif kind == "rename":
            state["title"] = _text(command["title"], "Match title")
        elif kind == "select":
            state["selected"] = index
            state["position"] = command.get("position", rally.start_time)
            history = False
        elif kind == "position":
            state["position"] = min(_number(command["position"], "Playback position"), state["video_duration"])
            history = False
        elif kind == "classify":
            assignment = next(row for row in match_timeline(rallies, state["match_settings"]) if row["index"] == index)
            if assignment["provisional"]:
                missing = next(row["rally_id"] for row in match_timeline(rallies, state["match_settings"])
                               if not row["winner"] and row["kind"] != "redo" and not rallies[row["index"]].rejected
                               and rallies[row["index"]].outcome != "Replay / no point")
                raise ProjectError("Classify the earlier missing outcome first so the server and receiver are correct.",
                                   code="missing_outcome", status=409, rally_id=missing)
            classification = {"version": 1, "kind": command["kind"]}
            if command["kind"] == "error":
                classification["player_id"] = command.get("player_id")
            classification = normalize_classification(classification)
            point = {}
            if "quick_log" in command:
                from models.quick_stats import quick_point_stats
                from models.statistics import point_issues, classified_point_issues
                point = quick_point_stats(command["quick_log"], classification, assignment)
                candidate = replace(rally, classification=classification, reviewed=True, point_stats=point)
                candidate_rallies = rallies[:]
                candidate_rallies[index] = candidate
                final_assignment = match_timeline(candidate_rallies, state["match_settings"])[index]
                issues = point_issues(point, final_assignment["winner"], start=rally.start_time, end=rally.end_time)
                issues += classified_point_issues(point, classification, final_assignment)
                if issues:
                    raise ValueError(" ".join(dict.fromkeys(issues)))
                if classification == rally.classification and point.get("quick_log") == rally.point_stats.get("quick_log"):
                    return False, False
            if classification == rally.classification and "quick_log" not in command:
                return False, False
            # Reconfirming an existing full log must not replace its observations
            # with defaults. Explicit corrections remain available via touches.
            if classification == rally.classification and rally.point_stats and not rally.point_stats.get("quick_log"):
                point = rally.point_stats
            rallies[index] = replace(rally, classification=classification, reviewed=True, point_stats=point)
            reconcile_classifications(rallies, state["match_settings"])
            state["selected"] = next((i for i in range(index + 1, len(rallies)) if not rallies[i].rejected
                                      and not rallies[i].classification and not rallies[i].winner
                                      and rallies[i].outcome != "Replay / no point"), index)
            state["position"] = rally.end_time
            state["explanation"] = self._explanation(rallies, state["match_settings"], index)
            if rally.point_stats and not point:
                state["explanation"] += " Earlier touch details cleared; Undo restores them."
            affects_stats = True
        elif kind == "clear_classification":
            rallies[index] = replace(rally, classification={}, winner="", outcome="", player="", point_stats={}, reviewed=False)
            state["explanation"] = "Outcome cleared. Later scores and serving assignments are provisional until it is classified."
            affects_stats = True
        elif kind == "add":
            new = Rally(_number(command["start_time"], "Clip start"), _number(command["end_time"], "Clip end"), reviewed=True)
            rallies.append(new)
            rallies.sort(key=lambda r: r.start_time)
            state["selected"] = rallies.index(new)
            state["position"] = new.end_time
            affects_stats = True
        elif kind == "trim":
            rallies[index] = rally.with_bounds(_number(command["start_time"], "Clip start"), _number(command["end_time"], "Clip end"))
            rallies.sort(key=lambda r: r.start_time)
            state["selected"] = next(i for i, r in enumerate(rallies) if r.rally_id == rally.rally_id)
            affects_stats = rally.start_time != rallies[state["selected"]].start_time or rally.end_time != rallies[state["selected"]].end_time
        elif kind == "split":
            time = _number(command["time"], "Split time")
            if not rally.start_time < time < rally.end_time:
                raise ValueError("Place the split inside the clip")
            clear = dict(classification={}, winner="", outcome="", player="", point_stats={}, reviewed=False)
            rallies[index:index + 1] = [replace(rally, end_time=time, **clear),
                                       replace(rally, start_time=time, rally_id=uuid4().hex, serve_confidence=0, **clear)]
            state["highlight_queue"] = [c for c in state["highlight_queue"] if c["rally_id"] != rally.rally_id]
            affects_stats = True
        elif kind == "merge":
            ids = command["rally_ids"]
            if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2:
                raise ValueError("Choose two neighboring clips to merge")
            positions = [i for i, r in enumerate(rallies) if r.rally_id in ids]
            if len(positions) != 2 or positions[1] != positions[0] + 1:
                raise ValueError("Choose two neighboring clips to merge")
            left, right = (rallies[i] for i in positions)
            merged = replace(left, start_time=min(left.start_time, right.start_time), end_time=max(left.end_time, right.end_time),
                             classification={}, winner="", outcome="", player="", point_stats={}, reviewed=False,
                             rejected=False, enabled=True, starred=left.starred or right.starred,
                             confidence=max(left.confidence, right.confidence), serve_confidence=max(left.serve_confidence, right.serve_confidence))
            rallies[positions[0]:positions[1] + 1] = [merged]
            state["selected"] = positions[0]
            state["highlight_queue"] = [c for c in state["highlight_queue"] if c["rally_id"] not in ids]
            affects_stats = True
        elif kind == "star":
            rallies[index] = replace(rally, starred=_boolean(command.get("starred", not rally.starred), "Starred"))
        elif kind == "reject":
            rejected = _boolean(command.get("rejected", not rally.rejected), "Not a rally")
            rallies[index] = replace(rally, rejected=rejected, enabled=not rejected, reviewed=True)
            if rejected:
                state["highlight_queue"] = [c for c in state["highlight_queue"] if c["rally_id"] != rally.rally_id]
            affects_stats = rejected != rally.rejected
        elif kind == "export_enabled":
            enabled = _boolean(command["enabled"], "Include in exported video")
            if enabled and rally.rejected:
                raise ValueError("Restore a rejected clip before including it in export")
            rallies[index] = replace(rally, enabled=enabled)
        elif kind == "caption":
            note = command["note"]
            if not isinstance(note, str) or len(note) > 2000:
                raise ValueError("Caption must be text with at most 2000 characters")
            rallies[index] = replace(rally, note=note)
        elif kind == "crop":
            rallies[index] = replace(rally, crop_keyframes=command["crop_keyframes"])
        elif kind == "touches":
            point = normalize_point_stats(command["point_stats"])
            rallies[index] = replace(rally, point_stats=point)
            affects_stats = point != rally.point_stats
        elif kind == "full_match":
            state["match_settings"]["stats_complete"] = _boolean(command["confirmed"], "Full-match confirmation")
        elif kind == "complete_review":
            state["complete"] = state["complete_review_confirmed"] = _boolean(command["confirmed"], "Full-recording review confirmation")
        elif kind == "court":
            from detection.court_context import normalize_court_context
            state["court_context"] = normalize_court_context(command.get("court_context"))
            roi = command.get("roi")
            if roi is not None:
                if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                    raise ValueError("Playing area needs x, y, width, and height")
                roi = [_number(v, "Playing area coordinate") for v in roi]
                x, y, w, h = roi
                if w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
                    raise ValueError("Playing area must fit inside the video")
            state["roi"] = roi
        elif kind == "settings":
            state["settings"] = normalize_settings({**state["settings"], **command["settings"]})
        elif kind == "export_settings":
            if not isinstance(command["settings"], dict):
                raise ValueError("Export settings must be an object")
            presentation_keys = {"aspect_ratio", "highlights_only", "scoreboard", "overlay_path", "include_stats", "include_notes", "stats_duration"}
            preferences = {"mode", "output_path", "prefer_hardware"}
            unknown = set(command["settings"]) - presentation_keys - preferences
            if unknown:
                raise ValueError(f"Unknown export setting: {sorted(unknown)[0]}")
            state["export_settings"].update(command["settings"])
            from video.presentation import prepare_export_options
            prepare_export_options({key: value for key, value in state["export_settings"].items() if key in presentation_keys})
            if state["export_settings"].get("mode", "full") not in {"full", "highlights"}:
                raise ValueError("Choose full match or highlights")
            if not isinstance(state["export_settings"].get("output_path", ""), str):
                raise ValueError("Export destination must be a path")
            _boolean(state["export_settings"].get("prefer_hardware", True), "Automatic hardware encoding")
        elif kind == "highlight_queue":
            state["highlight_queue"] = deepcopy(command["clips"])
        elif kind == "save_as":
            path = Path(command["path"]).expanduser().resolve()
            if path.suffix.lower() != ".json":
                raise ValueError("Save a .roundnet.json project file")
            if path == Path(state["video_path"]).resolve():
                raise ValueError("A project cannot overwrite the source recording")
            state["project_path"] = str(path)
            history = False
        elif kind == "relink":
            accepted_mismatch = _boolean(command.get("allow_source_mismatch", False), "Use changed source")
            path, duration, metadata = self._probe(command["video_path"])
            if not source_matches(state, path):
                if not accepted_mismatch:
                    raise ProjectError("This recording has changed. Confirm using its cuts and clearing previous outcomes and touch details.", code="source_mismatch", status=409)
                self._clear_source_analysis(state)
                rallies = [Rally.from_dict(r) for r in state["rallies"]]
            state.update(video_path=path, video_duration=duration, video_metadata=metadata)
            state["explanation"] = "Source recording located. Review saved clip boundaries before continuing."
            affects_stats = True
        else:
            raise ValueError("Unknown project command")
        state["rallies"] = [r.to_dict() for r in rallies]
        return affects_stats, history

    @staticmethod
    def _explanation(rallies, match, index):
        row = next(r for r in match_timeline(rallies, match) if r["index"] == index)
        kind = row["kind"]
        names = {p["player_id"]: p["name"] for p in normalize_roster(match.get("players"))}
        if kind == "redo":
            return "Redo: No point or player statistics; server and receiver stay the same."
        text = f"{CLASSIFICATIONS[kind]['label']}: {match['team_' + row['winner'].lower()]} +1 point."
        if kind == "ace":
            text += f" {names[row['server_id']]} +1 ace; {names[row['receiver_id']]} +1 aced."
        elif kind == "double_fault":
            text += f" {names[row['server_id']]} +1 double fault (two service errors)."
        elif kind == "error":
            text += f" {names[rallies[index].classification['player_id']]} +1 unforced error."
        return text

    def lock_job(self, project_id, job_id, revision=None):
        project = self.get(project_id)
        with project.lock:
            self._check_revision(project, revision)
            self.require_source(project_id)
            if project.busy_job_id:
                raise ProjectError("A job is already running for this project", code="project_busy", status=409)
            if not project.state["match_settings"]["setup_complete"]:
                raise ValueError("Complete match setup before starting a job")
            project.busy_job_id = job_id
            return deepcopy(project.state)

    def unlock_job(self, project_id, job_id):
        project = self.get(project_id)
        with project.lock:
            if project.busy_job_id == job_id:
                project.busy_job_id = None

    def finish_job(self, project_id, job_id, updates=None):
        project = self.get(project_id)
        with project.lock:
            if project.busy_job_id != job_id:
                raise ProjectError("This job no longer owns the project", code="job_conflict", status=409)
            if updates:
                self.require_source(project_id)
                state = {**deepcopy(project.state), **deepcopy(updates)}
                state["signals"] = normalize_signals(state.get("signals"))
                state["revision"] = project.state["revision"] + 1
                if "rallies" in updates:
                    state["match_settings"]["stats_complete"] = False
                    state["complete"] = False
                    state["complete_review_confirmed"] = False
                    state["highlight_queue"] = []
                    state["selected"] = 0 if state["rallies"] else -1
                    state["explanation"] = "Rallies found. Review clip boundaries and choose each outcome."
                persisted = self._persist(state)
                project.history.push(project.state)
                project.state = persisted
            project.busy_job_id = None
            return self.snapshot(project_id)
