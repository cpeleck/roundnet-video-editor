"""Project recovery, review commands, and match annotations for the editor."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QSignalBlocker, QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from models import Rally
from models.match_flow import CLASSIFICATIONS, match_timeline, normalize_classification
from models.point_stats import normalize_roster
from models.project import EditHistory, read_project, recovery_path, source_matches, write_project


def result_snapshot(result):
    if result is None:
        return None
    values = dict(result) if isinstance(result, dict) else vars(result)
    return {key: value.tolist() if isinstance(value, np.ndarray) else deepcopy(value)
            for key, value in values.items() if key != "rallies"}


class ProjectWorkflow:
    def _init_workflow(self):
        self.court_context = None
        self.match_settings = {"team_a": "Team A", "team_b": "Team B",
                               "initial_score_a": 0, "initial_score_b": 0,
                               "starting_server": "A1", "starting_receiver": "B1",
                               "target_score": 21, "setup_complete": False}
        self.project_path = None
        self._history = EditHistory()
        self._restoring = False
        self._project_dirty = False
        self._manual_start = None
        self.profile_path = ""
        self.analysis_settings = None
        self.analysis_roi = None
        self.analysis_court_context = None
        # Keep recovery data beside the application, never beside read-only footage.
        self._recovery_dir = Path(__file__).resolve().parents[1] / ".roundnet" / "recovery"
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(750)
        self._autosave_timer.timeout.connect(self._autosave)

    def _build_workflow_actions(self, file_menu):
        self.save_project_action = QAction("Save Project…", self)
        self.save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_project_action.triggered.connect(self.save_project)
        self.open_project_action = QAction("Open Project…", self)
        self.open_project_action.triggered.connect(self.open_project)
        self.decisions_action = QAction("Export Edit Timeline (EDL / JSON)…", self)
        self.decisions_action.triggered.connect(self.export_decisions)
        self.import_decisions_action = QAction("Import Corrected Timeline…", self)
        self.import_decisions_action.triggered.connect(self.import_decisions)
        for action in (self.open_project_action, self.save_project_action,
                       self.decisions_action, self.import_decisions_action):
            file_menu.addAction(action)
        edit = self.menuBar().addMenu("Edit")
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo_edit)
        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self.redo_edit)
        self.restore_action = QAction("Restore Selected Rally", self)
        self.restore_action.triggered.connect(self.restore_rally)
        for action in (self.undo_action, self.redo_action, self.restore_action):
            edit.addAction(action)
        learn_menu = self.menuBar().addMenu("Learning")
        train = learn_menu.addAction("Train / Evaluate Local Profile…")
        train.triggered.connect(self.show_learning)
        load = learn_menu.addAction("Use Saved Profile…")
        load.triggered.connect(self.load_learning_profile)
        reset = learn_menu.addAction("Use Default Detector")
        reset.triggered.connect(self.clear_learning_profile)

    def _build_review_toolbar(self, layout):
        row = QHBoxLayout()
        self.review_filter = QComboBox()
        for label, key in (("All rallies", "all"), ("Needs review", "review"),
                           ("Starred highlights", "starred"), ("Rejected / unchecked", "rejected")):
            self.review_filter.addItem(label, key)
        self.review_filter.currentIndexChanged.connect(self._apply_review_filter)
        self.next_review_button = QPushButton("Next uncertain")
        self.next_review_button.clicked.connect(self.next_uncertain)
        row.addWidget(self.review_filter, 1)
        row.addWidget(self.next_review_button)
        layout.addLayout(row)
        self.complete_review_checkbox = QCheckBox("I reviewed the whole video for missed rallies")
        self.complete_review_checkbox.setToolTip(
            "Only enable when every rally is accounted for. Otherwise unlabeled footage stays unknown during training."
        )
        self.complete_review_checkbox.toggled.connect(self._changed)
        layout.addWidget(self.complete_review_checkbox)

    def _build_point_panel(self):
        page = QWidget()
        form = QFormLayout(page)
        self.star_button = QPushButton("☆ Star highlight")
        self.star_button.clicked.connect(self.toggle_star)
        self.reviewed_button = QPushButton("Mark reviewed")
        self.reviewed_button.clicked.connect(self.mark_reviewed)
        row = QHBoxLayout()
        row.addWidget(self.star_button)
        row.addWidget(self.reviewed_button)
        form.addRow(row)
        self.winner_combo = QComboBox()
        for label, key in (("Point winner — unknown", ""), ("Team A wins", "A"), ("Team B wins", "B")):
            self.winner_combo.addItem(label, key)
        self.outcome_combo = QComboBox()
        self.outcome_combo.addItems(("", "Ace", "Double fault", "Service break", "Sideout",
                                     "Defensive break", "Defensive hold", "Error", "Replay / no point"))
        self.player_edit = QLineEdit()
        self.player_edit.setPlaceholderText("Player credited (optional)")
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Call or caption, e.g. OOB")
        save = QPushButton("Save older manual tag")
        save.clicked.connect(self.apply_point_tag)
        self.manual_tag_save_button = save
        row2 = QHBoxLayout()
        setup = QPushButton("Match setup: teams, players, first serve…")
        self.match_setup_button = setup
        setup.clicked.connect(self.edit_match)
        stats = QPushButton("Match statistics")
        stats.clicked.connect(self.show_match_statistics)
        row2.addWidget(setup)
        row2.addWidget(stats)
        form.addRow(row2)
        self.match_score_label = QLabel("Team A  0 : 0  Team B")
        self.match_score_label.setWordWrap(True)
        form.addRow(self.match_score_label)
        self.assignment_label = QLabel("Set up the match to see the next server and receiver")
        self.assignment_label.setWordWrap(True)
        form.addRow(self.assignment_label)
        classification_grid = QGridLayout()
        self.classification_buttons = {}
        for position, (kind, details) in enumerate(CLASSIFICATIONS.items()):
            button = QPushButton(details["label"])
            button.setToolTip(details["description"])
            button.clicked.connect(lambda checked=False, selected=kind: self.classify_selected(selected))
            self.classification_buttons[kind] = button
            classification_grid.addWidget(button, position // 2, position % 2)
        form.addRow(classification_grid)
        self.error_player_combo = QComboBox()
        self.error_player_combo.addItem("Choose player responsible for Error", "")
        form.addRow("Error by", self.error_player_combo)
        self.classification_status = QLabel("Choose one outcome to score this clip and advance to the next.")
        self.classification_status.setWordWrap(True)
        form.addRow(self.classification_status)
        self.last_classification_label = QLabel("")
        self.last_classification_label.setWordWrap(True)
        self.last_classification_label.setStyleSheet("color: #9bd8a7; font-weight: 600")
        form.addRow(self.last_classification_label)
        clear_classification = QPushButton("Clear clip outcome")
        clear_classification.clicked.connect(self.clear_selected_classification)
        self.clear_classification_button = clear_classification
        form.addRow(clear_classification)
        self.touch_stats_button = QPushButton("Optional touch details for percentages / RPR…")
        self.touch_stats_button.clicked.connect(self.edit_point_statistics)
        form.addRow(self.touch_stats_button)
        self.touch_stats_status = QLabel("No touch statistics recorded")
        self.touch_stats_status.setWordWrap(True)
        form.addRow(self.touch_stats_status)
        form.addRow("Caption", self.note_edit)
        self.save_caption_button = QPushButton("Save caption")
        self.save_caption_button.clicked.connect(lambda: self._set_selected(note=self.note_edit.text().strip()))
        form.addRow(self.save_caption_button)
        legacy_toggle = QCheckBox("Older manual tags")
        legacy_toggle.setToolTip("Use this for projects created before clip outcome buttons were added")
        legacy_panel = QWidget()
        legacy_form = QFormLayout(legacy_panel)
        legacy_form.addRow("Point winner", self.winner_combo)
        legacy_form.addRow("Outcome", self.outcome_combo)
        legacy_form.addRow("Player", self.player_edit)
        legacy_form.addRow(save)
        legacy_panel.setVisible(False)
        legacy_toggle.toggled.connect(legacy_panel.setVisible)
        form.addRow(legacy_toggle)
        form.addRow(legacy_panel)
        crop = QPushButton("Set assisted crop keyframes…")
        crop.clicked.connect(self.edit_crop)
        form.addRow(crop)
        return page

    def _edit_snapshot(self):
        return {"rallies": [r.to_dict() for r in self.rallies],
                "rejected_detections": deepcopy(self.rejected_detections),
                "match_settings": deepcopy(self.match_settings),
                "selected": self.selected_rally_index,
                "complete": self.complete_review_checkbox.isChecked()}

    def _analysis_snapshot(self):
        return {"signals": result_snapshot(self.detection_result),
                "initial_predictions": deepcopy(self.initial_detected_rallies),
                "analysis_settings": deepcopy(self.analysis_settings),
                "analysis_roi": deepcopy(self.analysis_roi),
                "analysis_court_context": deepcopy(self.analysis_court_context)}

    def _checkpoint(self, *, include_analysis=False, stats_affecting=True):
        if not self._restoring:
            state = self._edit_snapshot()
            if include_analysis:
                state["analysis"] = self._analysis_snapshot()
            self._history.push(state)
            if stats_affecting and self.match_settings.get("stats_complete"):
                self.match_settings["stats_complete"] = False
            if stats_affecting and self.complete_review_checkbox.isChecked():
                with QSignalBlocker(self.complete_review_checkbox):
                    self.complete_review_checkbox.setChecked(False)

    def _changed(self, *_):
        if self._restoring or not self.video_path:
            return
        self._project_dirty = True
        self._autosave_timer.start()
        self.undo_action.setEnabled(bool(self._history.undo_stack))
        self.redo_action.setEnabled(bool(self._history.redo_stack))
        self._apply_review_filter()

    def _restore_edit(self, state):
        was_restoring = self._restoring
        self._restoring = True
        try:
            if "analysis" in state:
                self._restore_analysis(state["analysis"])
            self.rallies = [Rally.from_dict(r) for r in state["rallies"]]
            self.rejected_detections = deepcopy(state["rejected_detections"])
            self.match_settings = deepcopy(state["match_settings"])
            self.selected_rally_index = state["selected"]
            self.complete_review_checkbox.setChecked(state["complete"])
            self._rebuild_rally_tree()
            self._update_summary()
            self._update_ui_state()
            self.last_classification_label.setText("Score and serving assignments refreshed from the saved clips.")
        finally:
            self._restoring = was_restoring
        self._changed()

    def undo_edit(self):
        if self._busy_editing():
            return
        current = self._edit_snapshot()
        if self._history.undo_stack and "analysis" in self._history.undo_stack[-1]:
            current["analysis"] = self._analysis_snapshot()
        state = self._history.undo(current)
        if state is not None:
            self._restore_edit(state)

    def redo_edit(self):
        if self._busy_editing():
            return
        current = self._edit_snapshot()
        if self._history.redo_stack and "analysis" in self._history.redo_stack[-1]:
            current["analysis"] = self._analysis_snapshot()
        state = self._history.redo(current)
        if state is not None:
            self._restore_edit(state)

    def _project_snapshot(self):
        return {**self._edit_snapshot(), "video_path": self.video_path,
                "video_duration": self.video_duration, "roi": self.roi,
                "court_context": self.court_context, "settings": self.detection_settings,
                "profile_path": self.profile_path, "initial_predictions": self.initial_detected_rallies,
                "signals": result_snapshot(self.detection_result),
                "analysis_settings": self.analysis_settings, "analysis_roi": self.analysis_roi,
                "analysis_court_context": self.analysis_court_context,
                "position": self.player.position_seconds}

    def _autosave(self):
        if not self.video_path or self._restoring:
            return True
        try:
            state = self._project_snapshot()
            saved = write_project(recovery_path(self.video_path, self._recovery_dir), state)
            if self.project_path:
                write_project(self.project_path, state)
            self._settings_store.setValue("project/last_recovery", str(saved))
            self._project_dirty = False
            self.statusBar().showMessage("Project autosaved", 2000)
            return True
        except Exception as exc:
            self.statusBar().showMessage(f"Autosave failed: {exc}. Use Save Project to choose another folder.", 12000)
            return False

    def _ensure_saved_before_leaving(self):
        if not self.video_path or not self._project_dirty or self._autosave():
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Your Edits Could Not Be Saved")
        box.setText("Autosave failed. Save the project somewhere writable before leaving this video, or explicitly discard the unsaved changes.")
        save = box.addButton("Save Project As…", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Discard Unsaved Changes", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is discard:
            return True
        if box.clickedButton() is not save:
            return False
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "game.roundnet.json", "Roundnet Project (*.json)")
        if not path:
            return False
        try:
            self.project_path = str(write_project(path, self._project_snapshot()))
            self._project_dirty = False
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Could Not Save Project", str(exc))
            return False

    def save_project(self):
        if not self.video_path:
            return
        path = self.project_path
        if not path:
            suggested = str(Path(self.video_path).with_suffix(".roundnet.json"))
            path, _ = QFileDialog.getSaveFileName(self, "Save Roundnet Project", suggested, "Roundnet Project (*.roundnet.json)")
        if path:
            try:
                self.project_path = str(write_project(path, self._project_snapshot()))
                self._project_dirty = False  # the explicit project save is durable
                self._autosave()
            except Exception as exc:
                QMessageBox.warning(self, "Could Not Save Project", str(exc))

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Roundnet Project", "", "Roundnet Project (*.json)")
        if path:
            self._open_project_path(path, explicit=True)

    def _open_project_path(self, path, *, explicit=False):
        if self._detection_worker and self._detection_worker.isRunning() or self._export_worker and self._export_worker.isRunning():
            return
        try:
            state = read_project(path)
            source = Path(state["video_path"])
            if not source.is_file():
                if not explicit:
                    return
                replacement, _ = QFileDialog.getOpenFileName(self, f"Locate source video: {source.name}")
                if not replacement:
                    return
                state["video_path"] = replacement
            if not source_matches(state, state["video_path"]):
                if not explicit:
                    self.statusBar().showMessage("Recovery skipped: the source recording has changed. Open Project to review it.", 10000)
                    return
                answer = QMessageBox.question(self, "Source Recording Changed",
                    "The source file no longer matches this project. Use its saved cuts with this recording anyway?")
                if answer != QMessageBox.StandardButton.Yes:
                    return
                # Cuts may be reused intentionally, but features and labels from
                # a different recording must never become training evidence.
                state = deepcopy(state)
                for key in ("signals", "analysis_settings", "analysis_roi", "analysis_court_context"):
                    state[key] = None
                state["initial_predictions"] = []
                state["rejected_detections"] = []
                state["complete"] = False
                state["rallies"] = [
                    {**r, "reviewed": False, "point_stats": {}, "classification": {},
                     **({"winner": "", "outcome": "", "player": ""} if r.get("classification") else {})}
                    for r in state.get("rallies", [])]
                state.setdefault("match_settings", {})["stats_complete"] = False
            if not self.load_video(state["video_path"], restore_recovery=False, prompt_setup=False):
                return
            if any(r["end_time"] > self.video_duration + .05 for r in state.get("rallies", [])):
                raise ValueError("Saved cuts extend beyond this recording. Locate the original source video.")
            self._apply_project(state)
            self.project_path = str(path) if explicit else None
            self.statusBar().showMessage("Restored saved project, including edits and analysis", 8000)
        except Exception as exc:
            if explicit:
                QMessageBox.warning(self, "Could Not Open Project", str(exc))
            else:
                self.statusBar().showMessage(f"Recovery could not be loaded: {exc}", 10000)

    def _recover_last_project(self):
        path = self._settings_store.value("project/last_recovery", "")
        if path and Path(str(path)).is_file() and not self.video_path:
            self._open_project_path(str(path))

    def _apply_project(self, state):
        self._restoring = True
        try:
            self.roi = tuple(state["roi"]) if state.get("roi") else None
            self.court_context = state.get("court_context")
            self.detection_settings.update(state.get("settings", {}))
            saved_match = state.get("match_settings", {})
            self.match_settings.update(saved_match)
            self.match_settings["setup_complete"] = saved_match.get("setup_complete", True)
            self.profile_path = state.get("profile_path", "")
            self._restore_analysis(state)
            self._restore_edit({"rallies": state.get("rallies", []),
                                "rejected_detections": state.get("rejected_detections", []),
                                "match_settings": self.match_settings, "selected": state.get("selected", -1),
                                "complete": state.get("complete", False)})
            self._history = EditHistory()
            self._update_roi_status()
            self.timeline.set_threshold(self.detection_settings.get("rally_threshold", .55))
            self.player.set_position(state.get("position", 0))
        finally:
            self._restoring = False
        self._update_workflow_state()
        self._project_dirty = False

    def _restore_analysis(self, state):
        self.initial_detected_rallies = deepcopy(state.get("initial_predictions", []))
        self.analysis_settings = deepcopy(state.get("analysis_settings"))
        self.analysis_roi = deepcopy(state.get("analysis_roi"))
        self.analysis_court_context = deepcopy(state.get("analysis_court_context"))
        signals = state.get("signals")
        self.detection_result = SimpleNamespace(**{
            key: np.asarray(value) if key == "timestamps" or key.endswith("_scores") else value
            for key, value in signals.items()}) if signals else None
        self.timeline.set_signal_data(self.detection_result)

    def _apply_review_filter(self, *_):
        if not hasattr(self, "review_filter"):
            return
        mode = self.review_filter.currentData()
        for i, rally in enumerate(self.rallies):
            show = (mode == "all" or mode == "review" and not rally.reviewed
                    or mode == "starred" and rally.starred and not rally.rejected
                    or mode == "rejected" and (rally.rejected or not rally.enabled))
            item = self.rally_tree.topLevelItem(i)
            if item:
                item.setHidden(not show)

    def next_uncertain(self):
        candidates = [(r.confidence + .2 * r.serve_confidence, i)
                      for i, r in enumerate(self.rallies) if not r.reviewed and not r.rejected]
        if candidates:
            with QSignalBlocker(self.review_filter):
                self.review_filter.setCurrentIndex(1)
            self._apply_review_filter()
            self._select_rally(min(candidates)[1], seek=True)
        else:
            self.statusBar().showMessage("Every candidate has been reviewed. Check the full timeline for missed rallies.", 7000)

    def _selected(self):
        return self.rallies[self.selected_rally_index] if 0 <= self.selected_rally_index < len(self.rallies) else None

    def _set_selected(self, **changes):
        rally = self._selected()
        if rally is None or self._busy_editing():
            return
        changed = {key for key, value in changes.items() if getattr(rally, key) != value}
        if not changed:
            return
        statistical_fields = {"winner", "point_stats", "rejected", "start_time", "end_time"}
        affects_stats = bool(statistical_fields.intersection(changed)) or (
            "outcome" in changed and (rally.outcome == "Replay / no point" or changes["outcome"] == "Replay / no point"))
        self._checkpoint(stats_affecting=affects_stats)
        self.rallies[self.selected_rally_index] = replace(rally, **changes)
        self._rebuild_rally_tree()
        self._update_summary()
        self._update_ui_state()
        self._changed()

    def _busy_editing(self):
        return bool(self._detection_worker and self._detection_worker.isRunning()
                    or self._export_worker and self._export_worker.isRunning()
                    or self.video_path and not self.match_settings.get("setup_complete", True))

    def restore_rally(self):
        rally = self._selected()
        if rally is None or self._busy_editing():
            return
        self._checkpoint()
        self.rejected_detections = [r for r in self.rejected_detections if r.get("rally_id") != rally.rally_id]
        self.rallies[self.selected_rally_index] = replace(rally, rejected=False, enabled=True, reviewed=True)
        self._rebuild_rally_tree()
        self._update_summary()
        self._update_ui_state()
        self._changed()

    def toggle_star(self):
        r = self._selected()
        if r:
            self._set_selected(starred=not r.starred)

    def mark_reviewed(self):
        self._set_selected(reviewed=True)

    def apply_point_tag(self):
        winner = self.winner_combo.currentData()
        outcome = self.outcome_combo.currentText()
        existing_rally = self._selected()
        if existing_rally and existing_rally.classification and (winner != existing_rally.winner or outcome != existing_rally.outcome):
            QMessageBox.warning(self, "Use the outcome buttons", "This clip is classified. Choose an outcome above to change its score, or clear the classification first.")
            return
        if outcome == "Replay / no point":
            winner = ""
        player_name = self.player_edit.text().strip()
        changes = {"winner": winner, "outcome": outcome, "player": player_name,
                   "note": self.note_edit.text().strip(), "reviewed": True}
        if outcome == "Ace":
            from models.point_stats import new_event, normalize_roster
            roster = normalize_roster(self.match_settings.get("players"))
            server = next((p["player_id"] for p in roster if p["name"].casefold() == player_name.casefold()), "")
            if not server:
                QMessageBox.warning(self, "Choose a roster player", "For an ace, enter one of the four player names set in Teams / players.")
                return
            current = self._selected()
            if current and current.point_stats.get("events"):
                existing = current.point_stats
                if (len(existing["events"]) != 1 or existing["events"][0]["kind"] != "serve"
                        or existing["events"][0]["result"] != "ace" or existing["server_id"] != server):
                    QMessageBox.warning(self, "Edit the touch log", "This point already has touches. Use Log point to change its ace and preserve the sequence.")
                    return
                changes["point_stats"] = existing
                changes["winner"] = server[0]
                self._set_selected(**changes)
                return
            receiver = self.match_settings.get("starting_receiver", "")
            if not receiver or receiver[0] == server[0]:
                receiver = next(p["player_id"] for p in roster if p["team"] != server[0])
            changes["winner"] = server[0]
            changes["point_stats"] = {"version": 1, "server_id": server, "receiver_id": receiver,
                                      "complete": True, "events": [new_event(server, "serve", "ace")]}
        self._set_selected(**changes)

    def _reconcile_classifications(self):
        """Refresh compatibility score/caption fields from canonical clip labels."""
        if not self.rallies or not self.match_settings.get("setup_complete", False):
            return
        roster = {p["player_id"]: p["name"] for p in normalize_roster(self.match_settings.get("players"))}
        for row in match_timeline(self.rallies, self.match_settings):
            rally = self.rallies[row["index"]]
            classification = rally.classification
            if not classification:
                continue
            kind = classification["kind"]
            actor = (classification.get("player_id") if kind == "error" else
                     row["server_id"] if kind in ("ace", "double_fault", "service_break") else
                     row["receiver_id"] if kind == "sideout" else "")
            if row["provisional"] and kind != "error":
                actor = ""
            outcome = "Replay / no point" if kind == "redo" else CLASSIFICATIONS[kind]["label"]
            player = roster.get(actor, "")
            if (rally.winner, rally.outcome, rally.player) != (row["winner"], outcome, player):
                self.rallies[row["index"]] = replace(rally, winner=row["winner"], outcome=outcome,
                                                      player=player)

    def classify_selected(self, kind):
        rally = self._selected()
        if not rally or rally.rejected or self._busy_editing():
            return
        timeline_before = match_timeline(self.rallies, self.match_settings)
        row_before = next(row for row in timeline_before if row["index"] == self.selected_rally_index)
        if row_before["provisional"]:
            earlier = next((row for row in timeline_before
                            if row["index"] != self.selected_rally_index and not row["winner"]
                            and row["kind"] != "redo" and not self.rallies[row["index"]].rejected), None)
            message = "Classify the earlier unresolved clip first so this server and receiver are correct."
            self.last_classification_label.setText(message)
            self.statusBar().showMessage(message, 8000)
            if earlier:
                self._select_rally(earlier["index"], seek=True)
            return
        player_id = self.error_player_combo.currentData() if kind == "error" else ""
        if kind == "error" and not player_id:
            self.classification_status.setText("Choose the player who made the unforced error, then click Error.")
            return
        value = normalize_classification({"version": 1, "kind": kind,
                                          **({"player_id": player_id} if kind == "error" else {})})
        if rally.classification == value:
            return
        current_index = self.selected_rally_index
        self._checkpoint()
        cleared_details = bool(rally.point_stats)
        self.rallies[current_index] = replace(rally, classification=value, reviewed=True, point_stats={})
        self._reconcile_classifications()
        next_index = next((i for i in range(current_index + 1, len(self.rallies))
                           if not self.rallies[i].rejected and not self.rallies[i].classification), current_index)
        self.selected_rally_index = next_index
        self._rebuild_rally_tree(next_index, seek=False)
        self._update_summary()
        self._update_ui_state()
        self._changed()
        timeline = match_timeline(self.rallies, self.match_settings)
        row = next(item for item in timeline if item["index"] == current_index)
        names = {p["player_id"]: p["name"] for p in normalize_roster(self.match_settings.get("players"))}
        if kind == "redo":
            credit = "No point or player statistics; serving assignment repeats."
        else:
            team_name = self.match_settings[f"team_{row['winner'].lower()}"]
            credit = f"{team_name} +1 point."
            if kind == "ace":
                credit += f" {names[row['server_id']]} +1 ace; {names[row['receiver_id']]} +1 aced."
            elif kind == "double_fault":
                credit += f" {names[row['server_id']]} +1 double fault (two service errors)."
            elif kind == "error":
                credit += f" {names[player_id]} +1 unforced error."
        next_row = next((item for item in timeline if item["index"] == next_index), None)
        next_serve = (" Next clip: " + names[next_row["server_id"]] + " serves to "
                      + names[next_row["receiver_id"]] + "."
                      if next_index != current_index and next_row else "")
        message = f"{CLASSIFICATIONS[kind]['label']}: {credit}{next_serve}"
        if cleared_details:
            message += " Earlier touch details cleared; Undo restores them."
        self.last_classification_label.setText(message)
        self.statusBar().showMessage(message, 8000)

    def clear_selected_classification(self):
        rally = self._selected()
        if not rally or not rally.classification or self._busy_editing():
            return
        self._checkpoint()
        self.rallies[self.selected_rally_index] = replace(rally, classification={}, winner="", outcome="", player="", point_stats={}, reviewed=False)
        self._reconcile_classifications()
        self._rebuild_rally_tree(self.selected_rally_index, seek=False)
        self._update_summary()
        self._update_ui_state()
        self._changed()
        self.last_classification_label.setText("Outcome cleared. This clip and later serving assignments are provisional until it is classified.")
        self.statusBar().showMessage("Outcome and any touch details cleared; Undo restores them", 6000)

    def _load_point_panel(self):
        rally = self._selected()
        if not hasattr(self, "winner_combo"):
            return
        for widget in (self.winner_combo, self.outcome_combo, self.player_edit, self.note_edit,
                       self.star_button, self.reviewed_button, self.touch_stats_button,
                       self.save_caption_button, self.manual_tag_save_button,
                       self.error_player_combo, self.clear_classification_button,
                       *self.classification_buttons.values()):
            widget.setEnabled(rally is not None and not self._busy_editing())
        if rally and rally.classification:
            for widget in (self.winner_combo, self.outcome_combo, self.player_edit, self.manual_tag_save_button):
                widget.setEnabled(False)
        current_error = self.error_player_combo.currentData()
        self.error_player_combo.clear()
        self.error_player_combo.addItem("Choose player responsible for Error", "")
        for player in normalize_roster(self.match_settings.get("players")):
            self.error_player_combo.addItem(f"{player['name']} · {player['player_id']}", player["player_id"])
        self.error_player_combo.setCurrentIndex(max(0, self.error_player_combo.findData(current_error)))
        if rally:
            self.touch_stats_button.setEnabled(not rally.rejected and not self._busy_editing())
            self.clear_classification_button.setEnabled(bool(rally.classification) and not self._busy_editing())
            if rally.classification.get("kind") == "error":
                self.error_player_combo.setCurrentIndex(self.error_player_combo.findData(rally.classification["player_id"]))
            for kind, button in self.classification_buttons.items():
                button.setEnabled(not rally.rejected and not self._busy_editing())
                button.setStyleSheet("background: #1f6feb; font-weight: bold" if rally.classification.get("kind") == kind else "")
            rows = match_timeline(self.rallies, self.match_settings)
            assignment = next((row for row in rows if row["index"] == self.selected_rally_index), None)
            names = {p["player_id"]: p["name"] for p in normalize_roster(self.match_settings.get("players"))}
            if assignment:
                a, b = assignment["score_before"]
                server_id, receiver_id = assignment["server_id"], assignment["receiver_id"]
                server_partner = server_id[0] + ("2" if server_id[1] == "1" else "1")
                receiver_partner = receiver_id[0] + ("2" if receiver_id[1] == "1" else "1")
                self.assignment_label.setText(
                    f"Point {assignment['scored_index'] + 1} · {self.match_settings['team_a']} {a} : {b} {self.match_settings['team_b']}\n"
                    f"Serving: {names[server_id]} · partner {names[server_partner]}\n"
                    f"Receiving: {names[receiver_id]} · partner {names[receiver_partner]}"
                    + (" · provisional until earlier clips are classified" if assignment["provisional"] else ""))
                if assignment["provisional"]:
                    for button in self.classification_buttons.values():
                        button.setEnabled(False)
            self.classification_status.setText(
                "Classify earlier unresolved clips first; this serving assignment may change."
                if assignment and assignment["provisional"] else
                CLASSIFICATIONS[rally.classification["kind"]]["description"] if rally.classification else
                "Choose one outcome to score this clip and advance to the next.")
            self.winner_combo.setCurrentIndex(max(0, self.winner_combo.findData(rally.winner)))
            self.outcome_combo.setCurrentText(rally.outcome)
            self.player_edit.setText(rally.player)
            self.note_edit.setText(rally.note)
            self.star_button.setText("★ Starred" if rally.starred else "☆ Star highlight")
            self.reviewed_button.setText("✓ Reviewed" if rally.reviewed else "Mark reviewed")
            from models.statistics import point_issues
            stats = rally.point_stats
            issues = point_issues(stats, rally.winner, start=rally.start_time, end=rally.end_time)
            self.touch_stats_status.setText("Replay — excluded from point statistics" if rally.outcome == "Replay / no point" else
                "Touch log needs review: " + issues[0] if issues else
                f"{len(stats.get('events', []))} events · {'Complete point' if stats.get('complete') else 'Draft / untagged'}")
        else:
            self.touch_stats_status.setText("Select a rally to record its touches")
            self.assignment_label.setText("Select a clip to see its server and receiver")
            self.classification_status.setText("Choose one outcome after selecting a clip")

    def edit_point_statistics(self):
        rally = self._selected()
        if not rally or rally.rejected or self._busy_editing() or not self.video_path:
            return
        from .point_statistics import PointStatisticsDialog
        self.player.pause()
        row = next((item for item in match_timeline(self.rallies, self.match_settings)
                    if item["index"] == self.selected_rally_index), None)
        point_settings = dict(self.match_settings)
        if row:
            point_settings.update(starting_server=row["server_id"], starting_receiver=row["receiver_id"])
        dialog = PointStatisticsDialog(self.video_path, rally, point_settings, self)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                if rally.classification:
                    self._set_selected(point_stats=dialog.point, reviewed=True)
                else:
                    outcome = "Replay / no point" if dialog.replay else "" if rally.outcome == "Replay / no point" else rally.outcome
                    self._set_selected(point_stats=dialog.point, winner=dialog.winner, outcome=outcome, reviewed=True)
        finally:
            dialog.video.unload()
            dialog.deleteLater()

    def edit_match(self):
        if self._detection_worker and self._detection_worker.isRunning() or self._export_worker and self._export_worker.isRunning():
            return
        first_setup = not self.match_settings.get("setup_complete", False)
        dialog = QDialog(self)
        dialog.setWindowTitle("Match setup")
        layout = QFormLayout(dialog)
        controls = {}
        for key, label in (("team_a", "Team A"), ("team_b", "Team B"),
                            ("initial_score_a", "Initial score A"), ("initial_score_b", "Initial score B")):
            if key.startswith("team"):
                default_name = "Team A" if key == "team_a" else "Team B"
                control = QLineEdit("" if first_setup and self.match_settings[key] == default_name
                                    else str(self.match_settings[key]))
                control.setPlaceholderText(default_name)
                control.setMaxLength(100)
            else:
                control = QSpinBox()
                control.setRange(0, 999)
                control.setValue(int(self.match_settings[key]))
            controls[key] = control
            layout.addRow(label, control)
        from models.point_stats import normalize_roster
        roster = normalize_roster(self.match_settings.get("players"))
        player_controls = {}
        for player in roster:
            default_name = f"Player {player['player_id']}"
            name = QLineEdit("" if first_setup and player["name"] == default_name else player["name"])
            name.setPlaceholderText(default_name)
            name.setMaxLength(80)
            player_controls[player["player_id"]] = name
            layout.addRow(f"Player {player['player_id']}", name)
        starter = QComboBox()
        receiver = QComboBox()
        for p in roster:
            for combo in (starter, receiver):
                combo.addItem(f"{p['name']} · {p['player_id']}", p["player_id"])
        def update_roster_choices():
            for player in roster:
                pid = player["player_id"]
                label = player_controls[pid].text().strip() or pid
                for combo in (starter, receiver):
                    combo.setItemText(combo.findData(pid), f"{label} · {pid}")
        for control in player_controls.values():
            control.textChanged.connect(lambda *_: update_roster_choices())
        update_roster_choices()
        starter.setCurrentIndex(max(0, starter.findData(self.match_settings.get("starting_server", "A1"))))
        receiver.setCurrentIndex(max(0, receiver.findData(self.match_settings.get("starting_receiver", "B1"))))
        layout.addRow("Starting server", starter)
        layout.addRow("Starting receiver", receiver)
        target_score = QSpinBox()
        target_score.setRange(2, 99)
        target_score.setValue(int(self.match_settings.get("target_score", 21)))
        layout.addRow("Points to win", target_score)
        note = QLabel("Player slots stay stable when names change, so past touch credits follow the rename.")
        note.setWordWrap(True)
        layout.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        def validate_roster_and_accept():
            try:
                normalize_roster([{**p, "name": player_controls[p["player_id"]].text()} for p in roster])
                if first_setup:
                    if any(player_controls[p["player_id"]].text().strip() == f"Player {p['player_id']}" for p in roster):
                        raise ValueError("Enter a name for each of the four players before editing")
                    if not controls["team_a"].text().strip() or not controls["team_b"].text().strip():
                        raise ValueError("Enter both team names before editing")
                if starter.currentData()[0] == receiver.currentData()[0]:
                    raise ValueError("Starting server and receiver must be on opposite teams")
                dialog.accept()
            except ValueError as exc:
                QMessageBox.warning(dialog, "Player names", str(exc))
        buttons.accepted.connect(validate_roster_and_accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            score_changed = any(controls[key].value() != self.match_settings[key]
                                for key in ("initial_score_a", "initial_score_b"))
            order_changed = (starter.currentData() != self.match_settings.get("starting_server")
                             or receiver.currentData() != self.match_settings.get("starting_receiver")
                             or target_score.value() != self.match_settings.get("target_score", 21))
            if self.match_settings.get("setup_complete", False):
                self._checkpoint(stats_affecting=score_changed or order_changed)
            self.match_settings.update({key: (value.text().strip() or ("Team A" if key == "team_a" else "Team B")) if key.startswith("team") else value.value()
                                        for key, value in controls.items()})
            self.match_settings["players"] = normalize_roster([{**p, "name": player_controls[p["player_id"]].text()} for p in roster])
            self.match_settings["starting_server"] = starter.currentData()
            self.match_settings["starting_receiver"] = receiver.currentData()
            self.match_settings["target_score"] = target_score.value()
            self.match_settings["setup_complete"] = True
            self._reconcile_classifications()
            self._update_summary()
            self._update_ui_state()
            self._changed()

    def _update_match_score(self):
        if not hasattr(self, "match_score_label"):
            return
        points = [r for r in self.rallies if not r.rejected and r.outcome != "Replay / no point"]
        a = self.match_settings["initial_score_a"] + sum(r.winner == "A" for r in points)
        b = self.match_settings["initial_score_b"] + sum(r.winner == "B" for r in points)
        rows = match_timeline(self.rallies, self.match_settings)
        provisional = any(row["provisional"] or (not row["winner"] and row["kind"] != "redo"
                          and not self.rallies[row["index"]].rejected
                          and self.rallies[row["index"]].outcome != "Replay / no point") for row in rows)
        suffix = " · provisional until every clip is classified" if provisional else ""
        self.match_score_label.setText(f"{self.match_settings['team_a']}  {a} : {b}  {self.match_settings['team_b']}{suffix}")

    def show_match_statistics(self):
        if self._busy_editing():
            return
        from .statistics_dialog import StatisticsDialog
        dialog = StatisticsDialog(self.rallies, self.match_settings, self,
                                  protected_paths=(self.video_path, self.project_path))
        def select_point(identity, timestamp):
            index = next((i for i, r in enumerate(self.rallies) if r.rally_id == identity), -1)
            if index >= 0:
                self.review_filter.setCurrentIndex(0)
                self._select_rally(index, seek=True)
        dialog.point_requested.connect(select_point)
        dialog.exec()
        confirmed = dialog.complete.isChecked()
        if confirmed != bool(self.match_settings.get("stats_complete")):
            self._checkpoint()
            self.match_settings["stats_complete"] = confirmed
            self._changed()
        dialog.deleteLater()

    def edit_crop(self):
        rally = self._selected()
        if not rally or not self.video_path or self._busy_editing():
            return
        from .crop_editor import CropEditorDialog
        try:
            dialog = CropEditorDialog(self.video_path, rally, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._set_selected(crop_keyframes=dialog.keyframes)
        except Exception as exc:
            QMessageBox.warning(self, "Crop Editor", str(exc))

    def select_court(self):
        if not self.video_path or self._busy_editing():
            return
        from .court_setup import CourtSetupDialog
        try:
            dialog = CourtSetupDialog(self.video_path, self.court_context, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.court_context = deepcopy(dialog.court_context)
                self._changed()
                self.statusBar().showMessage("Court setup saved. Run detection to use the new context.", 6000)
        except Exception as exc:
            QMessageBox.warning(self, "Court Setup", str(exc))

    def show_learning(self):
        from .learning_dialog import LearningDialog
        dialog = LearningDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.profile_path:
            self.profile_path = str(dialog.profile_path)
            self._changed()
            self.statusBar().showMessage("Local profile selected; Detect Rallies will use it", 6000)

    def load_learning_profile(self):
        from training.calibration import load_profile
        path, _ = QFileDialog.getOpenFileName(self, "Select Local Detection Profile", "", "Profile (*.json)")
        if path:
            try:
                load_profile(path)
                self.profile_path = path
                self._changed()
                self.statusBar().showMessage("Profile selected for next detection", 5000)
            except Exception as exc:
                QMessageBox.warning(self, "Invalid Profile", str(exc))

    def clear_learning_profile(self):
        self.profile_path = ""
        self._changed()
        self.statusBar().showMessage("Using default detector", 4000)

    def export_decisions(self):
        if not self.video_path:
            return
        from video.edit_decisions import save_edit_decisions
        path, _ = QFileDialog.getSaveFileName(self, "Export Editable Source Timeline", "rallies.edl", "EDL (*.edl);;JSON (*.json)")
        if path:
            try:
                fps = self._source_fps()
                save_edit_decisions(path, self.video_path, self.rallies, fps=fps, source_duration=self.video_duration)
                self.statusBar().showMessage("Exported source timeline for another editor", 6000)
            except Exception as exc:
                QMessageBox.warning(self, "Timeline Export Failed", str(exc))

    def import_decisions(self):
        if not self.video_path or self._busy_editing():
            return
        from training.evaluation import load_reference_intervals
        path, _ = QFileDialog.getOpenFileName(self, "Import Corrected Source Timeline", "", "Source timeline (*.json *.edl)")
        if not path:
            return
        try:
            intervals = load_reference_intervals(path, fps=self._source_fps())
            if any(float(r["end"]) > self.video_duration + .05 for r in intervals):
                raise ValueError("This timeline extends beyond the loaded recording. Import cuts from the same original video.")
            corrected = [Rally(float(r["start"]), min(self.video_duration, float(r["end"])), reviewed=True) for r in intervals]
            if QMessageBox.question(self, "Replace Current Rally Cuts?",
                    f"Import {len(corrected)} source-time cuts for {Path(self.video_path).name}? "
                    "Confirm these cuts use this original recording with a zero start time. "
                    "Current tags and cuts will be replaced; Undo restores them. "
                    "Only confirm full-video review if no rallies are missing.") != QMessageBox.StandardButton.Yes:
                return
            self._checkpoint()
            self.rallies = corrected
            self.rejected_detections = []
            self.selected_rally_index = 0 if corrected else -1
            self._rebuild_rally_tree()
            self._update_summary()
            self._update_ui_state()
            self._changed()
        except Exception as exc:
            QMessageBox.warning(self, "Timeline Import Failed", str(exc))

    def _source_fps(self):
        value = self.video_metadata.get("fps", 30) if isinstance(self.video_metadata, dict) else getattr(self.video_metadata, "fps", 30)
        return float(value or 30)

    def _update_workflow_state(self):
        if not hasattr(self, "undo_action"):
            return
        busy = self._busy_editing()
        processing = bool(self._detection_worker and self._detection_worker.isRunning()
                          or self._export_worker and self._export_worker.isRunning())
        self.match_setup_button.setEnabled(bool(self.video_path) and not processing)
        self.undo_action.setEnabled(not busy and bool(self._history.undo_stack))
        self.redo_action.setEnabled(not busy and bool(self._history.redo_stack))
        self.restore_action.setEnabled(not busy and self._selected() is not None)
        self.open_project_action.setEnabled(not processing)
        self.import_decisions_action.setEnabled(not busy)
        self.save_project_action.setEnabled(bool(self.video_path) and not busy)
        self.decisions_action.setEnabled(bool(self.video_path) and not busy)
        self.import_decisions_action.setEnabled(bool(self.video_path) and not busy)
        self.complete_review_checkbox.setEnabled(bool(self.video_path) and not busy)
        self.next_review_button.setEnabled(bool(self.rallies) and not busy)
        self._load_point_panel()

    def manual_mark_start(self):
        if self.video_path and not self._busy_editing():
            self._manual_start = self.player.position_seconds
            self.statusBar().showMessage("Rally start marked. Press X at the end to add it.", 7000)

    def manual_mark_end(self):
        if self._manual_start is None or self._busy_editing():
            return
        end = self.player.position_seconds
        if end <= self._manual_start:
            return
        self._checkpoint()
        rally = Rally(self._manual_start, end, reviewed=True)
        self.rallies.append(rally)
        self.rallies.sort(key=lambda r: r.start_time)
        self.selected_rally_index = self.rallies.index(rally)
        self._manual_start = None
        self._rebuild_rally_tree()
        self._update_summary()
        self._update_ui_state()
        self._changed()

    def play_all_rallies(self):
        self.player.preview_ranges([(r.start_time, r.end_time) for r in self._enabled_rallies()])

    def adjacent_rally(self, step):
        if self.rallies:
            self._select_rally(max(0, min(len(self.rallies)-1, self.selected_rally_index + step)), seek=True)
