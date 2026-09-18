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
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from models import Rally
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
                               "initial_score_a": 0, "initial_score_b": 0}
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
        form.addRow("Point winner", self.winner_combo)
        form.addRow("Outcome", self.outcome_combo)
        form.addRow("Player", self.player_edit)
        form.addRow("Caption", self.note_edit)
        save = QPushButton("Apply point tag")
        save.clicked.connect(self.apply_point_tag)
        form.addRow(save)
        row2 = QHBoxLayout()
        setup = QPushButton("Teams / initial score…")
        setup.clicked.connect(self.edit_match)
        stats = QPushButton("Match statistics")
        stats.clicked.connect(self.show_match_statistics)
        row2.addWidget(setup)
        row2.addWidget(stats)
        form.addRow(row2)
        self.match_score_label = QLabel("Team A  0 : 0  Team B")
        self.match_score_label.setWordWrap(True)
        form.addRow(self.match_score_label)
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

    def _checkpoint(self, *, include_analysis=False):
        if not self._restoring:
            state = self._edit_snapshot()
            if include_analysis:
                state["analysis"] = self._analysis_snapshot()
            self._history.push(state)
            if self.complete_review_checkbox.isChecked():
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
        if self._busy_editing():
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
                state["rallies"] = [{**r, "reviewed": False} for r in state.get("rallies", [])]
            if not self.load_video(state["video_path"], restore_recovery=False):
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
            self.match_settings.update(state.get("match_settings", {}))
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
        self._checkpoint()
        self.rallies[self.selected_rally_index] = replace(rally, **changes)
        self._rebuild_rally_tree()
        self._update_summary()
        self._update_ui_state()
        self._changed()

    def _busy_editing(self):
        return bool(self._detection_worker and self._detection_worker.isRunning()
                    or self._export_worker and self._export_worker.isRunning())

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
        if outcome == "Replay / no point":
            winner = ""
        self._set_selected(winner=winner, outcome=outcome, player=self.player_edit.text().strip(),
                           note=self.note_edit.text().strip(), reviewed=True)

    def _load_point_panel(self):
        rally = self._selected()
        if not hasattr(self, "winner_combo"):
            return
        for widget in (self.winner_combo, self.outcome_combo, self.player_edit, self.note_edit,
                       self.star_button, self.reviewed_button):
            widget.setEnabled(rally is not None and not self._busy_editing())
        if rally:
            self.winner_combo.setCurrentIndex(max(0, self.winner_combo.findData(rally.winner)))
            self.outcome_combo.setCurrentText(rally.outcome)
            self.player_edit.setText(rally.player)
            self.note_edit.setText(rally.note)
            self.star_button.setText("★ Starred" if rally.starred else "☆ Star highlight")
            self.reviewed_button.setText("✓ Reviewed" if rally.reviewed else "Mark reviewed")

    def edit_match(self):
        if self._busy_editing():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Teams and Initial Score")
        layout = QFormLayout(dialog)
        controls = {}
        for key, label in (("team_a", "Team A"), ("team_b", "Team B"),
                            ("initial_score_a", "Initial score A"), ("initial_score_b", "Initial score B")):
            if key.startswith("team"):
                control = QLineEdit(str(self.match_settings[key]))
                control.setMaxLength(100)
            else:
                control = QSpinBox()
                control.setRange(0, 999)
                control.setValue(int(self.match_settings[key]))
            controls[key] = control
            layout.addRow(label, control)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._checkpoint()
            self.match_settings.update({key: (value.text().strip() or ("Team A" if key == "team_a" else "Team B")) if key.startswith("team") else value.value()
                                        for key, value in controls.items()})
            self._update_summary()
            self._changed()

    def _update_match_score(self):
        if not hasattr(self, "match_score_label"):
            return
        a = self.match_settings["initial_score_a"] + sum(r.winner == "A" for r in self.rallies if not r.rejected)
        b = self.match_settings["initial_score_b"] + sum(r.winner == "B" for r in self.rallies if not r.rejected)
        self.match_score_label.setText(f"{self.match_settings['team_a']}  {a} : {b}  {self.match_settings['team_b']}")

    def show_match_statistics(self):
        from collections import Counter
        active = [r for r in self.rallies if not r.rejected]
        outcomes = Counter(r.outcome for r in active if r.outcome)
        players = Counter(r.player for r in active if r.player)
        lines = [self.match_score_label.text(), f"{sum(bool(r.winner) for r in active)} points tagged / {len(active)} candidates", ""]
        lines += [f"{key}: {value}" for key, value in sorted(outcomes.items())]
        if players:
            lines += ["", "Player credits (from manual tags)"] + [f"{key}: {value}" for key, value in sorted(players.items())]
        QMessageBox.information(self, "Match Statistics", "\n".join(lines))

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
        self.undo_action.setEnabled(not busy and bool(self._history.undo_stack))
        self.redo_action.setEnabled(not busy and bool(self._history.redo_stack))
        self.restore_action.setEnabled(not busy and self._selected() is not None)
        for action in (self.open_project_action, self.import_decisions_action):
            action.setEnabled(not busy and not (self._export_worker and self._export_worker.isRunning()))
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
