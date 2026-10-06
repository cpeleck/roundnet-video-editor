"""Video-assisted, ordered touch logging for one rally."""

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from models.point_stats import RESULTS, new_event, normalize_point_stats, normalize_roster
from models.statistics import classified_point_issues, point_issues, resolve_events
from .video_player import VideoPlayerWidget, format_timestamp


LABELS = {"in": "In", "ace": "Ace", "fault": "Fault", "rim": "Rim fault", "let": "Let (not an attempt)",
          "auto": "Auto — derive from later touches", "strong": "Strong", "weak": "Weak", "error": "Error",
          "unknown": "Ungraded / unknown", "put_away": "Put-away", "returned": "Returned to net",
          "get": "Get — returned to net", "touch_not_returned": "Touch — not returned", "no_touch": "No touch"}


class PointStatisticsDialog(QDialog):
    def __init__(self, video_path, rally, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Point statistics · touch by touch")
        self.resize(1220, 810)
        self.rally = rally
        self.classification = getattr(rally, "classification", {})
        self.expected_server = settings.get("starting_server", "A1")
        self.expected_receiver = settings.get("starting_receiver", "B1")
        self.roster = normalize_roster(settings.get("players"))
        self.names = {p["player_id"]: p["name"] for p in self.roster}
        self.point = normalize_point_stats(rally.point_stats) or {
            "version": 1, "server_id": settings.get("starting_server", "A1"),
            "receiver_id": settings.get("starting_receiver", "B1"), "complete": False, "events": []}
        self.events = deepcopy(self.point["events"])
        self.quick_history = []
        self.winner = rally.winner
        self.replay = rally.outcome == "Replay / no point"
        other_points = [item for item in getattr(parent, "rallies", [])
                        if item.rally_id != rally.rally_id and item.start_time < rally.start_time and not item.rejected
                        and item.outcome != "Replay / no point"]
        self.base_score_a = settings.get("initial_score_a", 0) + sum(item.winner == "A" for item in other_points)
        self.base_score_b = settings.get("initial_score_b", 0) + sum(item.winner == "B" for item in other_points)
        self.team_a = settings.get("team_a", "Team A")
        self.team_b = settings.get("team_b", "Team B")
        self._loading = False
        root = QVBoxLayout(self)
        intro = QLabel("Tap the four players in touch order. Ace, Fault, and Error finish a point; "
                       "Point Won finishes a winning hit. The score and player counts are saved together. "
                       "Successful receives and sets default to Strong; revise any different touch below.")
        if self.classification:
            intro.setText("Add optional touch details for percentages and RPR. The clip outcome already sets the score and known player statistics. "
                          "To change the outcome, use its button in the main editor.")
        intro.setWordWrap(True)
        root.addWidget(intro)
        if self.classification and (self.point["server_id"] != self.expected_server
                                    or self.point["receiver_id"] != self.expected_receiver):
            stale = QLabel("Earlier clip edits changed this point's server or receiver. The saved touches need review.")
            stale.setWordWrap(True)
            root.addWidget(stale)
            reset_details = QPushButton("Clear old touches and use this clip's serving assignment")
            reset_details.clicked.connect(self._reset_stale_details)
            root.addWidget(reset_details)
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.video = VideoPlayerWidget()
        self.video.audio_output.setVolume(.4)
        self._seek_pending = True
        self.video.player.mediaStatusChanged.connect(self._media_ready)
        self.video.load(video_path)
        left.addWidget(self.video, 1)
        loop = QPushButton("Play / loop this point")
        loop.clicked.connect(lambda: self.video.preview_range(rally.start_time, rally.end_time, loop=True))
        left.addWidget(loop)
        body.addLayout(left, 1)
        right = QVBoxLayout()
        meta = QFormLayout()
        self.server = self._player_combo(blank=True)
        self.receiver = self._player_combo(blank=True)
        self.server.setCurrentIndex(max(0, self.server.findData(self.point["server_id"])))
        self.receiver.setCurrentIndex(max(0, self.receiver.findData(self.point["receiver_id"])))
        self.winner_combo = QComboBox()
        self.winner_combo.addItem("Unknown / not decided", "")
        self.winner_combo.addItem(settings.get("team_a", "Team A"), "A")
        self.winner_combo.addItem(settings.get("team_b", "Team B"), "B")
        self.winner_combo.setCurrentIndex(max(0, self.winner_combo.findData(rally.winner)))
        meta.addRow("Server", self.server)
        meta.addRow("Receiver", self.receiver)
        meta.addRow("Point winner", self.winner_combo)
        self.replay_box = QCheckBox("Replay / no point — exclude from statistics")
        self.replay_box.setChecked(self.replay)
        if self.classification:
            self.replay_box.setEnabled(False)
        meta.addRow(self.replay_box)
        right.addLayout(meta)
        self.score_preview = QLabel()
        self.score_preview.setStyleSheet("font-weight: bold; color: #79c0ff")
        right.addWidget(self.score_preview)
        guide = QLabel("1  Tap server and receiver   →   2  Tap each next touch   →   3  Finish the point")
        guide.setWordWrap(True)
        right.addWidget(guide)
        player_row = QHBoxLayout()
        self.player_buttons = {}
        for player in self.roster:
            pid = player["player_id"]
            button = QPushButton(f"{player['name']}\n{pid}")
            button.setToolTip("Append this player's next touch in sequence")
            button.clicked.connect(lambda checked=False, identity=pid: self.quick_touch(identity))
            self.player_buttons[pid] = button
            player_row.addWidget(button)
        right.addLayout(player_row)
        finish_row = QHBoxLayout()
        for title, action in (("Ace", self.quick_ace), ("Fault", self.quick_fault),
                              ("Error", self.quick_error), ("Point Won", self.quick_point_won),
                              ("Undo touch", self.quick_undo)):
            button = QPushButton(title)
            button.clicked.connect(action)
            finish_row.addWidget(button)
        right.addLayout(finish_row)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(("#", "Player", "Touch", "Result", "Time"))
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(self._seek_event)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(180)
        right.addWidget(self.table, 1)
        controls = QFormLayout()
        self.actor = self._player_combo()
        self.kind = QComboBox()
        for kind in RESULTS:
            self.kind.addItem(kind.title(), kind)
        self.result_combo = QComboBox()
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.result_combo.currentIndexChanged.connect(self._result_changed)
        self._kind_changed()
        controls.addRow("Player", self.actor)
        controls.addRow("Touch", self.kind)
        controls.addRow("Result", self.result_combo)
        self.tough = QCheckBox("Costly tough touch (indirectly caused the lost point)")
        self.tough.setToolTip("Only for weak receives/sets. Weak alone does not apply the RPR tough-touch penalty.")
        self.timestamp = QCheckBox("Attach current video time (optional)")
        controls.addRow(self.tough)
        controls.addRow(self.timestamp)
        right.addLayout(controls)
        actions = QHBoxLayout()
        for title, action in (("Add touch", self.add_event), ("Update", self.update_event),
                              ("Remove", self.remove_event), ("↑", lambda: self.move_event(-1)),
                              ("↓", lambda: self.move_event(1))):
            button = QPushButton(title)
            button.clicked.connect(action)
            actions.addWidget(button)
        right.addLayout(actions)
        self.complete = QCheckBox("All attempts and touches for this point are recorded")
        self.complete.setChecked(self.point["complete"])
        right.addWidget(self.complete)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(60)
        right.addWidget(self.status)
        body.addLayout(right, 1)
        root.addLayout(body, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.finished.connect(lambda *_: self.video.unload())
        for combo in (self.server, self.receiver, self.winner_combo):
            combo.currentIndexChanged.connect(self.refresh_status)
        self.server.currentIndexChanged.connect(self._suggest_actor)
        self.receiver.currentIndexChanged.connect(self._suggest_actor)
        self.complete.toggled.connect(self.refresh_status)
        self.replay_box.toggled.connect(self.refresh_status)
        self.refresh()
        self._result_changed()

    def _media_ready(self, status):
        if self._seek_pending and status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            self._seek_pending = False
            self.video.set_position(self.rally.start_time)

    def _player_combo(self, blank=False):
        combo = QComboBox()
        if blank:
            combo.addItem("Not recorded", "")
        for p in self.roster:
            combo.addItem(f"{p['name']} · {p['player_id']}", p["player_id"])
        return combo

    def _kind_changed(self, *_):
        self.result_combo.clear()
        for result in RESULTS[self.kind.currentData()]:
            self.result_combo.addItem(LABELS[result], result)
        if not self._loading:
            preferred = self.server.currentData() if self.kind.currentData() == "serve" else self.receiver.currentData() if self.kind.currentData() == "receive" else None
            if preferred:
                self.actor.setCurrentIndex(self.actor.findData(preferred))
        self._result_changed()

    def _suggest_actor(self, *_):
        if self._loading or self.table.currentRow() >= 0:
            return
        preferred = self.server.currentData() if self.kind.currentData() == "serve" else self.receiver.currentData() if self.kind.currentData() == "receive" else None
        if preferred:
            self.actor.setCurrentIndex(self.actor.findData(preferred))

    def _result_changed(self, *_):
        if hasattr(self, "tough"):
            allowed = self.kind.currentData() in ("receive", "set") and self.result_combo.currentData() == "weak"
            self.tough.setEnabled(allowed)
            if not allowed:
                self.tough.setChecked(False)

    def _current_point(self):
        return {"version": 1, "server_id": self.server.currentData(), "receiver_id": self.receiver.currentData(),
                "complete": self.complete.isChecked() and not self.replay_box.isChecked(), "events": deepcopy(self.events)}

    def _control_event(self):
        return new_event(self.actor.currentData(), self.kind.currentData(), self.result_combo.currentData(),
                         time=round(self.video.position_seconds, 3) if self.timestamp.isChecked() else None,
                         tough=self.tough.isChecked())

    def _quick_snapshot(self):
        self.quick_history.append((deepcopy(self.events), self.server.currentData(), self.receiver.currentData(),
                                   self.winner_combo.currentData(), self.complete.isChecked()))

    def _quick_restore(self, snapshot):
        events, server, receiver, winner, complete = snapshot
        self.events = events
        self.server.setCurrentIndex(self.server.findData(server))
        self.receiver.setCurrentIndex(self.receiver.findData(receiver))
        self.winner_combo.setCurrentIndex(self.winner_combo.findData(winner))
        self.complete.setChecked(complete)
        self.refresh()

    def quick_undo(self):
        if self.quick_history:
            self._quick_restore(self.quick_history.pop())

    def quick_touch(self, player_id):
        """Add the usual next role; detailed controls can refine unusual plays."""
        if self.complete.isChecked() and self.events and not (
                self.events[-1]["kind"] == "serve" and self.events[-1]["result"] in ("fault", "rim")):
            self.status.setText("This point is complete. Undo the finish before adding a touch.")
            return
        if self.events and self.events[-1]["result"] in ("ace", "error", "no_touch"):
            self.status.setText("This point has ended. Undo the finish before adding a touch.")
            return
        self._quick_snapshot()
        if self.complete.isChecked():
            self.winner_combo.setCurrentIndex(self.winner_combo.findData(""))
        if not self.events or all(e["kind"] == "serve" and e["result"] in ("fault", "rim", "let") for e in self.events):
            kind, result = "serve", "in"
            self.server.setCurrentIndex(self.server.findData(player_id))
            if self.receiver.currentData() and self.receiver.currentData()[0] == player_id[0]:
                opponent = next(p["player_id"] for p in self.roster if p["team"] != player_id[0])
                self.receiver.setCurrentIndex(self.receiver.findData(opponent))
        else:
            last = self.events[-1]
            if last["kind"] == "serve":
                kind, result = "receive", "strong"
                self.receiver.setCurrentIndex(self.receiver.findData(player_id))
            elif last["kind"] == "hit":
                kind, result = "defense", "auto"
            elif last["kind"] in ("receive", "defense"):
                kind, result = "set", "strong"
            else:
                kind, result = "hit", "auto"
        self.events.append(new_event(player_id, kind, result))
        self.complete.setChecked(False)
        self.refresh()

    def _quick_finish(self, result):
        if not self.events:
            self.quick_touch(self.server.currentData())
            self.quick_history.pop()
        last = self.events[-1]
        if result in ("ace", "fault") and last["kind"] != "serve":
            self.status.setText("Ace and Fault apply to the last serve. Undo or select that serve first.")
            return
        if result == "error" and last["kind"] == "serve":
            self.status.setText("Use Fault for a serve error, or tap the player who made the next touch.")
            return
        self._quick_snapshot()
        if result == "error" and last["kind"] == "defense":
            result = "touch_not_returned"
        last["result"] = result
        winner = last["player_id"][0] if result == "ace" else ("B" if last["player_id"][0] == "A" else "A")
        self.winner_combo.setCurrentIndex(self.winner_combo.findData(winner))
        self.complete.setChecked(True)
        self.refresh()
        issues = point_issues(self._current_point(), winner, start=self.rally.start_time, end=self.rally.end_time)
        if issues:
            self.status.setText("Needs correction before saving: " + " ".join(issues[:2]))
        else:
            self.status.setText(f"{self.names[last['player_id']]}: {LABELS.get(result, result)}. Team {winner} gains one point; "
                                "the player event and score save together. Undo touch reverses this action.")

    def quick_ace(self):
        self._quick_finish("ace")

    def quick_fault(self):
        self._quick_finish("fault")

    def quick_error(self):
        self._quick_finish("error")

    def quick_point_won(self):
        if not self.events:
            self.status.setText("Tap the players in touch order before awarding a point.")
            return
        last = self.events[-1]
        if last["kind"] == "serve":
            self.quick_ace()
            return
        if last["kind"] != "hit":
            self.status.setText("Tap the winning hitter, or use Error for the last losing touch.")
            return
        self._quick_snapshot()
        winner = last["player_id"][0]
        self.winner_combo.setCurrentIndex(self.winner_combo.findData(winner))
        self.complete.setChecked(True)
        self.refresh()
        issues = point_issues(self._current_point(), winner, start=self.rally.start_time, end=self.rally.end_time)
        self.status.setText("Needs correction before saving: " + " ".join(issues[:2]) if issues else
                            f"Team {winner} gains one point. {self.names[last['player_id']]} gets a put-away "
                            "if the hit was not returned; the score and stats save together.")

    def add_event(self):
        event = self._control_event()
        self.events.append(event)
        if event["kind"] == "serve" and not self.server.currentData():
            self.server.setCurrentIndex(self.server.findData(event["player_id"]))
        if event["kind"] == "receive" and not self.receiver.currentData():
            self.receiver.setCurrentIndex(self.receiver.findData(event["player_id"]))
        self.complete.setChecked(False)
        self.refresh()
        kind = event["kind"]
        next_kind = {"receive": "set", "set": "hit", "hit": "defense", "defense": "set"}.get(kind)
        if kind == "serve" and event["result"] == "in":
            next_kind = "receive"
        if next_kind:
            self.kind.setCurrentIndex(self.kind.findData(next_kind))
            if next_kind in ("set", "hit"):
                partner = event["player_id"][0] + ("2" if event["player_id"][1] == "1" else "1")
                self.actor.setCurrentIndex(self.actor.findData(partner))
            elif next_kind == "defense":
                self.actor.setCurrentIndex(self.actor.findData("B1" if event["player_id"][0] == "A" else "A1"))

    def update_event(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.events):
            event = self._control_event()
            event["event_id"] = self.events[row]["event_id"]
            self.events[row] = event
            self.complete.setChecked(False)
            self.refresh(row)

    def remove_event(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.events):
            self.events.pop(row)
            self.complete.setChecked(False)
            self.refresh(min(row, len(self.events)-1))

    def move_event(self, direction):
        row = self.table.currentRow()
        target = row + direction
        if 0 <= row < len(self.events) and 0 <= target < len(self.events):
            self.events[row], self.events[target] = self.events[target], self.events[row]
            self.complete.setChecked(False)
            self.refresh(target)

    def _selection_changed(self):
        row = self.table.currentRow()
        if self._loading or not 0 <= row < len(self.events):
            return
        event = self.events[row]
        self._loading = True
        self.kind.setCurrentIndex(self.kind.findData(event["kind"]))
        self.result_combo.setCurrentIndex(self.result_combo.findData(event["result"]))
        self.actor.setCurrentIndex(self.actor.findData(event["player_id"]))
        self.tough.setChecked(event.get("tough", False))
        self.timestamp.setChecked(event.get("time") is not None)
        self._loading = False
        if event.get("time") is not None:
            self.video.set_position(event["time"])

    def _seek_event(self, row, _):
        timestamp = self.events[row].get("time")
        if timestamp is not None:
            self.video.set_position(timestamp)

    def refresh(self, selected=-1):
        self._loading = True
        self.table.setRowCount(len(self.events))
        for row, event in enumerate(self.events):
            values = (str(row+1), self.names[event["player_id"]], event["kind"].title(),
                      LABELS[event["result"]] + (" · tough" if event.get("tough") else ""),
                      format_timestamp(event["time"]) if event.get("time") is not None else "—")
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(3, 160)
        self.table.clearSelection()
        self.table.setCurrentCell(-1, -1)
        self._loading = False
        if selected >= 0:
            self.table.selectRow(selected)
        self.refresh_status()

    def refresh_status(self, *_):
        point = self._current_point()
        winner = "" if self.replay_box.isChecked() else self.winner_combo.currentData()
        self.score_preview.setText(
            f"Score with this point: {self.team_a} {self.base_score_a + (winner == 'A')} : "
            f"{self.base_score_b + (winner == 'B')} {self.team_b}")
        resolved = resolve_events(point, self.winner_combo.currentData())
        for row, (event, original) in enumerate(zip(resolved, self.events)):
            if original["result"] == "auto" and self.table.item(row, 3):
                label = {"get": "Get", "returned": "Returned", "put_away": "Put-away",
                         "touch_not_returned": "Not returned", "unknown": "Unresolved"}.get(event["result"], event["result"])
                self.table.item(row, 3).setText("Auto → " + label)
        if self.replay_box.isChecked():
            self.status.setText("Replay: stored for reference, excluded from all point statistics.")
            return
        issues = point_issues(point, self.winner_combo.currentData(), start=self.rally.start_time, end=self.rally.end_time)
        if self.classification:
            issues += classified_point_issues(point, self.classification, {
                "server_id": self.expected_server, "receiver_id": self.expected_receiver,
                "provisional": False})
        if issues:
            self.status.setText("Needs correction: " + " ".join(issues[:3]))
        else:
            inferred = [f"{self.names[e['player_id']]}: {LABELS.get(e['result'], e['result'])}" for e, original in zip(resolved, self.events) if original["result"] == "auto"]
            self.status.setText(("Complete point. " if point["complete"] else "Draft — only known events count. ") + " · ".join(inferred))

    def _reset_stale_details(self):
        self.events = []
        self.server.setCurrentIndex(self.server.findData(self.expected_server))
        self.receiver.setCurrentIndex(self.receiver.findData(self.expected_receiver))
        self.complete.setChecked(False)
        self.refresh()

    def _save(self):
        point = self._current_point()
        if self.classification:
            if point["server_id"] != self.expected_server or point["receiver_id"] != self.expected_receiver:
                QMessageBox.warning(self, "Check serving order", "The saved touch log must use the server and receiver shown for this clip. Change an earlier outcome or match setup if that assignment is wrong.")
                return
            self.replay = self.classification["kind"] == "redo"
            self.winner = self.rally.winner
            self.winner_combo.setCurrentIndex(self.winner_combo.findData(self.winner))
        else:
            self.replay = self.replay_box.isChecked()
            self.winner = "" if self.replay else self.winner_combo.currentData()
        issues = point_issues(point, self.winner, start=self.rally.start_time, end=self.rally.end_time)
        if self.classification:
            issues += classified_point_issues(point, self.classification, {
                "server_id": self.expected_server, "receiver_id": self.expected_receiver,
                "provisional": False})
        if issues and not self.replay:
            QMessageBox.warning(self, "Check the touch sequence", "\n".join(issues))
            return
        self.point = normalize_point_stats(point)
        self.accept()
