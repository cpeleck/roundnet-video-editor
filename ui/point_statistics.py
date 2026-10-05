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
from models.statistics import point_issues, resolve_events
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
        self.roster = normalize_roster(settings.get("players"))
        self.names = {p["player_id"]: p["name"] for p in self.roster}
        self.point = normalize_point_stats(rally.point_stats) or {
            "version": 1, "server_id": "", "receiver_id": "", "complete": False, "events": []}
        self.events = deepcopy(self.point["events"])
        self.winner = rally.winner
        self.replay = rally.outcome == "Replay / no point"
        self._loading = False
        root = QVBoxLayout(self)
        intro = QLabel("Record every serve attempt, then each receive, set, hit, and defensive touch. "
                       "Auto derives put-aways and defensive gets from the sequence; quality is your judgment. "
                       "Drafts keep unknown results unknown. Player names are set in Teams / players.")
        intro.setWordWrap(True)
        root.addWidget(intro)
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
        meta.addRow(self.replay_box)
        right.addLayout(meta)
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
        if issues:
            self.status.setText("Needs correction: " + " ".join(issues[:3]))
        else:
            inferred = [f"{self.names[e['player_id']]}: {LABELS.get(e['result'], e['result'])}" for e, original in zip(resolved, self.events) if original["result"] == "auto"]
            self.status.setText(("Complete point. " if point["complete"] else "Draft — only known events count. ") + " · ".join(inferred))

    def _save(self):
        point = self._current_point()
        self.replay = self.replay_box.isChecked()
        self.winner = "" if self.replay else self.winner_combo.currentData()
        issues = point_issues(point, self.winner, start=self.rally.start_time, end=self.rally.end_time)
        if issues and not self.replay:
            QMessageBox.warning(self, "Check the touch sequence", "\n".join(issues))
            return
        self.point = normalize_point_stats(point)
        self.accept()
