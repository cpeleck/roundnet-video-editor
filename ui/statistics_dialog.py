"""Player cards, original-RPR breakdown, audit trail, and portable exports."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox,
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout)

from models.statistics import COUNTERS, calculate_statistics, export_statistics
from video.presentation import render_player_end_card
from .video_player import format_timestamp


def percent(value):
    return "—" if value is None else f"{100*value:.1f}%"


def decimal(value):
    return "—" if value is None else f"{value:.1f}"


CARD_HEADERS = ("Player", "Serve %", "Aces : aced", "Put-away %", "Gets", "Strong : weak sets", "Errors", "Ace : rim", "Breaks : broken")


def card_values(p):
    return (p["name"], percent(p["serve_pct"]), f"{p['aces']} : {p['aced']}", percent(p["put_away_pct"]),
            str(p["defensive_gets"]), f"{p['strong_sets']} : {p['weak_sets']}", str(p["errors"]),
            f"{p['aces']} : {p['rims']}", f"{p['breaks']} : {p['broken']}")


def render_stat_card(path, report):
    """Save the same final-score card that the video exporter appends."""
    teams = report["teams"]
    summary = {"team_a": teams["A"]["name"], "team_b": teams["B"]["name"],
               "score_a": teams["A"]["score"], "score_b": teams["B"]["score"],
               "player_statistics": report}
    render_player_end_card(Path(path), 1800, 850, summary)


class StatisticsDialog(QDialog):
    point_requested = Signal(str, float)

    def __init__(self, rallies, settings, parent=None, *, protected_paths=()):
        super().__init__(parent)
        self.setWindowTitle("Match and player statistics")
        self.resize(1180, 700)
        self.rallies, self.settings = rallies, dict(settings)
        self.protected_paths = protected_paths
        self.report = calculate_statistics(rallies, settings)
        layout = QVBoxLayout(self)
        self.coverage = QLabel()
        self.coverage.setWordWrap(True)
        layout.addWidget(self.coverage)
        self.complete = QCheckBox("This is the entire match from 0–0, and every point is fully logged")
        self.complete.setChecked(bool(settings.get("stats_complete", False)))
        self.complete.toggled.connect(self.refresh)
        layout.addWidget(self.complete)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        actions = QHBoxLayout()
        for title, suffix in (("Export player CSV…", ".csv"), ("Export complete stats JSON…", ".json"), ("Save stat card PNG…", ".png")):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, ext=suffix: self._export(ext))
            actions.addWidget(button)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)
        self.refresh()

    def _table(self, title, headers, rows):
        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                table.setItem(r, c, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(table, title)
        return table

    def refresh(self, *_):
        self.settings["stats_complete"] = self.complete.isChecked()
        self.report = calculate_statistics(self.rallies, self.settings)
        report, players = self.report, self.report["players"]
        c = report["coverage"]
        a, b = report["teams"]["A"], report["teams"]["B"]
        self.coverage.setText(f"{a['name']}  {a['score']} : {b['score']}  {b['name']} (known winners + initial score)\n"
                             f"{c['complete']}/{c['points']} points fully logged · {c['partial']} drafts · "
                             f"{c['untagged']} untagged · {c['invalid']} invalid · {c['replays']} replays excluded. "
                             "Unchecked valid rallies count; rejected detections do not. "
                             + ("Original RPR enabled (players need a serve and hit attempt)." if report["rpr_eligible"] else
                                "RPR needs a complete match, all points/qualities recorded, a 0–0 initial score, and serve/hit attempts."))
        tab_index = self.tabs.currentIndex()
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            widget.deleteLater()
        self._table("Player cards", CARD_HEADERS, [card_values(p) for p in players])
        rates = ("serve_pct", "ace_pct", "put_away_pct", "receive_pct", "strong_set_pct", "defensive_conversion_pct", "error_pct", "break_pct")
        details = [(key.replace("_", " ").title(), *[p[key] for p in players]) for key in COUNTERS]
        details += [(key.replace("_pct", " %").replace("_", " ").title(), *[percent(p[key]) for p in players]) for key in rates]
        self._table("All counts / rates", ("Statistic", *[p["name"] for p in players]), details)
        self._table("Original RPR", ("Player", "Hitting", "Serving", "Defense", "Efficiency", "Overall"),
                    [(p["name"], *[decimal(p["rpr"][key]) for key in ("hitting", "serving", "defense", "efficiency", "overall")]) for p in players])
        self._table("Teams", ("Team", "Points won", "Breaks", "Broken", "Break %", "Sideouts", "Sideout %"),
                    [(t["name"], t["points_won"], t["breaks"], t["broken"], percent(t["break_pct"]), t["sideouts"], percent(t["sideout_pct"])) for t in report["teams"].values()])
        names = {p["player_id"]: p["name"] for p in players}
        aces = {(a["server_id"], a["receiver_id"]): a["aces"] for a in report["aces_by_opponent"]}
        self._table("Who aced whom", ("Server → receiver", *[p["name"] for p in players]),
                    [(p["name"], *["—" if p["team"] == q["team"] else aces.get((p["player_id"], q["player_id"]), 0) for q in players]) for p in players])
        log = self._table("Point log", ("Start", "Winner", "Server", "Receiver", "Events", "Status / issues"),
                         [(format_timestamp(p["start"]), p["winner"] or "?", names.get(p["point_stats"].get("server_id"), "—"),
                           names.get(p["point_stats"].get("receiver_id"), "—"), len(p["point_stats"].get("events", [])),
                           "; ".join(p["issues"]) or ("Complete" if p["point_stats"].get("complete") else "Draft / untagged")) for p in report["points"]])
        log.cellDoubleClicked.connect(self._open_point)
        self._table("Quick tags", ("Manual tag type (separate from touch stats)", "Tag", "Count"),
                    [(label, tag, count) for key, label in (("outcome_tags", "Outcome"), ("player_tags", "Player credit"))
                     for tag, count in sorted(report[key].items())])
        help_view = QTextBrowser()
        help_view.setOpenExternalLinks(True)
        help_view.setMarkdown((Path(__file__).resolve().parents[1] / "docs" / "statistics.md").read_text(encoding="utf-8"))
        self.tabs.addTab(help_view, "Definitions / formulas")
        self.tabs.setCurrentIndex(max(0, min(tab_index, self.tabs.count()-1)))

    def _open_point(self, row, _):
        point = self.report["points"][row]
        self.point_requested.emit(point["rally_id"], point["start"])
        self.accept()

    def _export(self, suffix):
        path, _ = QFileDialog.getSaveFileName(self, "Save Statistics", f"match-statistics{suffix}", f"Statistics (*{suffix})")
        if not path:
            return
        if Path(path).suffix.lower() != suffix:
            path += suffix
        try:
            if Path(path).resolve() in {Path(p).resolve() for p in self.protected_paths if p}:
                raise ValueError("Choose a filename other than the source or project")
            if suffix == ".png":
                render_stat_card(path, self.report)
            else:
                export_statistics(path, self.report, protected_paths=self.protected_paths)
        except Exception as exc:
            QMessageBox.warning(self, "Statistics Export Failed", str(exc))
