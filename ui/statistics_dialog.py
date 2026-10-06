"""Player cards, original-RPR breakdown, audit trail, and portable exports."""

from collections import Counter
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox,
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout)

from models.match_flow import CLASSIFICATIONS, match_timeline
from models.statistics import COUNTERS, calculate_statistics, export_statistics
from video.presentation import render_player_end_card
from .video_player import format_timestamp


def percent(value):
    return "Unknown" if value is None else f"{100*value:.1f}%"


def decimal(value):
    return "Unknown" if value is None else f"{value:.1f}"


CARD_HEADERS = ("Player", "Serve %", "Aces : aced", "Put-away %", "Gets", "Strong : weak sets", "Errors", "Ace : rim", "Breaks : broken")

# A clip outcome identifies its winner and some categorical facts. It cannot
# establish the number or quality of the touches that happened within a point.
TOUCH_DETAIL_COUNTERS = frozenset({
    "serve_attempts", "serves_in", "faults", "rims", "lets", "receives",
    "strong_receives", "weak_receives", "receive_errors", "sets",
    "strong_sets", "weak_sets", "set_errors", "hit_attempts", "put_aways",
    "hits_returned", "hit_errors", "defensive_touches", "defensive_gets",
    "defensive_not_returned", "no_touches", "tough_touches",
})
TOUCH_DETAIL_RATES = frozenset({
    "serve_pct", "ace_pct", "put_away_pct", "receive_pct",
    "strong_set_pct", "defensive_conversion_pct", "error_pct",
})
OUTCOME_DEPENDENT_COUNTERS = frozenset({
    "points_won", "aces", "aced", "faults", "double_faults", "errors",
    "breaks", "broken", "break_opportunities", "sideouts", "sideout_opportunities",
})


def _rally_field(rally, key, default=None):
    return rally.get(key, default) if isinstance(rally, dict) else getattr(rally, key, default)


def _touch_details_complete(report):
    coverage = report["coverage"]
    return bool(coverage["points"] and coverage["complete"] == coverage["points"]
                and not coverage["invalid"])


def _count(value, *, known):
    if known:
        return str(value)
    return f"{value} recorded; total unknown" if value else "Unknown"


def card_values(p, *, touch_details_complete=True, outcomes_complete=True):
    return (p["name"], percent(p["serve_pct"] if touch_details_complete else None),
            f"{p['aces']} : {p['aced']}" if outcomes_complete else "Unknown",
            percent(p["put_away_pct"] if touch_details_complete else None),
            _count(p["defensive_gets"], known=touch_details_complete),
            (f"{p['strong_sets']} : {p['weak_sets']}" if touch_details_complete else "Unknown"),
            str(p["errors"]) if outcomes_complete else "Unknown",
            f"{p['aces']} : {_count(p['rims'], known=touch_details_complete)}" if outcomes_complete else "Unknown",
            (f"{p['breaks']} : {p['broken']}" if outcomes_complete else "Unknown"))


def render_stat_card(path, report):
    """Save the same final-score card that the video exporter appends."""
    teams = report["teams"]
    summary = {"team_a": teams["A"]["name"], "team_b": teams["B"]["name"],
               "score_a": teams["A"]["score"], "score_b": teams["B"]["score"],
               "untagged_points": report["coverage"].get("unresolved", 0),
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
        self.complete = QCheckBox("Confirm this is the entire match from 0–0 for RPR")
        self.complete.setToolTip("Score, classified outcomes, and their known player counts update without this confirmation. "
                                 "RPR also needs complete touch details for every point.")
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

    def _outcome_label(self, point, assignment, names):
        kind = assignment.get("kind", "")
        if not kind:
            return point["outcome"] or "Unclassified"
        label = CLASSIFICATIONS[kind]["label"]
        if kind == "error":
            source = self.rallies[assignment["index"]]
            classification = (source.get("classification", {}) if isinstance(source, dict)
                              else source.classification)
            return f"{label} ({names.get(classification['player_id'], classification['player_id'])})"
        return label

    def refresh(self, *_):
        self.settings["stats_complete"] = self.complete.isChecked()
        self.report = calculate_statistics(self.rallies, self.settings)
        report, players = self.report, self.report["players"]
        c = report["coverage"]
        a, b = report["teams"]["A"], report["teams"]["B"]
        touch_details_complete = _touch_details_complete(report)
        timeline = match_timeline(self.rallies, self.settings)
        timeline_by_id = {row["rally_id"]: row for row in timeline}
        details_by_id = {point["rally_id"]: point for point in report["points"]}
        outcomes_complete = all(
            _rally_field(self.rallies[row["index"]], "rejected", False)
            or row["kind"] == "redo"
            or (not row["kind"] and _rally_field(self.rallies[row["index"]], "outcome", "") == "Replay / no point")
            or (not row["provisional"] and bool(row["winner"])
                and (bool(row["kind"]) or (
                    bool(details_by_id.get(row["rally_id"], {}).get("point_stats", {}).get("complete"))
                    and not details_by_id.get(row["rally_id"], {}).get("issues"))))
            for row in timeline)
        classified = sum(bool(row["kind"]) and not _rally_field(self.rallies[row["index"]], "rejected", False)
                         for row in timeline)
        self.coverage.setText(f"{a['name']}  {a['score']} : {b['score']}  {b['name']} (recorded winners + initial score)\n"
                             f"{classified} classified clips · {c['complete']}/{c['points']} scored points with complete "
                             f"touch details · {c['partial']} touch drafts · {c['untagged']} without touch details · "
                             f"{c['invalid']} invalid touch logs · {c['replays']} redos/replays. "
                             "Known outcome counts update immediately. Touch-based totals and rates show Unknown until "
                             "every point has valid touch details. "
                             + ("Unresolved clips make later roles and final totals provisional. "
                                if report["provisional"] or c["unresolved"] else
                                "Some older clips lack enough detail for every player total. "
                                if not outcomes_complete else "")
                             + ("RPR can be calculated for players with serve and hit attempts." if report["rpr_eligible"] else
                                "RPR needs complete touch logs, a 0–0 initial score, and match confirmation."))
        tab_index = self.tabs.currentIndex()
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            widget.deleteLater()
        self._table("Player cards", CARD_HEADERS,
                    [card_values(p, touch_details_complete=touch_details_complete,
                                 outcomes_complete=outcomes_complete) for p in players])
        rates = ("serve_pct", "ace_pct", "put_away_pct", "receive_pct", "strong_set_pct", "defensive_conversion_pct", "error_pct", "break_pct")
        details = [(key.replace("_", " ").title(),
                    *[_count(p[key], known=(touch_details_complete or key not in TOUCH_DETAIL_COUNTERS)
                             and (outcomes_complete or key not in OUTCOME_DEPENDENT_COUNTERS))
                      for p in players]) for key in COUNTERS]
        details += [(key.replace("_pct", " %").replace("_", " ").title(),
                     *[percent(p[key] if (touch_details_complete or key not in TOUCH_DETAIL_RATES)
                               and (outcomes_complete or key != "break_pct") else None)
                       for p in players]) for key in rates]
        self._table("All counts / rates", ("Statistic", *[p["name"] for p in players]), details)
        self._table("Original RPR", ("Player", "Hitting", "Serving", "Defense", "Efficiency", "Overall"),
                    [(p["name"], *[decimal(p["rpr"][key]) for key in ("hitting", "serving", "defense", "efficiency", "overall")]) for p in players])
        self._table("Teams", ("Team", "Points won", "Breaks", "Broken", "Break %", "Sideouts", "Sideout %"),
                    [(t["name"], t["points_won"], _count(t["breaks"], known=outcomes_complete),
                      _count(t["broken"], known=outcomes_complete),
                      percent(t["break_pct"] if outcomes_complete else None),
                      _count(t["sideouts"], known=outcomes_complete),
                      percent(t["sideout_pct"] if outcomes_complete else None))
                     for t in report["teams"].values()])
        names = {p["player_id"]: p["name"] for p in players}
        def role_name(point, key):
            assignment = timeline_by_id.get(point["rally_id"], {})
            if assignment.get("provisional"):
                return "Provisional"
            return names.get(assignment.get(key) or point["point_stats"].get(key), "—")
        aces = {(a["server_id"], a["receiver_id"]): a["aces"] for a in report["aces_by_opponent"]}
        self._table("Who aced whom", ("Server → receiver", *[p["name"] for p in players]),
                    [(p["name"], *["—" if p["team"] == q["team"] else
                                    "Unknown" if not outcomes_complete else
                                    aces.get((p["player_id"], q["player_id"]), 0) for q in players]) for p in players])
        log = self._table("Point log", ("Start", "Outcome", "Winner", "Server", "Receiver", "Events", "Status / issues"),
                         [(format_timestamp(p["start"]),
                           self._outcome_label(p, timeline_by_id.get(p["rally_id"], {}), names),
                           p["winner"] or "?",
                           role_name(p, "server_id"), role_name(p, "receiver_id"),
                           len(p["point_stats"].get("events", [])),
                           "; ".join(p["issues"]) or ("Touch details complete" if p["point_stats"].get("complete")
                                                       else "Outcome known; touch details unknown"
                                                       if timeline_by_id.get(p["rally_id"], {}).get("kind")
                                                       else "Touch draft / unclassified")) for p in report["points"]])
        log.cellDoubleClicked.connect(self._open_point)
        outcomes = Counter((row["kind"], row["winner"]) for row in timeline
                           if row["kind"] and not _rally_field(self.rallies[row["index"]], "rejected", False))
        outcome_rows = [(spec["label"], outcomes[kind, "A"], outcomes[kind, "B"],
                         sum(count for (saved_kind, _), count in outcomes.items() if saved_kind == kind))
                        for kind, spec in CLASSIFICATIONS.items()]
        outcome_rows += [(f"Legacy {label}: {tag}", "—", "—", count)
                         for key, label in (("outcome_tags", "outcome tag"), ("player_tags", "player credit"))
                         for tag, count in sorted(report[key].items())]
        self._table("Clip outcomes", ("Classification / legacy tag", "A wins", "B wins", "Clips"), outcome_rows)
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
