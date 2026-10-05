"""Main single-window workflow for the Roundnet rally editor."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PySide6.QtCore import QSettings, QSignalBlocker, QTimer, Qt, QUrl
from PySide6.QtGui import (
    QAction,
    QBrush,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .roi_selector import NormalizedROI, ROISelectorDialog, load_last_roi
from .settings_dialog import DEFAULT_SETTINGS, SettingsDialog
from .timeline import RallyTimeline
from .video_player import VideoPlayerWidget, format_timestamp
from .workers import DetectionWorker, ExportWorker
from .project_workflow import ProjectWorkflow


LOGGER = logging.getLogger(__name__)


try:
    from models import Rally as RallyModel
except ImportError:  # permits isolated UI development before the model package exists
    @dataclass
    class RallyModel:  # type: ignore[no-redef]
        start_time: float
        end_time: float
        confidence: float = 1.0
        enabled: bool = True
        serve_confidence: float = 0.0

        @property
        def duration(self) -> float:
            return max(0.0, self.end_time - self.start_time)


def _new_rally(
    start: float,
    end: float,
    confidence: float = 1.0,
    enabled: bool = True,
    serve_confidence: float = 0.0,
) -> Any:
    """Construct the shared model while tolerating positional-only variants."""

    values = (float(start), float(end), float(confidence), bool(enabled))
    try:
        return RallyModel(
            start_time=values[0],
            end_time=values[1],
            confidence=values[2],
            enabled=values[3],
            serve_confidence=float(serve_confidence),
        )
    except TypeError:
        try:
            return RallyModel(
                start_time=values[0],
                end_time=values[1],
                confidence=values[2],
                enabled=values[3],
            )
        except TypeError:
            return RallyModel(*values)


def _rally_values(rally: Any) -> tuple[float, float, float, bool]:
    return (
        float(getattr(rally, "start_time", 0.0)),
        float(getattr(rally, "end_time", 0.0)),
        float(getattr(rally, "confidence", 1.0)),
        bool(getattr(rally, "enabled", True)),
    )


def _serve_confidence(rally: Any) -> float:
    try:
        return max(0.0, min(1.0, float(getattr(rally, "serve_confidence", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _rally_snapshot(rally: Any) -> dict[str, Any]:
    if hasattr(rally, "to_dict"):
        return rally.to_dict()
    start, end, confidence, enabled = _rally_values(rally)
    return {
        "start_time": start,
        "end_time": end,
        "confidence": confidence,
        "serve_confidence": _serve_confidence(rally),
        "enabled": enabled,
    }


def _metadata_value(metadata: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(metadata, Mapping) and name in metadata:
            return metadata[name]
        value = getattr(metadata, name, None)
        if value is not None:
            return value
    return default


class ExportDialog(QDialog):
    """Small export confirmation dialog with output and quality choices."""

    def __init__(
        self,
        suggested_path: str,
        prefer_hardware: bool,
        save_labels: bool,
        rally_count: int,
        duration: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Rally Video")
        self.setMinimumWidth(610)

        summary = QLabel(
            f"Export <b>{rally_count}</b> enabled rallies totaling "
            f"<b>{format_timestamp(duration)}</b>. Original resolution, frame rate, and audio are preserved "
            "where the source codec permits."
        )
        summary.setWordWrap(True)

        self.path_edit = QLineEdit(suggested_path)
        self.path_edit.setMinimumWidth(410)
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self._browse)
        output_row = QHBoxLayout()
        output_row.addWidget(self.path_edit, 1)
        output_row.addWidget(self.browse_button)

        self.hardware_checkbox = QCheckBox("Use hardware acceleration when available")
        self.hardware_checkbox.setChecked(prefer_hardware)
        self.hardware_checkbox.setToolTip("On Apple Silicon this prefers VideoToolbox, with software fallback.")
        self.labels_checkbox = QCheckBox("Save corrected rally labels beside the export")
        self.labels_checkbox.setChecked(save_labels)

        self.aspect_combo = QComboBox()
        for label, ratio in (("Original framing", "source"), ("Landscape · 16:9", "16:9"),
                             ("Portrait · 9:16", "9:16"), ("Square · 1:1", "1:1")):
            self.aspect_combo.addItem(label, ratio)
        self.highlights_checkbox = QCheckBox("Only export starred, enabled rallies")
        self.scoreboard_checkbox = QCheckBox("Show score from manually tagged point winners")
        self.stats_checkbox = QCheckBox("Add a 3-second match statistics card")
        self.notes_checkbox = QCheckBox("Show rally captions")
        self.overlay_edit = QLineEdit()
        self.overlay_edit.setPlaceholderText("Optional transparent PNG logo / branding")
        overlay_button = QPushButton("Choose PNG…")
        overlay_button.clicked.connect(self._choose_overlay)
        overlay_row = QHBoxLayout()
        overlay_row.addWidget(self.overlay_edit, 1)
        overlay_row.addWidget(overlay_button)

        form = QFormLayout()
        form.addRow("Output file", output_row)
        form.addRow("Encoding", self.hardware_checkbox)
        form.addRow("Training data", self.labels_checkbox)
        form.addRow("Framing", self.aspect_combo)
        form.addRow("Highlights", self.highlights_checkbox)
        form.addRow("Scoreboard", self.scoreboard_checkbox)
        form.addRow("Statistics", self.stats_checkbox)
        form.addRow("Captions", self.notes_checkbox)
        form.addRow("Branding", overlay_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        save_button.setText("Export Video")
        save_button.setDefault(True)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addSpacing(8)
        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addWidget(buttons)

    @property
    def output_path(self) -> str:
        path = self.path_edit.text().strip()
        if path and Path(path).suffix.lower() != ".mp4":
            path += ".mp4"
        return path

    @property
    def prefer_hardware(self) -> bool:
        return self.hardware_checkbox.isChecked()

    @property
    def save_labels(self) -> bool:
        return self.labels_checkbox.isChecked()

    @property
    def export_options(self) -> dict[str, Any]:
        return {"aspect_ratio": self.aspect_combo.currentData(),
                "highlights_only": self.highlights_checkbox.isChecked(),
                "scoreboard": self.scoreboard_checkbox.isChecked(),
                "include_stats": self.stats_checkbox.isChecked(),
                "include_notes": self.notes_checkbox.isChecked(),
                "overlay_path": self.overlay_edit.text().strip() or None}

    def _choose_overlay(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose Branding Image", "", "PNG image (*.png)")
        if path:
            self.overlay_edit.setText(path)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Rally Video",
            self.output_path,
            "MP4 Video (*.mp4)",
        )
        if path:
            self.path_edit.setText(path)

    def _validate_and_accept(self) -> None:
        if not self.output_path:
            QMessageBox.information(self, "Choose an Output File", "Choose where to save the rally video.")
            return
        overlay = self.overlay_edit.text().strip()
        if overlay and (not Path(overlay).is_file() or Path(overlay).suffix.lower() != ".png"):
            QMessageBox.warning(self, "Branding Image", "Choose an existing PNG file, or clear the branding field.")
            return
        parent = self.parent()
        if self.highlights_checkbox.isChecked() and parent and not any(
                r.starred for r in parent._enabled_rallies()):
            QMessageBox.information(self, "No Starred Highlights", "Star at least one enabled rally before exporting highlights.")
            return
        output = Path(self.output_path).expanduser()
        if not output.parent.exists():
            QMessageBox.warning(self, "Folder Not Found", f"The folder does not exist:\n{output.parent}")
            return
        if output.exists():
            answer = QMessageBox.question(
                self,
                "Replace Existing File?",
                f"A file already exists at:\n{output}\n\nReplace it with this export?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.accept()


class MainWindow(ProjectWorkflow, QMainWindow):
    """Roundnet Editor's import → detect → correct → export workflow."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Roundnet Rally Editor")
        self.resize(1440, 900)
        self.setMinimumSize(1020, 700)

        self.video_path: str | None = None
        self.video_metadata: Any | None = None
        self.video_duration = 0.0
        self.roi: NormalizedROI | None = load_last_roi()
        self.rallies: list[Any] = []
        self.initial_detected_rallies: list[dict[str, float | bool]] = []
        self.rejected_detections: list[dict[str, Any]] = []
        self.selected_rally_index = -1
        self.detection_result: Any | None = None
        self._detection_worker: DetectionWorker | None = None
        self._export_worker: ExportWorker | None = None
        self._export_progress: QProgressDialog | None = None
        self._export_output_path: str | None = None
        self._save_labels_on_export = False
        self._settings_store = QSettings("RoundnetEditor", "RoundnetEditor")
        self.detection_settings = self._load_detection_settings()
        self._init_workflow()

        self._build_actions()
        self._build_ui()
        self._build_shortcuts()
        self._apply_style()
        self._update_ui_state()
        self._update_summary()
        QTimer.singleShot(0, self._recover_last_project)

    # ------------------------------------------------------------------ setup
    def _build_actions(self) -> None:
        self.open_action = QAction("Open Game Video…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_video)
        self.export_action = QAction("Export Rally Video…", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.triggered.connect(self.show_export_dialog)
        self.labels_action = QAction("Save Correction Labels…", self)
        self.labels_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.labels_action.triggered.connect(self.save_correction_labels)
        self.quit_action = QAction("Quit", self)
        self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_action.triggered.connect(self.close)

        self.settings_action = QAction("Advanced Detection Settings…", self)
        self.settings_action.triggered.connect(self.show_settings_dialog)
        self.debug_action = QAction("Show Debug Signal Graphs", self)
        self.debug_action.setCheckable(True)
        self.debug_action.setChecked(bool(self.detection_settings.get("debug_mode", False)))
        self.debug_action.toggled.connect(self._set_debug_mode)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_action)
        self._build_workflow_actions(file_menu)
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.labels_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.debug_action)
        edit_menu = self.menuBar().addMenu("Settings")
        edit_menu.addAction(self.settings_action)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 12, 16, 12)
        root_layout.setSpacing(10)
        self.setCentralWidget(root)

        header = QHBoxLayout()
        brand = QVBoxLayout()
        title = QLabel("Roundnet Rally Editor")
        title.setObjectName("appTitle")
        subtitle = QLabel("Find the points. Fix the edges. Export the game.")
        subtitle.setObjectName("subtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        header.addLayout(brand)
        header.addStretch(1)

        self.open_button = QPushButton("Open Game Video")
        self.open_button.setObjectName("primaryButton")
        self.open_button.clicked.connect(self.open_video)
        self.roi_button = QPushButton("Select Playing Area")
        self.roi_button.clicked.connect(self.select_roi)
        self.court_button = QPushButton("Net / Serve Zones")
        self.court_button.clicked.connect(self.select_court)
        self.detect_button = QPushButton("Detect Rallies")
        self.detect_button.setObjectName("accentButton")
        self.detect_button.clicked.connect(self.detect_rallies)
        self.settings_button = QPushButton("Advanced Settings")
        self.settings_button.clicked.connect(self.show_settings_dialog)
        self.export_button = QPushButton("Export Rally Video")
        self.export_button.setObjectName("primaryButton")
        self.export_button.clicked.connect(self.show_export_dialog)
        for button in (
            self.open_button,
            self.roi_button,
            self.court_button,
            self.detect_button,
            self.settings_button,
            self.export_button,
        ):
            button.setMinimumHeight(36)
            header.addWidget(button)
        root_layout.addLayout(header)

        self.metadata_frame = QFrame()
        self.metadata_frame.setObjectName("card")
        metadata_layout = QHBoxLayout(self.metadata_frame)
        metadata_layout.setContentsMargins(12, 8, 12, 8)
        self.file_label = QLabel("No video open")
        self.file_label.setObjectName("metadataFile")
        self.duration_label = QLabel("Duration  —")
        self.resolution_label = QLabel("Resolution  —")
        self.fps_label = QLabel("Frame rate  —")
        self.codec_label = QLabel("Codec  —")
        self.roi_status_label = QLabel("Playing area  —")
        metadata_layout.addWidget(self.file_label, 2)
        for label in (
            self.duration_label,
            self.resolution_label,
            self.fps_label,
            self.codec_label,
            self.roi_status_label,
        ):
            label.setObjectName("metadataValue")
            metadata_layout.addWidget(label)
        root_layout.addWidget(self.metadata_frame)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 7, 0)
        left_layout.setSpacing(8)
        self.player = VideoPlayerWidget()
        self.player.position_changed.connect(self._on_player_position)
        self.player.duration_changed.connect(self._on_player_duration)
        self.player.media_error.connect(self._on_media_error)
        left_layout.addWidget(self.player, 1)

        timeline_header = QHBoxLayout()
        timeline_title = QLabel("VIDEO TIMELINE")
        timeline_title.setObjectName("sectionLabel")
        self.timeline_hint = QLabel("Click/drag to scrub · highlighted regions are rallies")
        self.timeline_hint.setObjectName("hint")
        self.debug_checkbox = QCheckBox("Debug graphs")
        self.debug_checkbox.setChecked(bool(self.detection_settings.get("debug_mode", False)))
        self.debug_checkbox.toggled.connect(self._debug_checkbox_toggled)
        timeline_header.addWidget(timeline_title)
        timeline_header.addWidget(self.timeline_hint)
        timeline_header.addStretch(1)
        timeline_header.addWidget(self.debug_checkbox)
        left_layout.addLayout(timeline_header)

        self.timeline = RallyTimeline()
        self.timeline.set_debug_enabled(self.debug_checkbox.isChecked())
        self.timeline.set_threshold(float(self.detection_settings.get("rally_threshold", 0.55)))
        self.timeline.position_requested.connect(self.player.set_position)
        self.timeline.rally_selected.connect(self._timeline_selected_rally)
        left_layout.addWidget(self.timeline)

        edit_bar = QFrame()
        edit_bar.setObjectName("card")
        edit_layout = QHBoxLayout(edit_bar)
        edit_layout.setContentsMargins(8, 7, 8, 7)
        self.add_button = QPushButton("＋ Add Rally")
        self.add_button.setToolTip("Add a rally starting at the playhead (A)")
        self.add_button.clicked.connect(self.add_rally)
        self.delete_button = QPushButton("Reject / Restore")
        self.delete_button.setToolTip(
            "Reject a false detection, or restore a previously rejected rally (Delete)"
        )
        self.delete_button.clicked.connect(self.delete_selected_rally)
        self.split_button = QPushButton("Split at Playhead")
        self.split_button.setToolTip("Split the selected rally at the current frame (S)")
        self.split_button.clicked.connect(self.split_selected_rally)
        self.merge_button = QPushButton("Merge Neighbor")
        self.merge_button.setToolTip("Merge the selected rally with the next neighboring rally (M)")
        self.merge_button.clicked.connect(self.merge_selected_rally)
        self.preview_button = QPushButton("▶ Preview Selected")
        self.preview_button.setToolTip("Play only the selected rally (P)")
        self.preview_button.clicked.connect(self.preview_selected_rally)
        for button in (
            self.add_button,
            self.delete_button,
            self.split_button,
            self.merge_button,
            self.preview_button,
        ):
            edit_layout.addWidget(button)
        edit_layout.addStretch(1)
        left_layout.addWidget(edit_bar)
        transport = QHBoxLayout()
        play_all = QPushButton("▶ Play All Kept")
        play_all.clicked.connect(self.play_all_rallies)
        self.loop_checkbox = QCheckBox("Loop selected clip")
        self.loop_checkbox.setChecked(True)
        mark_in = QPushButton("Mark Rally Start (W)")
        mark_in.clicked.connect(self.manual_mark_start)
        mark_out = QPushButton("Mark Rally End (X)")
        mark_out.clicked.connect(self.manual_mark_end)
        for widget in (play_all, self.loop_checkbox, mark_in, mark_out):
            transport.addWidget(widget)
        left_layout.addLayout(transport)
        splitter.addWidget(left)

        right = QWidget()
        right.setMinimumWidth(385)
        right.setMaximumWidth(515)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(7, 0, 0, 0)
        right_layout.setSpacing(8)

        list_header = QHBoxLayout()
        list_title = QLabel("RALLIES")
        list_title.setObjectName("sectionLabel")
        self.rally_count_badge = QLabel("0")
        self.rally_count_badge.setObjectName("badge")
        self.save_labels_button = QPushButton("Save Correction Labels…")
        self.save_labels_button.setToolTip(
            "Save reviewed ranges plus detector features as local training data"
        )
        self.save_labels_button.clicked.connect(self.save_correction_labels)
        list_header.addWidget(list_title)
        list_header.addWidget(self.rally_count_badge)
        list_header.addStretch(1)
        list_header.addWidget(self.save_labels_button)
        right_layout.addLayout(list_header)
        self._build_review_toolbar(right_layout)

        self.rally_tree = QTreeWidget()
        self.rally_tree.setColumnCount(8)
        self.rally_tree.setHeaderLabels(
            ("Use", "Rally", "Start", "End", "Length", "Conf.", "Serve", "Review")
        )
        self.rally_tree.setRootIsDecorated(False)
        self.rally_tree.setAlternatingRowColors(True)
        self.rally_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.rally_tree.setUniformRowHeights(True)
        self.rally_tree.setMinimumHeight(140)
        self.rally_tree.setSortingEnabled(False)
        self.rally_tree.itemSelectionChanged.connect(self._tree_selection_changed)
        self.rally_tree.itemChanged.connect(self._tree_item_changed)
        self.rally_tree.itemDoubleClicked.connect(lambda *_: self.preview_selected_rally())
        header_view = self.rally_tree.header()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        right_layout.addWidget(self.rally_tree, 1)

        editor = QGroupBox("Selected rally")
        editor_layout = QGridLayout(editor)
        self.selected_label = QLabel("No rally selected")
        self.selected_label.setObjectName("selectedRally")
        self.selected_label.setWordWrap(True)
        editor_layout.addWidget(self.selected_label, 0, 0, 1, 6)
        self.start_spin = self._make_time_spin()
        self.end_spin = self._make_time_spin()
        editor_layout.addWidget(QLabel("Start"), 1, 0)
        editor_layout.addWidget(self.start_spin, 1, 1, 1, 3)
        editor_layout.addWidget(QLabel("End"), 2, 0)
        editor_layout.addWidget(self.end_spin, 2, 1, 1, 3)

        self.start_minus_button = QPushButton("−0.1")
        self.start_minus_button.clicked.connect(lambda: self._nudge_boundary("start", -0.1))
        self.start_plus_button = QPushButton("+0.1")
        self.start_plus_button.clicked.connect(lambda: self._nudge_boundary("start", 0.1))
        self.end_minus_button = QPushButton("−0.1")
        self.end_minus_button.clicked.connect(lambda: self._nudge_boundary("end", -0.1))
        self.end_plus_button = QPushButton("+0.1")
        self.end_plus_button.clicked.connect(lambda: self._nudge_boundary("end", 0.1))
        editor_layout.addWidget(self.start_minus_button, 1, 4)
        editor_layout.addWidget(self.start_plus_button, 1, 5)
        editor_layout.addWidget(self.end_minus_button, 2, 4)
        editor_layout.addWidget(self.end_plus_button, 2, 5)

        self.start_playhead_button = QPushButton("Set Start to Playhead")
        self.start_playhead_button.clicked.connect(lambda: self._set_boundary_to_playhead("start"))
        self.end_playhead_button = QPushButton("Set End to Playhead")
        self.end_playhead_button.clicked.connect(lambda: self._set_boundary_to_playhead("end"))
        editor_layout.addWidget(self.start_playhead_button, 3, 0, 1, 3)
        editor_layout.addWidget(self.end_playhead_button, 3, 3, 1, 3)
        self.apply_times_button = QPushButton("Apply Precise Times")
        self.apply_times_button.setObjectName("accentButton")
        self.apply_times_button.clicked.connect(self.apply_time_edits)
        editor_layout.addWidget(self.apply_times_button, 4, 0, 1, 6)
        tabs = QTabWidget()
        tabs.addTab(editor, "Trim")
        point_scroll = QScrollArea()
        point_scroll.setWidgetResizable(True)
        point_scroll.setFrameShape(QFrame.Shape.NoFrame)
        point_scroll.setWidget(self._build_point_panel())
        tabs.addTab(point_scroll, "Point / Highlight")
        tabs.setMaximumHeight(350)
        right_layout.addWidget(tabs)

        summary = QGroupBox("Edit summary")
        summary_layout = QFormLayout(summary)
        self.summary_original = QLabel("00:00.0")
        self.summary_count = QLabel("0")
        self.summary_play = QLabel("00:00.0")
        self.summary_removed = QLabel("00:00.0")
        self.summary_estimate = QLabel("00:00.0")
        summary_layout.addRow("Original video", self.summary_original)
        summary_layout.addRow("Detected rallies", self.summary_count)
        summary_layout.addRow("Actual play", self.summary_play)
        summary_layout.addRow("Removed downtime", self.summary_removed)
        summary_layout.addRow("Estimated final video", self.summary_estimate)
        right_layout.addWidget(summary)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes((1000, 420))

        analysis_bar = QFrame()
        analysis_bar.setObjectName("card")
        analysis_layout = QHBoxLayout(analysis_bar)
        analysis_layout.setContentsMargins(10, 7, 10, 7)
        self.analysis_status = QLabel("Open a game video to begin")
        self.analysis_status.setMinimumWidth(300)
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setTextVisible(True)
        self.analysis_progress.setMinimumWidth(230)
        self.detected_count_label = QLabel("Rallies found: 0")
        self.cancel_detection_button = QPushButton("Cancel Analysis")
        self.cancel_detection_button.clicked.connect(self.cancel_detection)
        self.cancel_detection_button.hide()
        analysis_layout.addWidget(self.analysis_status, 1)
        analysis_layout.addWidget(self.analysis_progress)
        analysis_layout.addWidget(self.detected_count_label)
        analysis_layout.addWidget(self.cancel_detection_button)
        root_layout.addWidget(analysis_bar)

        status = QStatusBar()
        self.setStatusBar(status)
        self.shortcut_hint = QLabel(
            "Space play  ·  ←/→ seek  ·  W/X mark  ·  N review  ·  F star  ·  S split  ·  M merge"
        )
        status.addPermanentWidget(self.shortcut_hint)
        status.showMessage("Ready")

    def _make_time_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 24 * 60 * 60.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setSuffix(" s")
        spin.setKeyboardTracking(False)
        return spin

    def _build_shortcuts(self) -> None:
        shortcuts: Sequence[tuple[str, Any]] = (
            ("Space", self.player.toggle_playback),
            ("Left", lambda: self.player.seek_relative(-0.25)),
            ("Right", lambda: self.player.seek_relative(0.25)),
            ("Shift+Left", lambda: self.player.seek_relative(-5.0)),
            ("Shift+Right", lambda: self.player.seek_relative(5.0)),
            ("A", self.add_rally),
            ("Delete", self.delete_selected_rally),
            ("Backspace", self.delete_selected_rally),
            ("S", self.split_selected_rally),
            ("M", self.merge_selected_rally),
            ("P", self.preview_selected_rally),
            ("I", lambda: self._set_boundary_to_playhead("start")),
            ("O", lambda: self._set_boundary_to_playhead("end")),
            ("W", self.manual_mark_start),
            ("X", self.manual_mark_end),
            ("N", self.next_uncertain),
            ("F", self.toggle_star),
            ("R", self.mark_reviewed),
            ("1", lambda: self._set_selected(winner="A", reviewed=True)),
            ("2", lambda: self._set_selected(winner="B", reviewed=True)),
            ("[", lambda: self.adjacent_rally(-1)),
            ("]", lambda: self.adjacent_rally(1)),
            (",", lambda: self.player.seek_relative(-1.0 / self._source_fps())),
            (".", lambda: self.player.seek_relative(1.0 / self._source_fps())),
        )
        self._shortcuts: list[QShortcut] = []
        for key, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda cb=callback: self._run_edit_shortcut(cb))
            self._shortcuts.append(shortcut)
        QApplication.instance().focusChanged.connect(self._update_shortcut_focus)

    def _update_shortcut_focus(self, old, focus):
        from PySide6.QtWidgets import QAbstractSpinBox, QPlainTextEdit, QTextEdit
        typing = isinstance(focus, (QAbstractSpinBox, QLineEdit, QTextEdit, QPlainTextEdit, QComboBox))
        for shortcut in self._shortcuts:
            shortcut.setEnabled(not typing)

    def _run_edit_shortcut(self, callback):
        # Letter shortcuts must never eat a team name, caption, or numeric edit.
        from PySide6.QtWidgets import QAbstractSpinBox, QLineEdit, QTextEdit
        focus = QApplication.focusWidget()
        if isinstance(focus, (QAbstractSpinBox, QLineEdit, QTextEdit, QComboBox)):
            return
        callback()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0d1117; color: #e6edf3; }
            QMenuBar { background: #161b22; }
            QMenuBar::item:selected, QMenu::item:selected { background: #263349; }
            QMenu { background: #161b22; border: 1px solid #30363d; }
            QLabel#appTitle { font-size: 22px; font-weight: 700; }
            QLabel#subtitle, QLabel#hint { color: #8b949e; }
            QLabel#sectionLabel { font-size: 11px; font-weight: 700; color: #8b949e; letter-spacing: 1px; }
            QLabel#badge { background: #238636; border-radius: 9px; padding: 1px 7px; font-weight: 700; }
            QLabel#metadataFile { font-weight: 600; }
            QLabel#metadataValue { color: #b1bac4; }
            QLabel#selectedRally { font-weight: 600; color: #58a6ff; }
            QFrame#card, QGroupBox { background: #161b22; border: 1px solid #30363d; border-radius: 7px; }
            QGroupBox { margin-top: 12px; padding: 9px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background: #21262d; border: 1px solid #3d444d; border-radius: 5px; padding: 6px 10px; }
            QPushButton:hover { background: #30363d; border-color: #58a6ff; }
            QPushButton:pressed { background: #1f6feb; }
            QPushButton:disabled { color: #6e7681; background: #161b22; border-color: #30363d; }
            QPushButton#primaryButton { background: #1f6feb; border-color: #388bfd; font-weight: 600; }
            QPushButton#accentButton { background: #238636; border-color: #2ea043; font-weight: 600; }
            QTreeWidget, QLineEdit, QDoubleSpinBox, QComboBox, QScrollArea {
                background: #0d1117; border: 1px solid #30363d; border-radius: 4px;
            }
            QTreeWidget::item { min-height: 25px; }
            QTreeWidget::item:selected { background: #1f6feb; }
            QHeaderView::section { background: #21262d; color: #b1bac4; border: 0; padding: 5px; }
            QProgressBar { background: #0d1117; border: 1px solid #30363d; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background: #238636; border-radius: 3px; }
            QStatusBar { background: #161b22; color: #8b949e; }
            QSplitter::handle { background: #30363d; width: 1px; }
            """
        )

    # -------------------------------------------------------------- persistence
    def _load_detection_settings(self) -> dict[str, Any]:
        values = dict(DEFAULT_SETTINGS)
        raw = self._settings_store.value("detection/settings")
        if raw:
            try:
                payload = json.loads(str(raw))
                if isinstance(payload, dict):
                    # Phase 1 stored a four-signal weight set.  Carrying those
                    # values forward and then adding the two new defaults would
                    # silently distort the intended Phase 2 balance, so migrate
                    # old profiles to the new six-signal defaults as a unit.
                    if "motion_spread_weight" not in payload and "serve_weight" not in payload:
                        for key in (
                            "motion_weight",
                            "roi_motion_weight",
                            "motion_spread_weight",
                            "audio_weight",
                            "temporal_weight",
                            "serve_weight",
                        ):
                            payload.pop(key, None)
                    values.update(payload)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                LOGGER.warning("Ignoring invalid saved detection settings: %s", exc)
        return values

    def _save_detection_settings(self) -> None:
        self._settings_store.setValue("detection/settings", json.dumps(self.detection_settings, sort_keys=True))

    # --------------------------------------------------------------- video open
    def open_video(self) -> None:
        start_dir = self._settings_store.value("paths/last_video_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Game Video",
            str(start_dir),
            "Video Files (*.mp4 *.mov *.m4v *.mkv *.avi *.mts *.m2ts *.webm *.MP4 *.MOV *.M4V *.MKV *.AVI *.MTS *.M2TS *.WEBM);;All Files (*)",
        )
        if path:
            self.load_video(path)

    def load_video(self, path: str, *, restore_recovery: bool = True) -> bool:
        if self._detection_worker and self._detection_worker.isRunning() or self._export_worker and self._export_worker.isRunning():
            return False
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            QMessageBox.warning(self, "Video Not Found", f"The selected file does not exist:\n{candidate}")
            return False
        if candidate.suffix.lower() not in {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".webm"}:
            answer = QMessageBox.question(
                self,
                "Unrecognized Video Type",
                "This file has an unrecognized video extension. Try opening it anyway?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        if not self._ensure_saved_before_leaving():
            return False
        self._autosave_timer.stop()
        self._project_dirty = False
        from models.project import EditHistory
        self._history = EditHistory()
        self.project_path = None
        self.court_context = None
        self.analysis_settings = None
        self.analysis_roi = None
        self.analysis_court_context = None
        self._manual_start = None
        self.match_settings = {"team_a": "Team A", "team_b": "Team B", "initial_score_a": 0, "initial_score_b": 0}
        with QSignalBlocker(self.complete_review_checkbox):
            self.complete_review_checkbox.setChecked(False)
        self.player.load(str(candidate))
        self.video_path = str(candidate)
        self.video_metadata = None
        self.video_duration = 0.0
        self.rallies = []
        self.initial_detected_rallies = []
        self.rejected_detections = []
        self.selected_rally_index = -1
        self.detection_result = None
        self.timeline.set_signal_data(None)
        self._settings_store.setValue("paths/last_video_dir", str(candidate.parent))
        self.analysis_status.setText("Reading video metadata…")
        QApplication.processEvents()

        metadata_error: Exception | None = None
        try:
            from video.metadata import probe_video_metadata

            self.video_metadata = probe_video_metadata(str(candidate))
        except Exception as exc:
            metadata_error = exc
            self.video_metadata = self._fallback_probe(str(candidate))

        self.video_duration = float(
            _metadata_value(self.video_metadata, "duration_seconds", "duration", default=0.0) or 0.0
        )
        self.timeline.set_duration(self.video_duration)
        self._update_metadata_labels()
        self._rebuild_rally_tree()
        self._update_summary()
        self._update_ui_state()
        self.analysis_progress.setValue(0)
        self.detected_count_label.setText("Rallies found: 0")
        self.analysis_status.setToolTip("")
        if metadata_error and not self.video_metadata:
            self.analysis_status.setText("Video opened; some metadata is unavailable")
            self.statusBar().showMessage(str(metadata_error), 8000)
        else:
            self.analysis_status.setText("Ready to select a playing area and detect rallies")
            self.statusBar().showMessage(f"Opened {candidate.name}", 5000)
        if restore_recovery:
            from models.project import read_project, recovery_path
            recovery = recovery_path(str(candidate), self._recovery_dir)
            if recovery.is_file():
                try:
                    state = read_project(recovery)
                    stat = candidate.stat()
                    expected = state.get("source_stat", {})
                    if expected.get("size", stat.st_size) == stat.st_size and expected.get("mtime_ns", stat.st_mtime_ns) == stat.st_mtime_ns:
                        self._apply_project(state)
                        self.statusBar().showMessage("Recovered saved edits and analysis for this video", 8000)
                except Exception as exc:
                    self.statusBar().showMessage(f"Could not restore saved edits: {exc}", 10000)
        return True

    def _fallback_probe(self, path: str) -> dict[str, Any] | None:
        try:
            import cv2

            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                return None
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            capture.release()
            duration = frame_count / fps if fps > 0 else 0.0
            return {
                "duration_seconds": duration,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "video_codec": "unknown",
            }
        except Exception as exc:
            LOGGER.warning("OpenCV metadata fallback failed for %s: %s", path, exc)
            return None

    def _update_metadata_labels(self) -> None:
        if not self.video_path:
            self.file_label.setText("No video open")
            return
        path = Path(self.video_path)
        self.file_label.setText(path.name)
        self.file_label.setToolTip(str(path))
        self.duration_label.setText(f"Duration  {format_timestamp(self.video_duration)}")
        display_resolution = _metadata_value(self.video_metadata, "display_resolution", default=None)
        if isinstance(display_resolution, (tuple, list)) and len(display_resolution) == 2:
            width, height = (int(value or 0) for value in display_resolution)
        else:
            width = int(_metadata_value(self.video_metadata, "width", default=0) or 0)
            height = int(_metadata_value(self.video_metadata, "height", default=0) or 0)
        self.resolution_label.setText(f"Resolution  {width}×{height}" if width and height else "Resolution  —")
        fps = float(_metadata_value(self.video_metadata, "fps", default=0.0) or 0.0)
        self.fps_label.setText(f"Frame rate  {fps:.2f} fps" if fps else "Frame rate  —")
        codec = str(_metadata_value(self.video_metadata, "video_codec", default="") or "")
        self.codec_label.setText(f"Codec  {codec.upper()}" if codec else "Codec  —")
        self._update_roi_status()

    def _on_player_duration(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self.video_duration <= 0 or abs(seconds - self.video_duration) > 0.25:
            self.video_duration = seconds
            self.timeline.set_duration(seconds)
            self.duration_label.setText(f"Duration  {format_timestamp(seconds)}")
            self._set_time_spin_ranges()
            self._update_summary()

    def _on_media_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Playback: {message}", 10000)

    def _stop_background_tasks_for_new_video(self) -> None:
        if self._detection_worker and self._detection_worker.isRunning():
            self._detection_worker.request_cancel()
        if self._export_worker and self._export_worker.isRunning():
            QMessageBox.information(
                self,
                "Export in Progress",
                "Wait for the current export to finish or cancel it before opening another video.",
            )

    # --------------------------------------------------------------- ROI picker
    def select_roi(self) -> None:
        if not self.video_path:
            QMessageBox.information(self, "Open a Video", "Open a game video before selecting its playing area.")
            return
        try:
            dialog = ROISelectorDialog(self.video_path, self.roi, self)
        except Exception as exc:
            QMessageBox.critical(self, "Could Not Show Video Frame", str(exc))
            return
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.roi = dialog.selected_roi
            self._update_roi_status()
            self.analysis_status.setText("Playing area saved — ready to detect rallies")
            self.statusBar().showMessage("Playing-area ROI saved", 4000)
            self._changed()

    def _update_roi_status(self) -> None:
        if not hasattr(self, "roi_status_label"):
            return
        if self.roi is None:
            self.roi_status_label.setText("Playing area  Full frame")
            self.roi_status_label.setToolTip("No ROI is selected; detection will analyze the full frame.")
        elif self.roi == (0.0, 0.0, 1.0, 1.0):
            self.roi_status_label.setText("Playing area  Full frame")
            self.roi_status_label.setToolTip("The full frame is selected.")
        else:
            _, _, width, height = self.roi
            self.roi_status_label.setText(f"Playing area  {width * 100:.0f}% × {height * 100:.0f}%")
            self.roi_status_label.setToolTip("A saved, normalized playing-area ROI will be reused.")

    # --------------------------------------------------------------- detection
    def detect_rallies(self) -> None:
        if not self.video_path:
            QMessageBox.information(self, "Open a Video", "Open a game video before detecting rallies.")
            return
        if self._busy_editing():
            return
        if self.rallies and QMessageBox.question(self, "Analyze Again?",
                "A new analysis will replace the current rally cuts and point tags. "
                "You can use Undo to return to this version. Continue?") != QMessageBox.StandardButton.Yes:
            return
        if self.roi is None:
            answer = QMessageBox.question(
                self,
                "Analyze the Full Frame?",
                "No playing area is selected. Detection can use the full frame, but an ROI is usually more accurate. "
                "Continue with the full frame?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.select_roi()
                return

        self.analysis_progress.setValue(0)
        self.analysis_status.setToolTip("")
        self.analysis_status.setText("Preparing video analysis…")
        self.detected_count_label.setText("Rallies found: 0")
        self.cancel_detection_button.setText("Cancel Analysis")
        self.cancel_detection_button.setEnabled(True)
        self.cancel_detection_button.show()
        self.detect_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.open_action.setEnabled(False)

        worker = DetectionWorker(self.video_path, self.roi, self.detection_settings, self)
        worker.court_context = deepcopy(self.court_context)
        worker.profile_path = self.profile_path
        self._pending_analysis_settings = deepcopy(self.detection_settings)
        self._pending_analysis_roi = deepcopy(self.roi)
        self._pending_analysis_court = deepcopy(self.court_context)
        self._detection_worker = worker
        worker.progress.connect(self._on_detection_progress)
        worker.rally_count.connect(lambda count: self.detected_count_label.setText(f"Rallies found: {count}"))
        worker.succeeded.connect(self._on_detection_succeeded)
        worker.failed.connect(self._on_detection_failed)
        worker.cancelled.connect(self._on_detection_cancelled)
        worker.finished.connect(lambda: self._detection_thread_finished(worker))
        worker.start()
        self._update_ui_state()

    def cancel_detection(self) -> None:
        if self._detection_worker and self._detection_worker.isRunning():
            self._detection_worker.request_cancel()
            self.cancel_detection_button.setEnabled(False)
            self.cancel_detection_button.setText("Cancelling…")
            self.analysis_status.setText("Cancelling analysis safely…")

    def _on_detection_progress(self, percent: int, message: str) -> None:
        self.analysis_progress.setValue(percent)
        self.analysis_status.setText(message or "Analyzing video…")

    def _on_detection_succeeded(self, result: Any) -> None:
        self._checkpoint(include_analysis=True)
        self.detection_result = result
        self.analysis_settings = deepcopy(getattr(self, "_pending_analysis_settings", self.detection_settings))
        self.analysis_roi = deepcopy(getattr(self, "_pending_analysis_roi", self.roi))
        self.analysis_court_context = deepcopy(getattr(self, "_pending_analysis_court", self.court_context))
        self.rallies = list(getattr(result, "rallies", ()) or ())
        self.rallies.sort(key=lambda rally: _rally_values(rally)[0])
        self.initial_detected_rallies = [
            _rally_snapshot(rally) for rally in self.rallies
        ]
        self.rejected_detections = []
        result_duration = float(getattr(result, "duration", 0.0) or 0.0)
        if result_duration > 0 and self.video_duration <= 0:
            self.video_duration = result_duration
            self.timeline.set_duration(result_duration)
        self.selected_rally_index = 0 if self.rallies else -1
        self.timeline.set_signal_data(result)
        self._rebuild_rally_tree(select_index=self.selected_rally_index, seek=False)
        self._update_summary()
        self._update_ui_state()
        self.analysis_progress.setValue(100)
        enabled_count = len(self._enabled_rallies())
        suppressed_count = len(self.rallies) - enabled_count
        if suppressed_count:
            completion_message = (
                f"Detection complete — {enabled_count} likely rallies; "
                f"{suppressed_count} compact-motion candidate"
                f"{'s' if suppressed_count != 1 else ''} left unchecked for review"
            )
            self.detected_count_label.setText(
                f"Likely rallies: {enabled_count} (+{suppressed_count} review)"
            )
        else:
            completion_message = (
                f"Detection complete — review {len(self.rallies)} rallies and correct any mistakes"
            )
            self.detected_count_label.setText(f"Rallies found: {len(self.rallies)}")
        self.analysis_status.setText(completion_message)
        warnings = tuple(getattr(result, "warnings", ()) or ())
        if warnings:
            self.analysis_status.setText(
                f"{completion_message}; note: {warnings[0]}"
            )
            self.analysis_status.setToolTip("\n".join(str(warning) for warning in warnings))
            self.statusBar().showMessage(f"Detection warning: {warnings[0]}", 10000)
        else:
            self.analysis_status.setToolTip("")
            self.statusBar().showMessage("Rally detection complete", 6000)
        self._changed()

    def _on_detection_failed(self, message: str) -> None:
        self.analysis_status.setText("Analysis failed")
        self.analysis_progress.setValue(0)
        QMessageBox.critical(
            self,
            "Rally Detection Failed",
            f"The video could not be analyzed.\n\n{message}",
        )

    def _on_detection_cancelled(self) -> None:
        self.analysis_status.setText("Analysis cancelled")
        self.analysis_progress.setValue(0)
        self.statusBar().showMessage("Analysis cancelled", 4000)

    def _detection_thread_finished(self, worker: DetectionWorker) -> None:
        if self._detection_worker is worker:
            self._detection_worker = None
        self.cancel_detection_button.hide()
        self._update_ui_state()
        worker.deleteLater()

    # ----------------------------------------------------------- rally editing
    def _rally_duration(self, rally: Any) -> float:
        start, end, _, _ = _rally_values(rally)
        return max(0.0, end - start)

    def _enabled_rallies(self) -> list[Any]:
        return [rally for rally in self.rallies if _rally_values(rally)[3] and not getattr(rally, "rejected", False)]

    def _replace_rally(
        self,
        index: int,
        *,
        start: float | None = None,
        end: float | None = None,
        confidence: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        old_start, old_end, old_confidence, old_enabled = _rally_values(self.rallies[index])
        old_serve_confidence = _serve_confidence(self.rallies[index])
        self.rallies[index] = replace(
            self.rallies[index], start_time=old_start if start is None else start,
            end_time=old_end if end is None else end,
            confidence=old_confidence if confidence is None else confidence,
            enabled=old_enabled if enabled is None else enabled,
        )

    def _rebuild_rally_tree(self, select_index: int | None = None, *, seek: bool = False) -> None:
        if select_index is None:
            select_index = self.selected_rally_index
        with QSignalBlocker(self.rally_tree):
            self.rally_tree.clear()
            for index, rally in enumerate(self.rallies):
                start, end, confidence, enabled = _rally_values(rally)
                serve_confidence = _serve_confidence(rally)
                item = QTreeWidgetItem(
                    (
                        "",
                        ("★ " if rally.starred else "") + str(index + 1),
                        format_timestamp(start),
                        format_timestamp(end),
                        f"{max(0.0, end - start):.2f} s",
                        f"{confidence * 100:.0f}%",
                        f"{serve_confidence * 100:.0f}%" if serve_confidence > 0 else "—",
                        "Rejected" if rally.rejected else "✓" if rally.reviewed else "Pending",
                    )
                )
                item.setData(0, Qt.ItemDataRole.UserRole, index)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)
                if not enabled:
                    for column in range(1, 8):
                        item.setForeground(column, Qt.GlobalColor.gray)
                self.rally_tree.addTopLevelItem(item)
            if 0 <= select_index < len(self.rallies):
                self.rally_tree.setCurrentItem(self.rally_tree.topLevelItem(select_index))
                self.selected_rally_index = select_index
            else:
                self.selected_rally_index = -1
        self.timeline.set_rallies(self.rallies, self.selected_rally_index)
        self.rally_count_badge.setText(str(len(self.rallies)))
        self._load_selected_into_editor()
        self._apply_review_filter()
        if seek and self.selected_rally_index >= 0:
            self.player.set_position(_rally_values(self.rallies[self.selected_rally_index])[0])

    def _tree_selection_changed(self) -> None:
        items = self.rally_tree.selectedItems()
        if not items:
            self.selected_rally_index = -1
            self.timeline.set_selected_index(-1)
            self._load_selected_into_editor()
            self._update_ui_state()
            return
        index = int(items[0].data(0, Qt.ItemDataRole.UserRole))
        self._select_rally(index, seek=True)

    def _tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        index = int(item.data(0, Qt.ItemDataRole.UserRole))
        if not 0 <= index < len(self.rallies):
            return
        enabled = item.checkState(0) == Qt.CheckState.Checked
        if self.rallies[index].rejected and enabled:
            self.selected_rally_index = index
            self.restore_rally()
            return
        self._checkpoint()
        self._replace_rally(index, enabled=enabled)
        foreground = QBrush(QColor("#e6edf3" if enabled else "#8b949e"))
        for text_column in range(1, 8):
            item.setForeground(text_column, foreground)
        self.timeline.set_rallies(self.rallies, self.selected_rally_index)
        self._update_summary()
        self._update_ui_state()
        self._changed()

    def _timeline_selected_rally(self, index: int) -> None:
        self._select_rally(index, seek=False)

    def _select_rally(self, index: int, *, seek: bool) -> None:
        if not 0 <= index < len(self.rallies):
            return
        self.selected_rally_index = index
        with QSignalBlocker(self.rally_tree):
            item = self.rally_tree.topLevelItem(index)
            self.rally_tree.setCurrentItem(item)
            self.rally_tree.scrollToItem(item)
        self.timeline.set_selected_index(index)
        self._load_selected_into_editor()
        self._update_ui_state()
        if seek:
            self.player.set_position(_rally_values(self.rallies[index])[0])

    def _load_selected_into_editor(self) -> None:
        self._load_point_panel()
        valid = 0 <= self.selected_rally_index < len(self.rallies)
        if not valid:
            self.selected_label.setText("No rally selected")
            with QSignalBlocker(self.start_spin), QSignalBlocker(self.end_spin):
                self.start_spin.setValue(0.0)
                self.end_spin.setValue(0.0)
            return
        start, end, confidence, enabled = _rally_values(self.rallies[self.selected_rally_index])
        serve_confidence = _serve_confidence(self.rallies[self.selected_rally_index])
        serve_text = (
            f" · {serve_confidence * 100:.0f}% serve cue"
            if serve_confidence > 0
            else ""
        )
        self.selected_label.setText(
            f"Rally {self.selected_rally_index + 1} · {end - start:.2f} s · "
            f"{confidence * 100:.0f}% confidence"
            + serve_text
            + ("" if enabled else " · disabled")
        )
        with QSignalBlocker(self.start_spin), QSignalBlocker(self.end_spin):
            self.start_spin.setValue(start)
            self.end_spin.setValue(end)

    def _set_time_spin_ranges(self) -> None:
        maximum = max(1.0, self.video_duration)
        self.start_spin.setMaximum(maximum)
        self.end_spin.setMaximum(maximum)

    def add_rally(self) -> None:
        if not self.video_path or self._busy_editing():
            return
        self._checkpoint()
        position = self.player.position_seconds
        default_length = max(2.0, float(self.detection_settings.get("min_rally_duration", 1.75)))
        if self.video_duration and position + default_length > self.video_duration:
            end = self.video_duration
            start = max(0.0, end - default_length)
        else:
            start = position
            end = position + default_length
        new_rally = replace(_new_rally(start, end, 1.0, True), reviewed=True)
        self.rallies.append(new_rally)
        self.rallies.sort(key=lambda rally: _rally_values(rally)[0])
        index = self.rallies.index(new_rally)
        self.selected_rally_index = index
        self._rebuild_rally_tree(index, seek=False)
        self._update_summary()
        self._update_ui_state()
        self.statusBar().showMessage(f"Added Rally {index + 1}; adjust its boundaries if needed", 5000)
        self._changed()

    def delete_selected_rally(self) -> None:
        index = self.selected_rally_index
        if not 0 <= index < len(self.rallies) or self._busy_editing():
            return
        if self.rallies[index].rejected:
            self.restore_rally()
            return
        self._checkpoint()
        rejected = _rally_snapshot(self.rallies[index])
        rejected["reason"] = "user_rejected_false_detection"
        self.rejected_detections.append(rejected)
        self.rallies[index] = replace(self.rallies[index], rejected=True, enabled=False, reviewed=True)
        next_index = index
        self.selected_rally_index = next_index
        self._rebuild_rally_tree(next_index, seek=False)
        self._update_summary()
        self._update_ui_state()
        self.statusBar().showMessage(
            "Rally rejected. It remains visible; Reject / Restore or Undo brings it back.", 6000
        )
        self._changed()

    def apply_time_edits(self) -> None:
        index = self.selected_rally_index
        if not 0 <= index < len(self.rallies) or self._busy_editing():
            return
        start = self.start_spin.value()
        end = self.end_spin.value()
        if end <= start:
            QMessageBox.warning(self, "Invalid Rally Range", "A rally's end must be later than its start.")
            self._load_selected_into_editor()
            return
        if self.video_duration:
            end = min(end, self.video_duration)
        self._checkpoint()
        self._replace_rally(index, start=start, end=end)
        self.rallies[index] = replace(self.rallies[index], reviewed=True)
        edited = self.rallies[index]
        self.rallies.sort(key=lambda rally: _rally_values(rally)[0])
        new_index = self.rallies.index(edited)
        self.selected_rally_index = new_index
        self._rebuild_rally_tree(new_index, seek=False)
        self._update_summary()
        self.statusBar().showMessage("Rally boundaries updated", 3500)
        self._changed()

    def _nudge_boundary(self, boundary: str, amount: float) -> None:
        if not 0 <= self.selected_rally_index < len(self.rallies):
            return
        control = self.start_spin if boundary == "start" else self.end_spin
        control.setValue(control.value() + amount)
        self.apply_time_edits()

    def _set_boundary_to_playhead(self, boundary: str) -> None:
        if not 0 <= self.selected_rally_index < len(self.rallies):
            return
        position = self.player.position_seconds
        if boundary == "start":
            self.start_spin.setValue(position)
        else:
            self.end_spin.setValue(position)
        self.apply_time_edits()

    def split_selected_rally(self) -> None:
        index = self.selected_rally_index
        if not 0 <= index < len(self.rallies) or self._busy_editing() or self.rallies[index].rejected:
            return
        start, end, confidence, enabled = _rally_values(self.rallies[index])
        serve_confidence = _serve_confidence(self.rallies[index])
        split = self.player.position_seconds
        minimum_piece = 0.05
        if not (start + minimum_piece < split < end - minimum_piece):
            QMessageBox.information(
                self,
                "Move the Playhead",
                "Place the playhead inside the selected rally, away from its start and end, then split again.",
            )
            return
        if self._busy_editing():
            return
        from uuid import uuid4
        if self.rallies[index].point_stats and QMessageBox.question(self, "Split Annotated Point?",
                "Splitting clears the touch log for both pieces so statistics are not counted twice. "
                "Undo restores the original point and its log. Continue?") != QMessageBox.StandardButton.Yes:
            return
        self._checkpoint()
        original = self.rallies[index]
        self.rallies[index : index + 1] = (
            replace(original, end_time=split, reviewed=False, point_stats={}),
            replace(original, start_time=split, rally_id=uuid4().hex, winner="", outcome="", reviewed=False, serve_confidence=0.0, point_stats={}),
        )
        self.selected_rally_index = index + 1
        self._rebuild_rally_tree(index + 1, seek=False)
        self._update_summary()
        self.statusBar().showMessage("Rally split at playhead", 3500)
        self._changed()

    def merge_selected_rally(self) -> None:
        index = self.selected_rally_index
        if not 0 <= index < len(self.rallies) or len(self.rallies) < 2 or self._busy_editing():
            return
        left_index = index if index < len(self.rallies) - 1 else index - 1
        left, right = self.rallies[left_index:left_index + 2]
        if left.rejected or right.rejected:
            self.statusBar().showMessage("Restore a rejected candidate before merging it", 5000)
            return
        if right.winner or right.outcome or right.player or right.note or left.point_stats or right.point_stats:
            if QMessageBox.question(self, "Merge Point Annotations?",
                    "Merging makes these clips one point. The earlier point's tags are retained, "
                    "and the later point's tags are removed. Touch logs are cleared to avoid combining two points' statistics. "
                    "You can undo this. Continue?") != QMessageBox.StandardButton.Yes:
                return
        first = _rally_values(self.rallies[left_index])
        second = _rally_values(self.rallies[left_index + 1])
        merged_serve_confidence = max(
            _serve_confidence(self.rallies[left_index]),
            _serve_confidence(self.rallies[left_index + 1]),
        )
        self._checkpoint()
        merged = replace(self.rallies[left_index], start_time=min(first[0], second[0]),
                         end_time=max(first[1], second[1]), confidence=max(first[2], second[2]),
                         enabled=first[3] or second[3], rejected=False, reviewed=False, point_stats={},
                         serve_confidence=merged_serve_confidence, starred=left.starred or right.starred,
                         crop_keyframes=sorted({k["time"]: k for k in [*left.crop_keyframes, *right.crop_keyframes]}.values(), key=lambda k: k["time"]))
        self.rallies[left_index : left_index + 2] = [merged]
        self.selected_rally_index = left_index
        self._rebuild_rally_tree(left_index, seek=False)
        self._update_summary()
        self.statusBar().showMessage("Neighboring rallies merged", 3500)
        self._changed()

    def preview_selected_rally(self) -> None:
        index = self.selected_rally_index
        if not 0 <= index < len(self.rallies):
            return
        start, end, _, _ = _rally_values(self.rallies[index])
        self.player.preview_range(start, end, loop=self.loop_checkbox.isChecked())
        self.statusBar().showMessage(f"Previewing Rally {index + 1}", max(1000, int((end - start) * 1000)))

    def _on_player_position(self, seconds: float) -> None:
        self.timeline.set_position(seconds)

    # --------------------------------------------------------------- summaries
    def _update_summary(self) -> None:
        self._update_match_score()
        enabled = self._enabled_rallies()
        play_time = sum(self._rally_duration(rally) for rally in enabled)
        removed = max(0.0, self.video_duration - play_time)
        self.summary_original.setText(format_timestamp(self.video_duration))
        self.summary_count.setText(f"{len(self.rallies)} ({len(enabled)} enabled)")
        self.summary_play.setText(format_timestamp(play_time))
        self.summary_removed.setText(format_timestamp(removed))
        self.summary_estimate.setText(format_timestamp(play_time))
        if hasattr(self, "rally_count_badge"):
            self.rally_count_badge.setText(str(len(self.rallies)))

    def _update_ui_state(self) -> None:
        has_video = self.video_path is not None
        analyzing = bool(self._detection_worker and self._detection_worker.isRunning())
        exporting = bool(self._export_worker and self._export_worker.isRunning())
        busy = analyzing or exporting
        has_selection = 0 <= self.selected_rally_index < len(self.rallies)
        enabled_count = len(self._enabled_rallies())
        self.roi_button.setEnabled(has_video and not busy)
        self.court_button.setEnabled(has_video and not busy)
        self.detect_button.setEnabled(has_video and not analyzing and not exporting)
        self.settings_button.setEnabled(not busy)
        self.open_button.setEnabled(not analyzing and not exporting)
        self.open_action.setEnabled(not analyzing and not exporting)
        can_export = has_video and enabled_count > 0 and not analyzing and not exporting
        self.export_button.setEnabled(can_export)
        self.export_action.setEnabled(can_export)
        self.labels_action.setEnabled(has_video and not busy)
        self.save_labels_button.setEnabled(has_video and not busy)
        self.add_button.setEnabled(has_video and not busy)
        self.delete_button.setEnabled(has_selection and not busy)
        self.split_button.setEnabled(has_selection and not busy)
        self.merge_button.setEnabled(has_selection and len(self.rallies) > 1 and not busy)
        self.rally_tree.setEnabled(not busy)
        self.preview_button.setEnabled(has_selection)
        for control in (
            self.start_spin,
            self.end_spin,
            self.start_minus_button,
            self.start_plus_button,
            self.end_minus_button,
            self.end_plus_button,
            self.start_playhead_button,
            self.end_playhead_button,
            self.apply_times_button,
        ):
            control.setEnabled(has_selection and not busy)
        self._set_time_spin_ranges()
        self._update_workflow_state()

    # --------------------------------------------------------------- settings
    def show_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.detection_settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.detection_settings.update(dialog.values())
        self._save_detection_settings()
        self.timeline.set_threshold(float(self.detection_settings.get("rally_threshold", 0.55)))
        self.debug_action.setChecked(bool(self.detection_settings.get("debug_mode", False)))
        self.statusBar().showMessage("Detection settings saved", 4000)
        self._changed()

    def _debug_checkbox_toggled(self, enabled: bool) -> None:
        self.debug_action.blockSignals(True)
        self.debug_action.setChecked(enabled)
        self.debug_action.blockSignals(False)
        self._set_debug_mode(enabled)

    def _set_debug_mode(self, enabled: bool) -> None:
        self.detection_settings["debug_mode"] = bool(enabled)
        self.timeline.set_debug_enabled(enabled)
        self.debug_checkbox.blockSignals(True)
        self.debug_checkbox.setChecked(enabled)
        self.debug_checkbox.blockSignals(False)
        self._save_detection_settings()

    # ------------------------------------------------------------------ export
    def show_export_dialog(self) -> None:
        if not self.video_path:
            return
        enabled = self._enabled_rallies()
        if not enabled:
            QMessageBox.information(self, "No Enabled Rallies", "Enable or add at least one rally before exporting.")
            return
        source = Path(self.video_path)
        suggested = source.with_name(f"{source.stem}_roundnet_rallies.mp4")
        dialog = ExportDialog(
            str(suggested),
            bool(self.detection_settings.get("prefer_hardware", True)),
            bool(self.detection_settings.get("save_labels_after_export", False)),
            len(enabled),
            sum(self._rally_duration(rally) for rally in enabled),
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.detection_settings["prefer_hardware"] = dialog.prefer_hardware
        self.detection_settings["save_labels_after_export"] = dialog.save_labels
        self._save_detection_settings()
        self._start_export(dialog.output_path, dialog.prefer_hardware, dialog.save_labels,
                           {**self.match_settings, **dialog.export_options})

    def _start_export(self, output_path: str, prefer_hardware: bool, save_labels: bool,
                      export_options: dict[str, Any] | None = None) -> None:
        if not self.video_path or self._export_worker and self._export_worker.isRunning():
            return
        self._export_output_path = output_path
        self._save_labels_on_export = save_labels
        progress = QProgressDialog("Preparing FFmpeg export…", "Cancel Export", 0, 100, self)
        progress.setWindowTitle("Exporting Rally Video")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        self._export_progress = progress

        worker = ExportWorker(
            self.video_path,
            output_path,
            self.rallies,
            prefer_hardware,
            self,
        )
        worker.export_options = deepcopy(export_options)
        self._export_worker = worker
        progress.canceled.connect(worker.request_cancel)
        worker.progress.connect(self._on_export_progress)
        worker.succeeded.connect(self._on_export_succeeded)
        worker.failed.connect(self._on_export_failed)
        worker.cancelled.connect(self._on_export_cancelled)
        worker.finished.connect(lambda: self._export_thread_finished(worker))
        self.analysis_status.setText("Exporting rally video in the background…")
        worker.start()
        self._update_ui_state()

    def _on_export_progress(self, percent: int, message: str) -> None:
        if self._export_progress:
            self._export_progress.setValue(percent)
            self._export_progress.setLabelText(message)
        self.statusBar().showMessage(message)

    def _on_export_succeeded(self, result: Any) -> None:
        output = str(
            getattr(result, "output_path", None)
            or getattr(result, "path", None)
            or self._export_output_path
            or ""
        )
        if self._export_progress:
            self._export_progress.setValue(100)
            self._export_progress.close()
        labels_message = ""
        if self._save_labels_on_export and output:
            labels_path = str(Path(output).with_suffix(".labels.json"))
            try:
                feedback = self._write_labels(labels_path)
                feature_note = (
                    f"\nTraining features: {feedback.features_path}"
                    if getattr(feedback, "features_path", None)
                    else ""
                )
                labels_message = f"\n\nCorrection labels: {labels_path}{feature_note}"
            except Exception as exc:
                labels_message = f"\n\nThe video exported, but labels could not be saved: {exc}"
        self.analysis_status.setText("Export complete")
        box = QMessageBox(self)
        box.setWindowTitle("Export Complete")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(f"Your rally video is ready.\n\n{output}{labels_message}")
        reveal = box.addButton("Show in Finder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is reveal and output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(output).parent)))

    def _on_export_failed(self, message: str) -> None:
        if self._export_progress:
            self._export_progress.close()
        self.analysis_status.setText("Export failed")
        QMessageBox.critical(self, "Export Failed", f"FFmpeg could not create the rally video.\n\n{message}")

    def _on_export_cancelled(self) -> None:
        if self._export_progress:
            self._export_progress.close()
        self.analysis_status.setText("Export cancelled")
        self.statusBar().showMessage("Export cancelled", 4000)

    def _export_thread_finished(self, worker: ExportWorker) -> None:
        if self._export_worker is worker:
            self._export_worker = None
        self._export_progress = None
        self._update_ui_state()
        worker.deleteLater()

    # ------------------------------------------------------------ label saving
    def save_correction_labels(self) -> None:
        if not self.video_path:
            return
        source = Path(self.video_path)
        suggested = source.with_name(f"{source.stem}_roundnet_labels.json")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Correction Labels",
            str(suggested),
            "JSON Labels (*.json)",
        )
        if not path:
            return
        if Path(path).suffix.lower() != ".json":
            path += ".json"
        try:
            feedback = self._write_labels(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could Not Save Labels", str(exc))
            return
        if getattr(feedback, "features_path", None):
            message = (
                f"Saved reviewed labels and {feedback.sample_count} aligned feature samples"
            )
        else:
            message = f"Saved correction labels to {Path(path).name}"
        self.statusBar().showMessage(message, 7000)

    def _write_labels(self, path: str) -> Any:
        if not self.video_path:
            raise RuntimeError("No source video is open")
        from training.feedback import ANNOTATION_ALL_RALLIES, ANNOTATION_UNKNOWN, save_feedback

        complete = getattr(self, "complete_review_checkbox", None)
        settings = getattr(self, "analysis_settings", None) or self.detection_settings
        return save_feedback(
            path,
            source_path=self.video_path,
            duration_seconds=self.video_duration,
            roi=getattr(self, "analysis_roi", self.roi),
            initial_predictions=self.initial_detected_rallies,
            final_intervals=[r for r in self.rallies if not getattr(r, "rejected", False)],
            rejected_detections=self.rejected_detections,
            annotation_completeness=ANNOTATION_ALL_RALLIES if complete and complete.isChecked() else ANNOTATION_UNKNOWN,
            signal_data=self.detection_result,
            detection_settings={**settings, "court_context": getattr(self, "analysis_court_context", None)},
        )

    # ------------------------------------------------------------------ close
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        running = []
        if self._detection_worker and self._detection_worker.isRunning():
            running.append(self._detection_worker)
        if self._export_worker and self._export_worker.isRunning():
            running.append(self._export_worker)
        if running:
            answer = QMessageBox.question(
                self,
                "Background Work in Progress",
                "Cancel the running analysis/export and quit?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            for worker in running:
                worker.request_cancel()
            # Keep QThread objects alive if a backend needs a moment to observe
            # cancellation.  Avoid force-terminating FFmpeg/detector work.
            if any(not worker.wait(2500) for worker in running):
                self.statusBar().showMessage("Waiting for background work to stop safely…")
                event.ignore()
                return
        if not self._ensure_saved_before_leaving():
            event.ignore()
            return
        self._autosave_timer.stop()
        self.player.stop()
        event.accept()
