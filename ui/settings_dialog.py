"""Advanced detection and export preferences."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


DEFAULT_SETTINGS: dict[str, Any] = {
    # Detector-facing defaults mirror config.DetectionSettings exactly.  The
    # sensitivity values are multipliers (1.0 is neutral), not 0..1 scores.
    "motion_sensitivity": 1.0,
    "audio_sensitivity": 1.0,
    "serve_sensitivity": 1.0,
    "motion_weight": 0.20,
    "roi_motion_weight": 0.22,
    "motion_spread_weight": 0.20,
    "audio_weight": 0.10,
    "temporal_weight": 0.18,
    "serve_weight": 0.10,
    "retrieval_filter_enabled": True,
    "player_tracking_enabled": True,
    "person_detection_interval": 0.75,
    "pose_model_path": "",
    "court_context_strength": 0.15,
    "rally_threshold": 0.55,
    "end_threshold": 0.38,
    "start_active_duration": 0.375,
    "end_inactive_duration": 1.00,
    "min_rally_duration": 1.5,
    "max_rally_duration": 30.0,
    "pre_roll": 0.75,
    "post_roll": 1.25,
    "merge_gap": 1.00,
    "analysis_fps": 8.0,
    "downscale_width": 640,
    "debug_mode": False,
    "prefer_hardware": True,
    "save_labels_after_export": False,
}


PRESETS: dict[str, dict[str, Any]] = {
    "Indoor Roundnet": {
        "motion_sensitivity": 1.0,
        "audio_sensitivity": 1.15,
        "serve_sensitivity": 1.1,
        "rally_threshold": 0.54,
        "audio_weight": 0.16,
        "roi_motion_weight": 0.20,
        "serve_weight": 0.12,
    },
    "Outdoor Roundnet": {
        "motion_sensitivity": 1.05,
        "audio_sensitivity": 0.85,
        "serve_sensitivity": 1.0,
        "rally_threshold": 0.58,
        "audio_weight": 0.08,
        "roi_motion_weight": 0.24,
        "motion_spread_weight": 0.23,
    },
    "Tournament Sideline Camera": {
        "motion_sensitivity": 1.1,
        "audio_sensitivity": 0.95,
        "serve_sensitivity": 1.1,
        "rally_threshold": 0.56,
        "roi_motion_weight": 0.24,
        "motion_spread_weight": 0.22,
        "merge_gap": 0.8,
    },
    "High Camera Angle": {
        "motion_sensitivity": 1.25,
        "audio_sensitivity": 1.0,
        "serve_sensitivity": 1.15,
        "rally_threshold": 0.54,
        "motion_weight": 0.22,
        "roi_motion_weight": 0.24,
        "motion_spread_weight": 0.22,
    },
}


class SettingsDialog(QDialog):
    """Editable, resettable set of all MVP detector settings."""

    def __init__(self, values: Mapping[str, Any] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Detection Settings")
        self.resize(620, 690)
        self._initial = deepcopy(DEFAULT_SETTINGS)
        self._initial.update(dict(values or {}))
        self._controls: dict[str, Any] = {}

        intro = QLabel(
            "Tune detection for this camera and venue. Defaults are conservative; the debug timeline helps "
            "identify which signal needs adjustment."
        )
        intro.setWordWrap(True)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Custom / current settings", None)
        for name in PRESETS:
            self.preset_combo.addItem(name, name)
        self.preset_combo.currentIndexChanged.connect(self._apply_selected_preset)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Starting preset:"))
        preset_row.addWidget(self.preset_combo, 1)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.addWidget(self._build_signal_group())
        content_layout.addWidget(self._build_timing_group())
        content_layout.addWidget(self._build_performance_group())
        content_layout.addWidget(self._build_display_group())
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)

        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        button_row = QHBoxLayout()
        button_row.addWidget(self.reset_button)
        button_row.addStretch(1)
        button_row.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(preset_row)
        layout.addWidget(scroll, 1)
        layout.addLayout(button_row)
        self.set_values(self._initial)

    def _double_control(
        self,
        key: str,
        minimum: float,
        maximum: float,
        step: float,
        *,
        decimals: int = 2,
        suffix: str = "",
        tooltip: str = "",
    ) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setSingleStep(step)
        control.setDecimals(decimals)
        control.setSuffix(suffix)
        control.setKeyboardTracking(False)
        if tooltip:
            control.setToolTip(tooltip)
        self._controls[key] = control
        return control

    def _int_control(
        self,
        key: str,
        minimum: int,
        maximum: int,
        step: int,
        *,
        suffix: str = "",
        tooltip: str = "",
    ) -> QSpinBox:
        control = QSpinBox()
        control.setRange(minimum, maximum)
        control.setSingleStep(step)
        control.setSuffix(suffix)
        if tooltip:
            control.setToolTip(tooltip)
        self._controls[key] = control
        return control

    def _build_signal_group(self) -> QGroupBox:
        group = QGroupBox("Activity scoring")
        form = QFormLayout(group)
        form.addRow(
            "Motion sensitivity",
            self._double_control(
                "motion_sensitivity", 0.1, 2.0, 0.05,
                tooltip="Higher values make smaller or slower movements count more strongly.",
            ),
        )
        form.addRow(
            "Audio sensitivity",
            self._double_control(
                "audio_sensitivity", 0.1, 2.0, 0.05,
                tooltip="Higher values increase response to short impact-like sounds.",
            ),
        )
        form.addRow(
            "Serve sensitivity",
            self._double_control(
                "serve_sensitivity", 0.1, 2.0, 0.05,
                tooltip="Higher values give more credit to quiet-to-active serve-like starts.",
            ),
        )
        form.addRow("Overall motion weight", self._double_control("motion_weight", 0.0, 1.0, 0.05))
        form.addRow("ROI motion weight", self._double_control("roi_motion_weight", 0.0, 1.0, 0.05))
        form.addRow(
            "Multi-player spread weight",
            self._double_control(
                "motion_spread_weight", 0.0, 1.0, 0.05,
                tooltip="Rewards motion distributed around the playing area instead of one compact mover.",
            ),
        )
        form.addRow("Audio weight", self._double_control("audio_weight", 0.0, 1.0, 0.05))
        form.addRow("Temporal activity weight", self._double_control("temporal_weight", 0.0, 1.0, 0.05))
        form.addRow(
            "Serve-start weight",
            self._double_control(
                "serve_weight", 0.0, 1.0, 0.05,
                tooltip="Boosts confidence briefly after a likely serve; it is not a required gate.",
            ),
        )
        retrieval_filter = QCheckBox(
            "Leave compact, no-serve candidates unchecked for review"
        )
        retrieval_filter.setToolTip(
            "Conservatively excludes likely ball collection from export while keeping it visible."
        )
        self._controls["retrieval_filter_enabled"] = retrieval_filter
        form.addRow("Ball-retrieval filter", retrieval_filter)
        people = QCheckBox("Use local player tracking and body-pose cues")
        people.setToolTip("Uses macOS Vision where available, with OpenCV people detection as fallback. No uploads.")
        self._controls["player_tracking_enabled"] = people
        form.addRow("Player analysis", people)
        form.addRow("Player detection interval", self._double_control(
            "person_detection_interval", 0.1, 5.0, 0.1, suffix=" s"))
        form.addRow("Court-context influence", self._double_control(
            "court_context_strength", 0.0, 1.0, 0.05,
            tooltip="Uses your net and serving zones to interpret player movement. Not a guaranteed serve classifier."))
        pose_path = QLineEdit()
        pose_path.setPlaceholderText("Optional compatible YOLO pose ONNX model; blank uses local defaults")
        self._controls["pose_model_path"] = pose_path
        form.addRow("Advanced pose model", pose_path)
        form.addRow(
            "Rally start threshold",
            self._double_control("rally_threshold", 0.05, 0.99, 0.01),
        )
        form.addRow(
            "Rally end threshold",
            self._double_control("end_threshold", 0.01, 0.95, 0.01),
        )
        return group

    def _build_timing_group(self) -> QGroupBox:
        group = QGroupBox("Rally timing")
        form = QFormLayout(group)
        form.addRow(
            "Activity needed to start",
            self._double_control("start_active_duration", 0.05, 5.0, 0.05, suffix=" s"),
        )
        form.addRow(
            "Inactivity needed to end",
            self._double_control("end_inactive_duration", 0.1, 10.0, 0.1, suffix=" s"),
        )
        form.addRow(
            "Minimum rally duration",
            self._double_control("min_rally_duration", 0.1, 30.0, 0.25, suffix=" s"),
        )
        form.addRow(
            "Maximum rally duration",
            self._double_control("max_rally_duration", 1.0, 180.0, 1.0, suffix=" s", decimals=1),
        )
        form.addRow("Pre-roll", self._double_control("pre_roll", 0.0, 10.0, 0.05, suffix=" s"))
        form.addRow("Post-roll", self._double_control("post_roll", 0.0, 10.0, 0.05, suffix=" s"))
        form.addRow(
            "Merge gaps shorter than",
            self._double_control("merge_gap", 0.0, 10.0, 0.05, suffix=" s"),
        )
        return group

    def _build_performance_group(self) -> QGroupBox:
        group = QGroupBox("Analysis and export")
        form = QFormLayout(group)
        form.addRow(
            "Analysis frame rate",
            self._double_control(
                "analysis_fps", 1.0, 30.0, 1.0, suffix=" FPS", decimals=1,
                tooltip="Higher values can detect briefer motion, but take longer to analyze.",
            ),
        )
        form.addRow(
            "Analysis image width",
            self._int_control(
                "downscale_width", 240, 1920, 40, suffix=" px",
                tooltip="Only analysis is downscaled; export always uses the original video.",
            ),
        )
        hardware = QCheckBox("Use Apple VideoToolbox when available")
        hardware.setToolTip("Falls back to software encoding when hardware encoding is unavailable.")
        self._controls["prefer_hardware"] = hardware
        form.addRow("Hardware acceleration", hardware)
        labels = QCheckBox("Offer to save corrected rally labels after export")
        self._controls["save_labels_after_export"] = labels
        form.addRow("Training labels", labels)
        return group

    def _build_display_group(self) -> QGroupBox:
        group = QGroupBox("Diagnostics")
        form = QFormLayout(group)
        debug = QCheckBox("Show motion, spread, serve, audio, score, and threshold graphs")
        self._controls["debug_mode"] = debug
        form.addRow("Debug timeline", debug)
        return group

    def values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, control in self._controls.items():
            if isinstance(control, QCheckBox):
                values[key] = control.isChecked()
            elif isinstance(control, QLineEdit):
                values[key] = control.text().strip()
            else:
                values[key] = control.value()
        return values

    def set_values(self, values: Mapping[str, Any]) -> None:
        merged = deepcopy(DEFAULT_SETTINGS)
        merged.update(dict(values))
        for key, control in self._controls.items():
            value = merged.get(key, DEFAULT_SETTINGS.get(key))
            if isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            elif isinstance(control, QLineEdit):
                control.setText(str(value or ""))
            elif value is not None:
                control.setValue(value)

    def reset_to_defaults(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        self.set_values(DEFAULT_SETTINGS)

    def _apply_selected_preset(self) -> None:
        name = self.preset_combo.currentData()
        if not name:
            return
        values = deepcopy(DEFAULT_SETTINGS)
        values.update(PRESETS[str(name)])
        self.set_values(values)

    def _validate_and_accept(self) -> None:
        values = self.values()
        if values["end_threshold"] > values["rally_threshold"]:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Invalid Thresholds",
                "The rally end threshold cannot be greater than the rally start threshold.",
            )
            return
        if values["max_rally_duration"] < values["min_rally_duration"]:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Invalid Rally Durations",
                "Maximum rally duration must be at least the minimum rally duration.",
            )
            return
        weights = (
            "motion_weight",
            "roi_motion_weight",
            "motion_spread_weight",
            "audio_weight",
            "temporal_weight",
            "serve_weight",
        )
        if not any(values[key] > 0 for key in weights):
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Invalid Signal Weights",
                "At least one activity-signal weight must be greater than zero.",
            )
            return
        if values["start_active_duration"] <= 0:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Invalid Start Duration",
                "Activity needed to start must be greater than zero.",
            )
            return
        self.accept()
