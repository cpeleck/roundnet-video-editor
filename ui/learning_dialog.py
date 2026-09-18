"""Train and inspect a local profile without blocking the review window."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from training.calibration import load_profile, train_from_feedback


class _TrainingThread(QThread):
    status = Signal(str)
    trained = Signal(object)
    failed = Signal(str)

    def __init__(self, paths: list[str], destination: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.paths = paths
        self.destination = destination

    def run(self) -> None:
        try:
            report = train_from_feedback(
                self.paths, self.destination, progress=self.status.emit,
                cancelled=self.isInterruptionRequested,
            )
            self.trained.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))


class LearningDialog(QDialog):
    """Explicit file selection, training, diagnostics, and profile activation.

    ``profile_path`` is set only when the user presses Use Profile, or chooses
    an existing valid profile and then presses Use Profile. Saving a trained
    model by itself never activates it.
    """

    def __init__(self, parent: QWidget | None = None, feedback_paths: Sequence[str | Path] = ()) -> None:
        super().__init__(parent)
        self.setWindowTitle("Learn from reviewed games")
        self.resize(780, 680)
        self.profile_path: str | None = None
        self._candidate_path: str | None = None
        self._thread: _TrainingThread | None = None
        self._close_after_training = False

        intro = QLabel(
            "Select correction JSON files saved after reviewing games. Training uses only your local "
            "signals and labels. At least three different original recordings and examples of rallies "
            "and non-rallies are required. Each game is held out in turn to measure performance on "
            "footage the model has not seen. Pending clips are ignored unless you confirm the whole "
            "recording is reviewed. This is supervised learning from corrections, not reinforcement learning."
        )
        intro.setWordWrap(True)
        self.files_list = QListWidget()
        self.files_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._add_paths([str(path) for path in feedback_paths])
        self.add_button = QPushButton("Add correction files…")
        self.remove_button = QPushButton("Remove selected")
        self.train_button = QPushButton("Train and save profile…")
        self.load_button = QPushButton("Open existing profile…")
        self.add_button.clicked.connect(self._choose_files)
        self.remove_button.clicked.connect(self._remove_files)
        self.train_button.clicked.connect(self._train)
        self.load_button.clicked.connect(self._load)
        file_row = QHBoxLayout()
        file_row.addWidget(self.add_button)
        file_row.addWidget(self.remove_button)
        file_row.addStretch()
        train_row = QHBoxLayout()
        train_row.addWidget(self.train_button)
        train_row.addWidget(self.load_button)
        train_row.addStretch()

        self.status_label = QLabel("No model has been trained or selected.")
        self.status_label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.report_view = QPlainTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setPlaceholderText(
            "Validation results will compare the original detector and learned profile. "
            "This application includes no model trained on your footage until you explicitly train one."
        )
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.use_button = self.buttons.addButton("Use Profile", QDialogButtonBox.ButtonRole.AcceptRole)
        self.use_button.setEnabled(False)
        self.buttons.rejected.connect(self.reject)
        self.use_button.clicked.connect(self._use_profile)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.files_list, 1)
        layout.addLayout(file_row)
        layout.addLayout(train_row)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.report_view, 2)
        layout.addWidget(self.buttons)

    def _add_paths(self, paths: Sequence[str]) -> None:
        existing = {self.files_list.item(i).text() for i in range(self.files_list.count())}
        for path in paths:
            value = str(Path(path).expanduser().resolve())
            if value not in existing:
                self.files_list.addItem(value)
                existing.add(value)

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose reviewed correction files", "", "Correction JSON (*.json)")
        self._add_paths(paths)

    def _remove_files(self) -> None:
        for item in self.files_list.selectedItems():
            self.files_list.takeItem(self.files_list.row(item))

    def _train(self) -> None:
        paths = [self.files_list.item(i).text() for i in range(self.files_list.count())]
        if not paths:
            QMessageBox.information(self, "Add reviewed games", "Choose correction JSON files before training.")
            return
        destination, _ = QFileDialog.getSaveFileName(
            self, "Save learned profile", str(Path(paths[0]).parent / "roundnet-profile.json"), "Model profile (*.json)",
        )
        if not destination:
            return
        if not destination.lower().endswith(".json"):
            destination += ".json"
        if str(Path(destination).resolve()) in paths:
            QMessageBox.warning(self, "Choose a different filename", "Save the profile separately from your correction files.")
            return
        self._candidate_path = None
        self.use_button.setEnabled(False)
        self.report_view.clear()
        self._set_busy(True)
        self.status_label.setText("Loading reviewed games…")
        self._thread = _TrainingThread(paths, destination, self)
        self._thread.status.connect(self.status_label.setText)
        self._thread.trained.connect(self._trained)
        self._thread.failed.connect(self._failed)
        self._thread.finished.connect(self._finished)
        self._thread.start()

    def _trained(self, report: dict[str, Any]) -> None:
        self._candidate_path = report["profile_path"]
        self.report_view.setPlainText(_format_report(report))
        self.status_label.setText("Profile saved. Review the results, then select Use Profile to enable it.")

    def _failed(self, message: str) -> None:
        self.status_label.setText(message)
        self.report_view.setPlainText(message)

    def _finished(self) -> None:
        self._set_busy(False)
        self.use_button.setEnabled(self._candidate_path is not None)
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        if self._close_after_training:
            super().reject()

    def _load(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Open local profile", "", "Model profile (*.json)")
        if not filename:
            return
        try:
            profile = load_profile(filename)
            report_text = _format_report(profile.get("report", {}))
        except Exception as exc:
            QMessageBox.warning(self, "Cannot load profile", str(exc))
            return
        self._candidate_path = filename
        self.report_view.setPlainText(report_text)
        self.status_label.setText(f"Selected {Path(filename).name}. Select Use Profile to enable it.")
        self.use_button.setEnabled(True)

    def _use_profile(self) -> None:
        if self._candidate_path and not self._busy():
            self.profile_path = self._candidate_path
            self.accept()

    def _busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.add_button, self.remove_button, self.train_button, self.load_button, self.files_list):
            widget.setEnabled(not busy)
        self.progress.setVisible(busy)

    def reject(self) -> None:
        if self._busy():
            self._close_after_training = True
            self._thread.requestInterruption()
            self.status_label.setText("Cancelling training…")
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._busy():
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)


def _format_report(report: dict[str, Any]) -> str:
    if not report:
        return "This profile has no saved validation report. Its accuracy has not been established here."
    lines = [
        f"Whole-recording validation: {report.get('recording_count', '?')} original games",
        f"Reviewed samples: {report.get('positive_samples', '?')} rally / {report.get('negative_samples', '?')} non-rally",
        "Each validation game was excluded from fitting and normalization.", "",
    ]
    for key, label in (("heuristic_events", "Original detector"), ("learned_events", "Learned profile")):
        result = report.get(key)
        lines.append(label)
        if result:
            lines.extend([
                f"  Event precision: {_percent(result.get('precision'))}   Recall: {_percent(result.get('recall'))}",
                f"  False detections/hour: {_number(result.get('false_positives_per_hour'))}",
                f"  Mean start/end error: {_number(result.get('mean_start_error_seconds'))}s / {_number(result.get('mean_end_error_seconds'))}s",
                f"  Matched: {result.get('true_positives', 0)}   False: {result.get('false_positives', 0)}   Missed: {result.get('false_negatives', 0)}",
            ])
        else:
            lines.append("  Event metrics unavailable without fully reviewed games.")
        lines.append("")
    lines.append("Sample-level balanced accuracy (reviewed interiors only):")
    lines.append(f"  Original: {_percent(report.get('heuristic_samples', {}).get('balanced_accuracy'))}")
    lines.append(f"  Learned: {_percent(report.get('learned_samples', {}).get('balanced_accuracy'))}")
    lines.extend(["", "Boundary errors refer to export cuts, including viewing padding, not exact ball-contact times.",
                  "Scores are class-balanced activity estimates; results may change with camera, lighting, or venue."])
    for warning in report.get("warnings", []):
        lines.extend(["", f"Note: {warning}"])
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2f}"


__all__ = ["LearningDialog"]
