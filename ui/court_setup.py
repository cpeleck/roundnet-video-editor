"""Net and polygon serving-zone setup over an oriented source-video frame."""

from __future__ import annotations

import copy
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout,
                              QLabel, QMessageBox, QPushButton, QSizePolicy,
                              QVBoxLayout, QWidget)

from detection.court_context import normalize_court_context
from .roi_selector import extract_roi_frame, _array_to_qimage


class CourtCanvas(QWidget):
    changed = Signal()

    def __init__(self, image: QImage, context: dict[str, Any] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image = image
        self.context = normalize_court_context(context) or {
            "net": None, "net_radius": 0.08, "serving_zones": [],
        }
        self.mode = "net"
        self.pending: list[list[float]] = []
        self._net_dragging = False
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def image_rect(self) -> QRectF:
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def normalize(self, point: QPointF) -> list[float]:
        rect = self.image_rect()
        return [max(0, min(1, (point.x() - rect.left()) / rect.width())),
                max(0, min(1, (point.y() - rect.top()) / rect.height()))]

    def widget_point(self, point: list[float]) -> QPointF:
        rect = self.image_rect()
        return QPointF(rect.left() + point[0] * rect.width(), rect.top() + point[1] * rect.height())

    def finish_zone(self) -> bool:
        if len(self.pending) < 3 or len(self.context["serving_zones"]) >= 4:
            return False
        candidate = copy.deepcopy(self.context)
        candidate["serving_zones"].append(self.pending)
        try:
            self.context = normalize_court_context(candidate)
        except ValueError:
            return False
        self.pending = []
        self.changed.emit()
        self.update()
        return True

    def undo_point(self) -> None:
        if self.pending:
            self.pending.pop()
        elif self.context["serving_zones"]:
            self.context["serving_zones"].pop()
        self.changed.emit()
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self.image_rect().contains(event.position()):
            return super().mousePressEvent(event)
        point = self.normalize(event.position())
        if self.mode == "net":
            self.context["net"] = point
            self._net_dragging = True
        elif len(self.context["serving_zones"]) < 4 and len(self.pending) < 12:
            self.pending.append(point)
        self.changed.emit()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._net_dragging and self.context["net"] is not None:
            from math import hypot
            point = self.normalize(event.position())
            net = self.context["net"]
            radius = hypot(point[0] - net[0], (point[1] - net[1]) * self.image.height() / self.image.width())
            self.context["net_radius"] = max(0.005, min(0.5, radius))
            self.changed.emit()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._net_dragging = False

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.mode == "zone":
            self.finish_zone()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#090c12"))
        painter.drawImage(self.image_rect(), self.image)
        colors = ["#58d68d", "#58a6ff", "#da9bff", "#ffae57"]
        for index, zone in enumerate(self.context["serving_zones"]):
            color = QColor(colors[index])
            painter.setPen(QPen(color, 2))
            color.setAlpha(45)
            painter.setBrush(color)
            polygon = QPolygonF([self.widget_point(point) for point in zone])
            painter.drawPolygon(polygon)
            painter.drawText(polygon.boundingRect(), Qt.AlignmentFlag.AlignCenter, f"Serve zone {index + 1}")
        if self.pending:
            painter.setPen(QPen(QColor("#ffffff"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(QPolygonF([self.widget_point(point) for point in self.pending]))
            for point in self.pending:
                painter.drawEllipse(self.widget_point(point), 4, 4)
        if self.context["net"] is not None:
            center = self.widget_point(self.context["net"])
            radius = self.context["net_radius"] * self.image_rect().width()
            painter.setPen(QPen(QColor("#ffd33d"), 2))
            painter.setBrush(QColor(255, 211, 61, 30))
            painter.drawEllipse(center, radius, radius)
            painter.drawLine(center + QPointF(-7, 0), center + QPointF(7, 0))
            painter.drawLine(center + QPointF(0, -7), center + QPointF(0, 7))
            painter.drawText(center + QPointF(10, -10), "Net / collection area")
        painter.end()


class CourtSetupDialog(QDialog):
    """Output via ``court_context``: normalized net + up to four polygons."""

    def __init__(self, video_path: str, court_context: dict[str, Any] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.setWindowTitle("Court & Serving Zones")
        self.resize(1040, 760)
        self.canvas = CourtCanvas(extract_roi_frame(video_path), court_context)
        self.canvas.changed.connect(self._refresh)
        title = QLabel("Mark the net, then the areas where players usually stand to serve.")
        title.setWordWrap(True)
        instructions = QLabel(
            "Net: click its center and drag a circle around the collection area. "
            "Serving zone: click 3–12 corners and finish the zone. Up to four optional zones. "
            "These cues increase confidence; players do not have to start inside a zone."
        )
        instructions.setWordWrap(True)
        self.mode = QComboBox()
        self.mode.addItem("Mark net", "net")
        self.mode.addItem("Draw serving zone", "zone")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.finish_button = QPushButton("Finish Zone")
        self.finish_button.clicked.connect(self._finish_zone)
        self.undo_button = QPushButton("Undo Corner / Zone")
        self.undo_button.clicked.connect(self.canvas.undo_point)
        self.clear_button = QPushButton("Clear Setup")
        self.clear_button.clicked.connect(self._clear)
        self.timestamp = QDoubleSpinBox()
        self.timestamp.setRange(0, 86_400)
        self.timestamp.setDecimals(1)
        self.timestamp.setSuffix(" s")
        self.preview_button = QPushButton("Show Frame")
        self.preview_button.clicked.connect(self._load_frame)
        controls = QHBoxLayout()
        for widget in (self.mode, self.finish_button, self.undo_button, self.clear_button):
            controls.addWidget(widget)
        controls.addStretch(1)
        controls.addWidget(self.timestamp)
        controls.addWidget(self.preview_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save Court Setup")
        save.clicked.connect(self._save)
        buttons = QHBoxLayout()
        buttons.addWidget(self.status, 1)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(instructions)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(buttons)
        self._refresh()

    @property
    def court_context(self) -> dict[str, Any] | None:
        context = self.canvas.context
        if context["net"] is None and not context["serving_zones"]:
            return None
        return copy.deepcopy(context)

    def _mode_changed(self) -> None:
        self.canvas.mode = self.mode.currentData()
        self._refresh()

    def _refresh(self) -> None:
        count = len(self.canvas.context["serving_zones"])
        self.status.setText(f"Net: {'marked' if self.canvas.context['net'] else 'optional'} · {count}/4 serving zones · {len(self.canvas.pending)} pending corners")
        self.finish_button.setEnabled(len(self.canvas.pending) >= 3 and count < 4)

    def _finish_zone(self) -> None:
        if not self.canvas.finish_zone():
            QMessageBox.information(self, "Draw a Zone", "Add at least three corners enclosing the serving area.")

    def _clear(self) -> None:
        self.canvas.context = {"net": None, "net_radius": 0.08, "serving_zones": []}
        self.canvas.pending = []
        self.canvas.update()
        self._refresh()

    def _load_frame(self) -> None:
        try:
            from video.video_reader import extract_frame
            self.canvas.image = _array_to_qimage(extract_frame(
                self.video_path, self.timestamp.value(), rgb=True, max_size=(1280, 800)))
            self.canvas.update()
        except Exception as exc:
            QMessageBox.warning(self, "Could Not Load Frame", str(exc))

    def _save(self) -> None:
        if self.canvas.pending:
            if not self.canvas.finish_zone():
                QMessageBox.information(self, "Finish Serving Zone", "Finish the current zone or undo its pending corners before saving.")
                return
        self.accept()


__all__ = ["CourtSetupDialog", "CourtCanvas"]
