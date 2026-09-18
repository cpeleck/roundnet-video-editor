"""First-frame region-of-interest selection and reusable ROI presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)


NormalizedROI = tuple[float, float, float, float]


def _array_to_qimage(frame: Any, *, rgb: bool = True) -> QImage:
    """Convert an OpenCV/numpy frame to an owning QImage."""

    if isinstance(frame, QImage):
        return frame.copy()
    if frame is None or not hasattr(frame, "shape"):
        raise ValueError("Frame extractor returned no image")
    if len(frame.shape) == 2:
        height, width = frame.shape
        image = QImage(frame.data, width, height, int(frame.strides[0]), QImage.Format.Format_Grayscale8)
        return image.copy()
    height, width, channels = frame.shape
    if channels == 4:
        # ARGB32 is stored BGRA on little-endian systems, matching OpenCV's
        # four-channel byte order.  Qt does not expose a Format_BGRA8888 enum.
        image_format = QImage.Format.Format_RGBA8888 if rgb else QImage.Format.Format_ARGB32
    elif channels == 3:
        image_format = QImage.Format.Format_RGB888 if rgb else QImage.Format.Format_BGR888
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")
    image = QImage(frame.data, width, height, int(frame.strides[0]), image_format)
    return image.copy()


def extract_roi_frame(video_path: str) -> QImage:
    """Extract a representative early frame using the video layer or OpenCV."""

    try:
        from video.video_reader import extract_thumbnail

        # The video layer intentionally samples a moment near the beginning to
        # avoid the black first frames common in phone recordings.
        frame = extract_thumbnail(video_path, max_size=(1280, 800), rgb=True)
        return _array_to_qimage(frame, rgb=True)
    except Exception as backend_error:
        try:
            import cv2

            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                raise RuntimeError("OpenCV could not open the video")
            ok, frame = capture.read()
            capture.release()
            if not ok or frame is None:
                raise RuntimeError("OpenCV could not decode the first frame")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return _array_to_qimage(frame, rgb=True)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Could not extract a frame for ROI selection. {backend_error}; {fallback_error}"
            ) from fallback_error


class ROICanvas(QWidget):
    """Aspect-correct image canvas that draws a normalized rectangle."""

    roi_changed = Signal(object)

    def __init__(self, image: QImage, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = image
        self._roi: NormalizedROI | None = None
        self._drag_start: QPointF | None = None
        self._drag_current: QPointF | None = None
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    @property
    def roi(self) -> NormalizedROI | None:
        return self._roi

    def set_roi(self, roi: NormalizedROI | None) -> None:
        if roi is None:
            self._roi = None
        else:
            x, y, width, height = (float(value) for value in roi)
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            width = max(0.0, min(1.0 - x, width))
            height = max(0.0, min(1.0 - y, height))
            self._roi = (x, y, width, height)
        self.roi_changed.emit(self._roi)
        self.update()

    def _image_rect(self) -> QRectF:
        if self._image.isNull():
            return QRectF(self.rect())
        scale = min(self.width() / self._image.width(), self.height() / self._image.height())
        width = self._image.width() * scale
        height = self._image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _normalized_point(self, point: QPointF) -> QPointF:
        image_rect = self._image_rect()
        x = max(image_rect.left(), min(image_rect.right(), point.x()))
        y = max(image_rect.top(), min(image_rect.bottom(), point.y()))
        return QPointF(
            (x - image_rect.left()) / max(1.0, image_rect.width()),
            (y - image_rect.top()) / max(1.0, image_rect.height()),
        )

    def _widget_roi_rect(self) -> QRectF | None:
        if self._roi is None:
            return None
        image_rect = self._image_rect()
        x, y, width, height = self._roi
        return QRectF(
            image_rect.left() + x * image_rect.width(),
            image_rect.top() + y * image_rect.height(),
            width * image_rect.width(),
            height * image_rect.height(),
        )

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#070a0f"))
        image_rect = self._image_rect()
        painter.drawImage(image_rect, self._image)
        roi_rect = self._widget_roi_rect()
        if roi_rect is None:
            painter.setPen(QColor("#f0f6fc"))
            painter.drawText(image_rect, Qt.AlignmentFlag.AlignCenter, "Drag around the net and active playing area")
            return

        # Darken everything outside the selection without obscuring the ROI.
        shade = QPainterPath()
        shade.addRect(image_rect)
        cutout = QPainterPath()
        cutout.addRect(roi_rect)
        shade = shade.subtracted(cutout)
        painter.fillPath(shade, QColor(0, 0, 0, 120))
        painter.setBrush(QColor(46, 160, 67, 35))
        painter.setPen(QPen(QColor("#58d68d"), 3))
        painter.drawRect(roi_rect)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._image_rect().contains(event.position()):
            point = self._normalized_point(event.position())
            self._drag_start = point
            self._drag_current = point
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._drag_current = self._normalized_point(event.position())
            self._update_drag_roi()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            self._drag_current = self._normalized_point(event.position())
            self._update_drag_roi()
            self._drag_start = None
            self._drag_current = None
            if self._roi and (self._roi[2] < 0.01 or self._roi[3] < 0.01):
                self.set_roi(None)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _update_drag_roi(self) -> None:
        if self._drag_start is None or self._drag_current is None:
            return
        left = min(self._drag_start.x(), self._drag_current.x())
        top = min(self._drag_start.y(), self._drag_current.y())
        right = max(self._drag_start.x(), self._drag_current.x())
        bottom = max(self._drag_start.y(), self._drag_current.y())
        self._roi = (left, top, right - left, bottom - top)
        self.roi_changed.emit(self._roi)
        self.update()


class ROISelectorDialog(QDialog):
    """Modal ROI editor with persistent named camera presets."""

    def __init__(
        self,
        video_path: str,
        current_roi: NormalizedROI | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Playing Area")
        self.resize(960, 680)
        self.setModal(True)
        self._settings = QSettings("RoundnetEditor", "RoundnetEditor")
        self._video_path = video_path

        image = extract_roi_frame(video_path)
        self.canvas = ROICanvas(image)
        self.canvas.set_roi(current_roi)
        self.canvas.roi_changed.connect(self._update_roi_label)

        title = QLabel("Draw a rectangle around the net and the area where players rally.")
        title.setObjectName("dialogTitle")
        detail = QLabel(
            "Motion inside this area receives extra weight. Include normal player movement around the net, "
            "but exclude spectators and unrelated courts when practical."
        )
        detail.setWordWrap(True)

        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(210)
        self.preset_combo.currentIndexChanged.connect(self._load_selected_preset)
        self.save_preset_button = QPushButton("Save as Preset…")
        self.save_preset_button.clicked.connect(self._save_preset)
        self.delete_preset_button = QPushButton("Delete Preset")
        self.delete_preset_button.clicked.connect(self._delete_preset)
        self._populate_presets()

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Camera preset:"))
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(self.save_preset_button)
        preset_row.addWidget(self.delete_preset_button)
        preset_row.addStretch(1)

        self.roi_label = QLabel()
        self.roi_label.setStyleSheet("color: #8b949e;")
        self.reset_button = QPushButton("Reset ROI")
        self.reset_button.clicked.connect(lambda: self.canvas.set_roi(None))
        self.full_frame_button = QPushButton("Use Full Frame")
        self.full_frame_button.clicked.connect(lambda: self.canvas.set_roi((0.0, 0.0, 1.0, 1.0)))
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.save_button = QPushButton("Save ROI")
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self._accept_if_valid)

        button_row = QHBoxLayout()
        button_row.addWidget(self.roi_label, 1)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.full_frame_button)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(detail)
        layout.addLayout(preset_row)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(button_row)
        self._update_roi_label(self.canvas.roi)

    @property
    def selected_roi(self) -> NormalizedROI | None:
        return self.canvas.roi

    def _read_presets(self) -> dict[str, NormalizedROI]:
        raw = self._settings.value("roi/presets", "{}")
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        presets: dict[str, NormalizedROI] = {}
        if isinstance(payload, dict):
            for name, roi in payload.items():
                if isinstance(roi, list) and len(roi) == 4:
                    try:
                        presets[str(name)] = tuple(float(value) for value in roi)  # type: ignore[assignment]
                    except (TypeError, ValueError):
                        continue
        return presets

    def _write_presets(self, presets: dict[str, NormalizedROI]) -> None:
        self._settings.setValue("roi/presets", json.dumps(presets, sort_keys=True))

    def _populate_presets(self, selected_name: str | None = None) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Choose a saved preset…", None)
        for name in sorted(self._read_presets(), key=str.casefold):
            self.preset_combo.addItem(name, name)
        if selected_name:
            index = self.preset_combo.findData(selected_name)
            if index >= 0:
                self.preset_combo.setCurrentIndex(index)
        self.preset_combo.blockSignals(False)
        self.delete_preset_button.setEnabled(self.preset_combo.currentData() is not None)

    def _load_selected_preset(self) -> None:
        name = self.preset_combo.currentData()
        self.delete_preset_button.setEnabled(name is not None)
        if name is None:
            return
        roi = self._read_presets().get(str(name))
        if roi:
            self.canvas.set_roi(roi)

    def _save_preset(self) -> None:
        if self.canvas.roi is None:
            QMessageBox.information(self, "Draw a Playing Area", "Draw an ROI before saving a preset.")
            return
        name, accepted = QInputDialog.getText(self, "Save Camera Preset", "Preset name:")
        name = name.strip()
        if not accepted or not name:
            return
        presets = self._read_presets()
        presets[name] = self.canvas.roi
        self._write_presets(presets)
        self._populate_presets(name)

    def _delete_preset(self) -> None:
        name = self.preset_combo.currentData()
        if name is None:
            return
        answer = QMessageBox.question(self, "Delete Preset", f'Delete the preset "{name}"?')
        if answer != QMessageBox.StandardButton.Yes:
            return
        presets = self._read_presets()
        presets.pop(str(name), None)
        self._write_presets(presets)
        self._populate_presets()

    def _update_roi_label(self, roi: NormalizedROI | None) -> None:
        self.save_button.setEnabled(roi is not None)
        self.save_preset_button.setEnabled(roi is not None)
        if roi is None:
            self.roi_label.setText("No playing area selected")
            return
        x, y, width, height = roi
        self.roi_label.setText(
            f"ROI: x {x * 100:.1f}% · y {y * 100:.1f}% · width {width * 100:.1f}% · height {height * 100:.1f}%"
        )

    def _accept_if_valid(self) -> None:
        if self.canvas.roi is None:
            QMessageBox.information(self, "Draw a Playing Area", "Draw an ROI or choose Use Full Frame.")
            return
        self._settings.setValue("roi/last", json.dumps(self.canvas.roi))
        self.accept()


def load_last_roi() -> NormalizedROI | None:
    """Load the most recently accepted normalized ROI, if any."""

    raw = QSettings("RoundnetEditor", "RoundnetEditor").value("roi/last")
    if not raw:
        return None
    try:
        values = json.loads(str(raw))
        if isinstance(values, list) and len(values) == 4:
            return tuple(float(value) for value in values)  # type: ignore[return-value]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return None
