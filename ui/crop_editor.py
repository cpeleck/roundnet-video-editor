"""Click-to-place crop keyframes on original video frames."""

from copy import deepcopy

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget)

from .roi_selector import _array_to_qimage


class CropCanvas(QWidget):
    center_changed = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.center = (.5, .5)
        self.ratio = "9:16"
        self.setMinimumSize(640, 360)

    def image_rect(self):
        if self.image.isNull():
            return QRectF(self.rect())
        scale = min(self.width()/self.image.width(), self.height()/self.image.height())
        w, h = self.image.width()*scale, self.image.height()*scale
        return QRectF((self.width()-w)/2, (self.height()-h)/2, w, h)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0d1117"))
        area = self.image_rect()
        painter.drawImage(area, self.image)
        if not self.image.isNull():
            from video.presentation import crop_dimensions
            cw, ch = crop_dimensions(self.image.width(), self.image.height(), self.ratio)
            w, h = cw/self.image.width()*area.width(), ch/self.image.height()*area.height()
            x = min(area.right()-w, max(area.left(), area.left()+self.center[0]*area.width()-w/2))
            y = min(area.bottom()-h, max(area.top(), area.top()+self.center[1]*area.height()-h/2))
            painter.setPen(QPen(QColor("#39d0c5"), 3))
            painter.drawRect(QRectF(x, y, w, h))

    def mousePressEvent(self, event):
        area = self.image_rect()
        if event.button() == Qt.MouseButton.LeftButton and area.contains(event.position()):
            self.center = ((event.position().x()-area.left())/area.width(),
                           (event.position().y()-area.top())/area.height())
            self.center_changed.emit(*self.center)
            self.update()


class CropEditorDialog(QDialog):
    def __init__(self, video_path, rally, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assisted Highlight Crop")
        self.resize(900, 700)
        self.keyframes = deepcopy(rally.crop_keyframes)
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("Could not open the video for crop editing")
        self.capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        layout = QVBoxLayout(self)
        explanation = QLabel("Move to a key moment, then click the ball or player to keep in view. The export smoothly moves between your points. This is assisted cropping.")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.canvas = CropCanvas()
        self.canvas.center_changed.connect(self.add_center)
        layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.time = QDoubleSpinBox()
        self.time.setRange(rally.start_time, rally.end_time)
        self.time.setDecimals(3)
        self.time.setSingleStep(1/30)
        self.time.setSuffix(" s")
        self.time.setValue(rally.start_time)
        self.time.valueChanged.connect(self.load_frame)
        self.aspect = QComboBox()
        self.aspect.addItems(("9:16", "1:1", "16:9"))
        self.aspect.currentTextChanged.connect(self.change_ratio)
        row.addWidget(QLabel("Source time"))
        row.addWidget(self.time)
        row.addWidget(QLabel("Crop preview"))
        row.addWidget(self.aspect)
        layout.addLayout(row)
        self.list = QListWidget()
        self.list.setMaximumHeight(120)
        self.list.currentRowChanged.connect(self.select_key)
        layout.addWidget(self.list)
        remove = QPushButton("Remove selected keyframe")
        remove.clicked.connect(self.remove_key)
        layout.addWidget(remove)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.finished.connect(lambda *_: self.capture.release())
        self.load_frame(rally.start_time)
        self.refresh_list()

    def change_ratio(self, ratio):
        self.canvas.ratio = ratio
        self.canvas.update()

    def load_frame(self, timestamp):
        self.capture.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)
        ok, frame = self.capture.read()
        if ok:
            self.canvas.image = _array_to_qimage(frame, rgb=False)
            if self.keyframes:
                left = right = self.keyframes[0]
                for key in self.keyframes:
                    right = key
                    if key["time"] >= timestamp:
                        break
                    left = key
                span = right["time"] - left["time"]
                amount = max(0.0, min(1.0, (timestamp-left["time"])/span)) if span else 0.0
                amount = amount*amount*(3-2*amount)
                self.canvas.center = tuple(left[axis] + (right[axis]-left[axis])*amount for axis in ("x", "y"))
            else:
                self.canvas.center = (.5, .5)
            self.canvas.update()

    def add_center(self, x, y):
        time = round(self.time.value(), 3)
        self.keyframes = [k for k in self.keyframes if abs(k["time"]-time) > .001]
        self.keyframes.append({"time": time, "x": x, "y": y})
        self.keyframes.sort(key=lambda k: k["time"])
        self.refresh_list()

    def refresh_list(self):
        self.list.clear()
        self.list.addItems([f"{k['time']:.3f}s — center {k['x']:.0%}, {k['y']:.0%}" for k in self.keyframes])

    def select_key(self, row):
        if 0 <= row < len(self.keyframes):
            key = self.keyframes[row]
            self.canvas.center = key["x"], key["y"]
            self.time.setValue(key["time"])
            self.canvas.update()

    def remove_key(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.keyframes):
            self.keyframes.pop(row)
            self.refresh_list()
            self.load_frame(self.time.value())
