"""Interactive rally and diagnostic-signal timeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from .video_player import format_timestamp


def _read_value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


class RallyTimeline(QWidget):
    """Paint and manipulate the full-video time axis.

    The timeline always displays enabled/disabled rally spans and the playhead.
    In debug mode it additionally overlays normalized detector traces.
    """

    position_requested = Signal(float)
    rally_selected = Signal(int)

    _TRACE_SPECS = (
        ("motion_scores", "Motion", QColor("#58a6ff")),
        ("roi_motion_scores", "ROI", QColor("#a371f7")),
        ("motion_spread_scores", "Spread", QColor("#39d0c5")),
        ("audio_scores", "Audio", QColor("#f2cc60")),
        ("serve_scores", "Serve", QColor("#ff9bce")),
        ("readiness_scores", "Ready", QColor("#d2a8ff")),
        ("retrieval_scores", "Retrieval", QColor("#ffa657")),
        ("pose_serve_scores", "Pose", QColor("#79c0ff")),
        ("learned_scores", "Learned", QColor("#ffffff")),
        ("rally_scores", "Rally score", QColor("#3fb950")),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration = 0.0
        self._position = 0.0
        self._rallies: list[Any] = []
        self._selected_index = -1
        self._debug_enabled = False
        self._timestamps: list[float] = []
        self._traces: dict[str, list[float]] = {}
        self._threshold = 0.55
        self._dragging = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(96)
        self.setToolTip("Click or drag to scrub; click a highlighted block to select its rally")

    def sizeHint(self):  # type: ignore[override]
        from PySide6.QtCore import QSize

        return QSize(720, 105 if not self._debug_enabled else 185)

    def set_duration(self, seconds: float) -> None:
        self._duration = max(0.0, float(seconds or 0.0))
        self._position = min(self._position, self._duration) if self._duration else 0.0
        self.update()

    def set_position(self, seconds: float) -> None:
        position = max(0.0, float(seconds or 0.0))
        if self._duration:
            position = min(position, self._duration)
        if abs(position - self._position) > 0.001:
            self._position = position
            self.update()

    def set_rallies(self, rallies: Sequence[Any], selected_index: int = -1) -> None:
        self._rallies = list(rallies)
        self._selected_index = selected_index if 0 <= selected_index < len(self._rallies) else -1
        self.update()

    def set_selected_index(self, index: int) -> None:
        normalized = index if 0 <= index < len(self._rallies) else -1
        if normalized != self._selected_index:
            self._selected_index = normalized
            self.update()

    def set_debug_enabled(self, enabled: bool) -> None:
        self._debug_enabled = bool(enabled)
        self.setMinimumHeight(178 if enabled else 96)
        self.updateGeometry()
        self.update()

    def set_threshold(self, threshold: float) -> None:
        self._threshold = max(0.0, min(1.0, float(threshold)))
        self.update()

    def set_signal_data(self, result: Any | None) -> None:
        self._timestamps = []
        self._traces = {}
        if result is None:
            self.update()
            return
        try:
            raw_timestamps = _read_value(result, "timestamps", ())
            self._timestamps = [float(value) for value in (() if raw_timestamps is None else raw_timestamps)]
        except (TypeError, ValueError):
            self._timestamps = []
        for key, _, _ in self._TRACE_SPECS:
            try:
                raw_values = _read_value(result, key, ())
                values = [float(value) for value in (() if raw_values is None else raw_values)]
            except (TypeError, ValueError):
                values = []
            if values:
                self._traces[key] = values
        self.update()

    def _plot_rect(self) -> QRectF:
        top = 86.0 if self._debug_enabled else 13.0
        return QRectF(12.0, top, max(1.0, self.width() - 24.0), max(24.0, self.height() - top - 27.0))

    def _graph_rect(self) -> QRectF:
        return QRectF(12.0, 14.0, max(1.0, self.width() - 24.0), 62.0)

    def _x_for_time(self, seconds: float, rect: QRectF) -> float:
        if self._duration <= 0:
            return rect.left()
        return rect.left() + max(0.0, min(1.0, seconds / self._duration)) * rect.width()

    def _time_for_x(self, x: float, rect: QRectF) -> float:
        if self._duration <= 0 or rect.width() <= 0:
            return 0.0
        ratio = max(0.0, min(1.0, (x - rect.left()) / rect.width()))
        return ratio * self._duration

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt virtual
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#10151d"))

        plot = self._plot_rect()
        if self._debug_enabled:
            self._paint_traces(painter, self._graph_rect())

        painter.setPen(QPen(QColor("#303946"), 1))
        painter.setBrush(QColor("#1a222d"))
        painter.drawRoundedRect(plot, 4, 4)

        if self._duration <= 0:
            painter.setPen(QColor("#8b949e"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Open a video to view its timeline")
            return

        for index, rally in enumerate(self._rallies):
            try:
                start = float(getattr(rally, "start_time"))
                end = float(getattr(rally, "end_time"))
            except (TypeError, ValueError, AttributeError):
                continue
            x1 = self._x_for_time(start, plot)
            x2 = self._x_for_time(end, plot)
            block = QRectF(x1, plot.top() + 2, max(2.0, x2 - x1), plot.height() - 4)
            enabled = bool(getattr(rally, "enabled", True))
            color = QColor("#238636" if enabled else "#4b5563")
            color.setAlpha(220 if index == self._selected_index else 150)
            painter.setBrush(color)
            border = QColor("#f0f6fc") if index == self._selected_index else QColor("#3fb950")
            painter.setPen(QPen(border, 2 if index == self._selected_index else 1))
            painter.drawRoundedRect(block, 3, 3)
            if block.width() >= 24:
                painter.setPen(QColor("#f0f6fc"))
                painter.drawText(block, Qt.AlignmentFlag.AlignCenter, str(index + 1))

        # Ticks and labels under the track.
        painter.setPen(QColor("#8b949e"))
        metrics = QFontMetrics(painter.font())
        tick_count = max(2, min(8, self.width() // 120))
        for tick in range(tick_count + 1):
            seconds = self._duration * tick / tick_count
            x = self._x_for_time(seconds, plot)
            painter.drawLine(QPointF(x, plot.bottom()), QPointF(x, plot.bottom() + 4))
            text = format_timestamp(seconds, tenths=False)
            text_width = metrics.horizontalAdvance(text)
            painter.drawText(QPointF(max(1.0, min(self.width() - text_width - 1.0, x - text_width / 2)), self.height() - 5), text)

        playhead_x = self._x_for_time(self._position, plot)
        painter.setPen(QPen(QColor("#ff7b72"), 2))
        painter.drawLine(QPointF(playhead_x, 5), QPointF(playhead_x, plot.bottom() + 2))
        painter.setBrush(QColor("#ff7b72"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(
            QPolygonF(
                [
                QPointF(playhead_x - 5, 4),
                QPointF(playhead_x + 5, 4),
                QPointF(playhead_x, 10),
                ]
            )
        )

    def _paint_traces(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor("#303946"), 1))
        painter.setBrush(QColor("#0d1117"))
        painter.drawRoundedRect(rect, 4, 4)
        if not self._timestamps or not self._traces:
            painter.setPen(QColor("#8b949e"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Detector signals appear here after analysis")
            return

        painter.save()
        painter.setClipRect(rect)
        threshold_y = rect.bottom() - self._threshold * rect.height()
        painter.setPen(QPen(QColor("#ff7b72"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(rect.left(), threshold_y), QPointF(rect.right(), threshold_y))

        for key, _, color in self._TRACE_SPECS:
            values = self._traces.get(key)
            if not values:
                continue
            count = min(len(values), len(self._timestamps))
            if count < 2:
                continue
            stride = max(1, count // max(1, int(rect.width() * 2)))
            path = QPainterPath()
            started = False
            for i in range(0, count, stride):
                x = self._x_for_time(self._timestamps[i], rect)
                normalized = max(0.0, min(1.0, values[i]))
                y = rect.bottom() - normalized * rect.height()
                if not started:
                    path.moveTo(x, y)
                    started = True
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(color, 1.5))
            painter.drawPath(path)
        painter.restore()

        # Compact legend across the top edge.
        legend_x = rect.left() + 5
        legend_y = rect.top()
        painter.setFont(self.font())
        metrics = QFontMetrics(painter.font())
        for key, label, color in self._TRACE_SPECS:
            if key not in self._traces:
                continue
            label_width = 19 + metrics.horizontalAdvance(label)
            if legend_x + label_width > rect.right():
                legend_x = rect.left() + 5
                legend_y += metrics.height() + 2
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPointF(legend_x, legend_y + 7), QPointF(legend_x + 10, legend_y + 7))
            painter.setPen(QColor("#c9d1d9"))
            painter.drawText(QPointF(legend_x + 14, legend_y + 11), label)
            legend_x += label_width

    def _rally_at(self, seconds: float) -> int:
        candidates: list[tuple[float, int]] = []
        for index, rally in enumerate(self._rallies):
            try:
                start = float(getattr(rally, "start_time"))
                end = float(getattr(rally, "end_time"))
            except (TypeError, ValueError, AttributeError):
                continue
            if start <= seconds <= end:
                candidates.append((end - start, index))
        return min(candidates)[1] if candidates else -1

    def _request_position(self, x: float, *, select: bool) -> None:
        if self._duration <= 0:
            return
        seconds = self._time_for_x(x, self._plot_rect())
        self.position_requested.emit(seconds)
        if select:
            index = self._rally_at(seconds)
            if index >= 0:
                self.rally_selected.emit(index)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._request_position(event.position().x(), select=True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        seconds = self._time_for_x(event.position().x(), self._plot_rect())
        if self._dragging:
            self._request_position(event.position().x(), select=False)
            event.accept()
            return
        index = self._rally_at(seconds)
        if index >= 0:
            rally = self._rallies[index]
            start = float(getattr(rally, "start_time", 0.0))
            end = float(getattr(rally, "end_time", 0.0))
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"Rally {index + 1}  {format_timestamp(start)} → {format_timestamp(end)}  ({end - start:.2f} s)",
                self,
            )
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._duration <= 0:
            return
        step = 1.0 if event.angleDelta().y() > 0 else -1.0
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step *= 5.0
        self.position_requested.emit(max(0.0, min(self._duration, self._position + step)))
        event.accept()
