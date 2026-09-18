"""Reusable Qt Multimedia video player with precise seeking controls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QSignalBlocker, QTimer, QUrl, Signal
from PySide6.QtGui import QPalette
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def format_timestamp(seconds: float, *, tenths: bool = True) -> str:
    """Format seconds as ``H:MM:SS.t`` or ``MM:SS.t``."""

    seconds = max(0.0, float(seconds or 0.0))
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    suffix = f".{int((seconds - whole) * 10 + 1e-7)}" if tenths else ""
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}{suffix}"
    return f"{minutes:02d}:{secs:02d}{suffix}"


class VideoPlayerWidget(QWidget):
    """Video surface plus transport, scrub, volume, and speed controls."""

    position_changed = Signal(float)
    duration_changed = Signal(float)
    playing_changed = Signal(bool)
    media_error = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration_ms = 0
        self._slider_is_down = False
        self._preview_end_ms: int | None = None
        self._preview_ranges: list[tuple[int, int]] = []
        self._preview_index = 0
        self._preview_loop = False
        self._loaded_path: str | None = None

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.8)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)

        self.video_widget = QVideoWidget(self)
        self.video_widget.setMinimumSize(480, 270)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        palette = self.video_widget.palette()
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
        self.video_widget.setPalette(palette)
        self.video_widget.setAutoFillBackground(True)
        self.player.setVideoOutput(self.video_widget)

        self.play_button = QPushButton("▶  Play")
        self.play_button.setToolTip("Play or pause (Space)")
        self.play_button.clicked.connect(self.toggle_playback)

        self.back_button = QPushButton("−5 s")
        self.back_button.setToolTip("Seek backward five seconds")
        self.back_button.clicked.connect(lambda: self.seek_relative(-5.0))
        self.forward_button = QPushButton("+5 s")
        self.forward_button.setToolTip("Seek forward five seconds")
        self.forward_button.clicked.connect(lambda: self.seek_relative(5.0))

        self.position_label = QLabel("00:00.0 / 00:00.0")
        self.position_label.setMinimumWidth(132)
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setTracking(True)
        self.seek_slider.setToolTip("Drag or click to scrub")
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)

        self.mute_button = QPushButton("🔊")
        self.mute_button.setFixedWidth(42)
        self.mute_button.setToolTip("Mute audio")
        self.mute_button.clicked.connect(self._toggle_mute)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(90)
        self.volume_slider.setToolTip("Volume")
        self.volume_slider.valueChanged.connect(lambda value: self.audio_output.setVolume(value / 100.0))

        self.speed_combo = QComboBox()
        self.speed_combo.setToolTip("Playback speed")
        for speed in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._set_playback_rate)

        time_row = QHBoxLayout()
        time_row.setContentsMargins(0, 0, 0, 0)
        time_row.addWidget(self.seek_slider, 1)
        time_row.addWidget(self.position_label)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.play_button)
        controls.addWidget(self.back_button)
        controls.addWidget(self.forward_button)
        controls.addStretch(1)
        controls.addWidget(self.mute_button)
        controls.addWidget(self.volume_slider)
        controls.addWidget(QLabel("Speed"))
        controls.addWidget(self.speed_combo)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.addWidget(self.video_widget, 1)
        layout.addLayout(time_row)
        layout.addLayout(controls)

        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.player.errorOccurred.connect(self._on_error)

        # A very small polling guard makes preview-stop behavior reliable across
        # multimedia backends that emit positionChanged at a coarse cadence.
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(35)
        self._preview_timer.timeout.connect(self._check_preview_boundary)

    @property
    def loaded_path(self) -> str | None:
        return self._loaded_path

    @property
    def position_seconds(self) -> float:
        return max(0.0, self.player.position() / 1000.0)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self._duration_ms / 1000.0)

    @property
    def is_playing(self) -> bool:
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def load(self, path: str | Path) -> None:
        self.stop()
        self._loaded_path = str(Path(path).expanduser().resolve())
        self._preview_end_ms = None
        self.player.setSource(QUrl.fromLocalFile(self._loaded_path))
        self.play_button.setEnabled(True)

    def unload(self) -> None:
        self.stop()
        self._loaded_path = None
        self.player.setSource(QUrl())
        self._duration_ms = 0
        self.seek_slider.setRange(0, 0)
        self._update_time_label(0)

    def toggle_playback(self) -> None:
        if not self._loaded_path:
            return
        if self.is_playing:
            self.player.pause()
        else:
            self.player.play()

    def play(self) -> None:
        if self._loaded_path:
            self.player.play()

    def pause(self) -> None:
        self.player.pause()

    def stop(self) -> None:
        self._preview_end_ms = None
        self._preview_ranges = []
        self._preview_timer.stop()
        self.player.stop()

    def set_position(self, seconds: float) -> None:
        if not self._loaded_path:
            return
        self._preview_end_ms = None
        self._preview_ranges = []
        self._preview_timer.stop()
        target = int(max(0.0, float(seconds)) * 1000.0)
        if self._duration_ms:
            target = min(target, self._duration_ms)
        self.player.setPosition(target)

    def seek_relative(self, seconds: float) -> None:
        self.set_position(self.position_seconds + float(seconds))

    def preview_range(self, start_seconds: float, end_seconds: float, *, loop: bool = False) -> None:
        """Play only the supplied source range and pause at its end."""
        self.preview_ranges([(start_seconds, end_seconds)], loop=loop)

    def preview_ranges(self, ranges: list[tuple[float, float]], *, loop: bool = False) -> None:
        """Preview an ordered edit directly from the source without rendering."""

        if not self._loaded_path or not ranges:
            return
        self._preview_ranges = [(max(0, round(s*1000)), min(round(e*1000), self._duration_ms or round(e*1000)))
                                for s, e in ranges if e > s]
        if not self._preview_ranges:
            return
        self._preview_index = 0
        self._preview_loop = loop
        self._preview_end_ms = self._preview_ranges[0][1]
        self.player.setPosition(self._preview_ranges[0][0])
        self._preview_timer.start()
        self.player.play()

    def _on_slider_pressed(self) -> None:
        self._slider_is_down = True

    def _on_slider_moved(self, value: int) -> None:
        self._update_time_label(value)

    def _on_slider_released(self) -> None:
        self._slider_is_down = False
        self.set_position(self.seek_slider.value() / 1000.0)

    def _on_position_changed(self, position_ms: int) -> None:
        if not self._slider_is_down:
            with QSignalBlocker(self.seek_slider):
                self.seek_slider.setValue(position_ms)
            self._update_time_label(position_ms)
        self.position_changed.emit(position_ms / 1000.0)
        self._check_preview_boundary()

    def _on_duration_changed(self, duration_ms: int) -> None:
        self._duration_ms = max(0, int(duration_ms))
        self.seek_slider.setRange(0, self._duration_ms)
        self._update_time_label(self.player.position())
        self.duration_changed.emit(self.duration_seconds)

    def _on_playback_state_changed(self, state: Any) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setText("❚❚  Pause" if playing else "▶  Play")
        if not playing and self._preview_end_ms is None:
            self._preview_timer.stop()
        self.playing_changed.emit(playing)

    def _check_preview_boundary(self) -> None:
        if self._preview_end_ms is None:
            return
        if self.player.position() >= self._preview_end_ms - 15:
            boundary = self._preview_end_ms
            self._preview_index += 1
            if self._preview_index >= len(self._preview_ranges) and self._preview_loop:
                self._preview_index = 0
            if self._preview_index < len(self._preview_ranges):
                start, end = self._preview_ranges[self._preview_index]
                self._preview_end_ms = end
                self.player.setPosition(start)
                self.player.play()
                return
            self._preview_end_ms = None
            self._preview_timer.stop()
            self.player.pause()
            self.player.setPosition(boundary)

    def _toggle_mute(self) -> None:
        muted = not self.audio_output.isMuted()
        self.audio_output.setMuted(muted)
        self.mute_button.setText("🔇" if muted else "🔊")
        self.mute_button.setToolTip("Unmute audio" if muted else "Mute audio")

    def _set_playback_rate(self) -> None:
        speed = self.speed_combo.currentData()
        if speed is not None:
            self.player.setPlaybackRate(float(speed))

    def _update_time_label(self, position_ms: int) -> None:
        self.position_label.setText(
            f"{format_timestamp(position_ms / 1000.0)} / {format_timestamp(self.duration_seconds)}"
        )

    def _on_error(self, *_: Any) -> None:
        message = self.player.errorString() or "The video could not be played."
        self.media_error.emit(message)
