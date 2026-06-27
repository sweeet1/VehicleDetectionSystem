"""
视频控制栏：始终显示在视频画面下方。
- 视频模式：播放/暂停 + 停止 + 时间 + 进度滑块 + 总时长 + 倍速 + 切换视频
- 摄像头模式：仅播放/暂停按钮
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QSlider,
    QComboBox, QFileDialog,
)

from ui.stylesheet import COLORS

BUTTON_STYLE = (
    f"QPushButton {{ background: {COLORS['bg_raised']}; border: 1px solid {COLORS['border']}; "
    f"border-radius: 5px; color: {COLORS['text']}; font-size: 12px; font-weight: bold; }} "
    f"QPushButton:hover {{ border-color: {COLORS['border_focus']}; background: {COLORS['bg_hover']}; }}"
    f"QPushButton:pressed {{ background: {COLORS['bg_pressed']}; }}"
)

TRANSPORT_STYLE = (
    f"QPushButton#transport_play_pause {{ background: {COLORS['primary']}; border: 1px solid {COLORS['primary']}; "
    f"border-radius: 6px; color: {COLORS['on_primary']}; font-size: 11px; font-weight: bold; padding: 0 10px; }}"
    f"QPushButton#transport_play_pause:hover {{ background: {COLORS['primary_hover']}; border-color: {COLORS['primary_hover']}; }}"
    f"QPushButton#transport_play_pause:pressed {{ background: {COLORS['primary_pressed']}; border-color: {COLORS['primary_pressed']}; }}"
    f"QPushButton#transport_stop {{ background: transparent; border: 1px solid {COLORS['border']}; "
    f"border-radius: 6px; color: {COLORS['danger']}; font-size: 11px; font-weight: bold; padding: 0 10px; }}"
    f"QPushButton#transport_stop:hover {{ background: {COLORS['danger_soft']}; border-color: {COLORS['danger']}; color: {COLORS['danger']}; }}"
    f"QPushButton#transport_stop:pressed {{ background: {COLORS['danger_pressed']}; border-color: {COLORS['danger']}; }}"
)

SPEED_OPTIONS = ["0.25x", "0.5x", "0.75x", "1x", "1.5x", "2x"]
SPEED_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def _fmt_ms(ms: int) -> str:
    s = max(ms, 0) // 1000
    return f"{s // 60}:{s % 60:02d}"


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSeekSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()


class VideoControlBar(QWidget):
    """统一的视频控制栏"""

    play_pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    seek_requested = pyqtSignal(int)
    speed_changed = pyqtSignal(float)           # 倍速值
    open_video_clicked = pyqtSignal()           # 切换视频文件

    drag_started = pyqtSignal()
    drag_seeking = pyqtSignal(int)
    drag_ended = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dragging = False
        self._total_ms = 0
        self._is_playing = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(6)

        # ── 播放/暂停 ──
        self._btn_play_pause = QPushButton("PLAY")
        self._btn_play_pause.setObjectName("transport_play_pause")
        self._btn_play_pause.setFixedSize(76, 30)
        self._btn_play_pause.setToolTip("播放 / 暂停")
        self._btn_play_pause.setStyleSheet(TRANSPORT_STYLE)
        self._btn_play_pause.clicked.connect(self.play_pause_clicked.emit)
        layout.addWidget(self._btn_play_pause)

        # ── 停止 ──
        self._btn_stop = QPushButton("STOP")
        self._btn_stop.setObjectName("transport_stop")
        self._btn_stop.setFixedSize(66, 30)
        self._btn_stop.setToolTip("停止播放")
        self._btn_stop.setStyleSheet(TRANSPORT_STYLE)
        self._btn_stop.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self._btn_stop)

        # ── 当前时间 ──
        self._time_label = QLabel("0:00")
        self._time_label.setObjectName("stats_label")
        self._time_label.setFixedWidth(38)
        self._time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._time_label)

        # ── 进度滑块 ──
        self._slider = NoWheelSeekSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.sliderPressed.connect(self._on_press)
        self._slider.sliderReleased.connect(self._on_release)
        self._slider.sliderMoved.connect(self._on_drag)
        layout.addWidget(self._slider, 1)

        # ── 总时长 ──
        self._total_label = QLabel("0:00")
        self._total_label.setObjectName("stats_label")
        self._total_label.setFixedWidth(38)
        layout.addWidget(self._total_label)

        # ── 倍速 ──
        self._speed_combo = NoWheelComboBox()
        self._speed_combo.addItems(SPEED_OPTIONS)
        self._speed_combo.setCurrentIndex(3)  # 1x
        self._speed_combo.setFixedWidth(64)
        self._speed_combo.setStyleSheet(
            f"QComboBox {{ background: {COLORS['bg_input']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 5px; padding: 2px 5px; color: {COLORS['text']}; font-size: 11px; }} "
            f"QComboBox:hover {{ border-color: {COLORS['border_focus']}; }} "
            f"QComboBox::drop-down {{ border: none; }} "
            f"QComboBox QAbstractItemView {{ background: {COLORS['bg_raised']}; "
            f"selection-background-color: {COLORS['primary']}; selection-color: {COLORS['on_primary']}; color: {COLORS['text']}; }}"
        )
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        layout.addWidget(self._speed_combo)

        # ── 切换视频 ──
        self._btn_switch = QPushButton("切换视频")
        self._btn_switch.setFixedHeight(28)
        self._btn_switch.setToolTip("选择其他视频文件")
        self._btn_switch.setStyleSheet(
            f"QPushButton {{ background: {COLORS['primary']}; border: none; "
            f"border-radius: 5px; color: {COLORS['on_primary']}; font-size: 11px; font-weight: bold; "
            f"padding-left: 10px; padding-right: 10px; }} "
            f"QPushButton:hover {{ background: {COLORS['primary_hover']}; }}"
        )
        self._btn_switch.clicked.connect(self.open_video_clicked.emit)
        layout.addWidget(self._btn_switch)

        # 视频模式才显示的子控件
        self._video_widgets = [
            self._btn_stop, self._time_label, self._slider, self._total_label,
            self._speed_combo, self._btn_switch,
        ]

    # --- 公开接口 ---

    def set_mode(self, is_camera: bool):
        for w in self._video_widgets:
            w.setVisible(not is_camera)

    def set_playing(self, playing: bool):
        self._is_playing = playing
        self._btn_play_pause.setText("PAUSE" if playing else "PLAY")

    def update_position(self, current_ms: int, total_ms: int):
        self._total_ms = total_ms
        if total_ms > 0:
            self._total_label.setText(_fmt_ms(total_ms))
        if not self._dragging:
            self._time_label.setText(_fmt_ms(current_ms))
            if total_ms > 0:
                self._slider.blockSignals(True)
                self._slider.setValue(int(current_ms / total_ms * 1000))
                self._slider.blockSignals(False)

    def reset(self):
        self._total_ms = 0
        self._is_playing = False
        self._btn_play_pause.setText("PLAY")
        self._slider.setValue(0)
        self._time_label.setText("0:00")
        self._total_label.setText("0:00")

    # --- 内部 ---

    def _on_press(self):
        self._dragging = True
        self.drag_started.emit()

    def _on_drag(self, value: int):
        if self._total_ms > 0:
            ms = int(value / 1000 * self._total_ms)
            self._time_label.setText(_fmt_ms(ms))
            self.drag_seeking.emit(ms)

    def _on_release(self):
        self._dragging = False
        if self._total_ms > 0:
            ms = int(self._slider.value() / 1000 * self._total_ms)
            self.seek_requested.emit(ms)
            self.drag_ended.emit(ms)

    def _on_speed_changed(self, index: int):
        self.speed_changed.emit(SPEED_VALUES[index])
