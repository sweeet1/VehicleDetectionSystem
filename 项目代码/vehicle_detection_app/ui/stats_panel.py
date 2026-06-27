"""
统计面板：显示 IN/OUT 双向计数、车型分类、车速、FPS。
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QGridLayout, QFrame, QPushButton,
)

from ui.stylesheet import COLORS


class StatsPanel(QWidget):
    """车辆统计面板"""

    clear_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_fps = 0
        self.setMinimumHeight(280)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(0)

        group = QGroupBox("车辆统计")
        group.setStyleSheet(
            f"QGroupBox {{ background-color: {COLORS['bg_surface']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 7px; margin-top: 14px; padding: 11px 8px 7px 8px; font-weight: bold; "
            f"color: {COLORS['text_dim']}; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; "
            f"left: 14px; top: 2px; padding: 0 6px; color: {COLORS['text_dim']}; font-size: 11px; font-weight: bold; }}"
        )
        grid = QGridLayout(group)
        grid.setContentsMargins(4, 16, 4, 5)
        grid.setVerticalSpacing(4)
        grid.setHorizontalSpacing(6)

        # ── 总计数 ──
        self._total_label = QLabel("0")
        self._total_label.setObjectName("stats_value")
        self._total_label.setAlignment(Qt.AlignCenter)
        self._total_label.setStyleSheet(f"font-size: 30px; color: {COLORS['accent']}; font-weight: bold; padding: 1px 0;")
        grid.addWidget(self._total_label, 0, 0, 1, 4)

        grid.addWidget(self._make_caption("总车辆数"), 1, 0, 1, 4)

        # ── IN / OUT ──
        self._in_label = QLabel("IN: 0")
        self._in_label.setAlignment(Qt.AlignCenter)
        self._in_label.setStyleSheet(
            f"font-size: 13px; color: {COLORS['success']}; font-weight: bold; "
            f"background: {COLORS['bg_input']}; border: 1px solid {COLORS['border_soft']}; "
            f"border-radius: 5px; padding: 5px 0;"
        )
        grid.addWidget(self._in_label, 2, 0, 1, 2)

        self._out_label = QLabel("OUT: 0")
        self._out_label.setAlignment(Qt.AlignCenter)
        self._out_label.setStyleSheet(
            f"font-size: 13px; color: {COLORS['danger']}; font-weight: bold; "
            f"background: {COLORS['bg_input']}; border: 1px solid {COLORS['border_soft']}; "
            f"border-radius: 5px; padding: 5px 0;"
        )
        grid.addWidget(self._out_label, 2, 2, 1, 2)

        # ── 分隔 ──
        sep = QFrame()
        sep.setObjectName("separator"); sep.setFrameShape(QFrame.HLine)
        grid.addWidget(sep, 3, 0, 1, 4)

        # ── 车型分类 ──
        self._class_widgets = {}
        classes = [("轿车", "car"), ("其他", "moto"), ("公交", "bus"), ("货车", "truck")]
        for i, (name, key) in enumerate(classes):
            col = i % 4; row = 4
            vl = QLabel("0")
            vl.setAlignment(Qt.AlignCenter)
            vl.setStyleSheet(f"font-size: 15px; color: {COLORS['text']}; font-weight: bold;")
            grid.addWidget(vl, row, col)
            grid.addWidget(self._make_caption(name), row + 1, col)
            self._class_widgets[key] = vl

        # ── 平均车速 + FPS ──
        row6 = QHBoxLayout()
        row6.setContentsMargins(0, 3, 0, 0)
        self._speed_label = QLabel("实时均速: -- km/h")
        self._speed_label.setObjectName("stats_label")
        row6.addWidget(self._speed_label, 1)

        self._fps_label = QLabel("FPS: --")
        self._fps_label.setObjectName("stats_label")
        row6.addWidget(self._fps_label, 1)

        self._btn_clear = QPushButton("↺ 清零")
        self._btn_clear.setFixedHeight(22)
        self._btn_clear.setMinimumWidth(50)
        self._btn_clear.setToolTip("手动清零统计数据")
        self._btn_clear.setCursor(Qt.PointingHandCursor)
        self._btn_clear.setStyleSheet(
            f"QPushButton {{ background: transparent; border: 1px solid {COLORS['border']}; "
            f"border-radius: 5px; color: {COLORS['text_dim']}; font-size: 10px; padding: 0 8px; }} "
            f"QPushButton:hover {{ background: {COLORS['danger_soft']}; border-color: {COLORS['danger']}; "
            f"color: {COLORS['danger']}; }}"
        )
        self._btn_clear.clicked.connect(self.clear_clicked.emit)
        row6.addWidget(self._btn_clear)

        grid.addLayout(row6, 6, 0, 1, 4)

        layout.addWidget(group)

    @staticmethod
    def _make_caption(text):
        l = QLabel(text)
        l.setObjectName("stats_label"); l.setAlignment(Qt.AlignCenter)
        return l

    # ── 公开接口 ──

    def update_stats(self, stats: dict):
        total = stats.get("total_count", 0)
        self._total_label.setText(str(total))
        self._in_label.setText(f"IN: {stats.get('entry_count', 0)}")
        self._out_label.setText(f"OUT: {stats.get('exit_count', 0)}")
        cc = stats.get("class_counts", {})
        for key, w in self._class_widgets.items():
            w.setText(str(cc.get(key, 0)))
        speed = stats.get("avg_speed_kmh", 0)
        self._speed_label.setText(f"实时均速: {speed} km/h" if speed else "实时均速: -- km/h")

    def update_fps(self, fps: float):
        self._last_fps = fps
        self._fps_label.setText(f"FPS: {fps:.1f}")

    def get_last_fps(self) -> float:
        return self._last_fps

    def reset(self):
        self._total_label.setText("0")
        self._in_label.setText("IN: 0")
        self._out_label.setText("OUT: 0")
        for w in self._class_widgets.values():
            w.setText("0")
        self._speed_label.setText("均速: -- km/h")
        self._fps_label.setText("FPS: --")
