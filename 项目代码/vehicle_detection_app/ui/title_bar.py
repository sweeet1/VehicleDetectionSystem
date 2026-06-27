"""
标题栏：品牌标识 + 产品标题。
"""

from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from ui.app_icon import draw_kinpaku_mark
from ui.stylesheet import COLORS


class ControlRoomIcon(QWidget):
    """交通监测感的小型品牌图标。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("kinpaku_mark")
        self.setFixedSize(34, 34)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        draw_kinpaku_mark(painter, min(self.width(), self.height()))
        painter.end()


class TitleBar(QWidget):
    """应用顶部标题区域。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("app_title_bar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(10)

        self.icon = ControlRoomIcon()
        layout.addWidget(self.icon)

        text_stack = QVBoxLayout()
        text_stack.setContentsMargins(0, 0, 0, 0)
        text_stack.setSpacing(0)

        self.title = QLabel("车辆流量监测控制台")
        self.title.setObjectName("title_label")
        self.title.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 17px; font-weight: bold; background: transparent;"
        )
        text_stack.addWidget(self.title)

        self.subtitle = QLabel("YOLO26 Detection - Real-time Traffic Analytics")
        self.subtitle.setObjectName("subtitle_label")
        self.subtitle.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; font-family: Consolas, 'Courier New', monospace; background: transparent;"
        )
        text_stack.addWidget(self.subtitle)

        layout.addLayout(text_stack)
        layout.addStretch()
