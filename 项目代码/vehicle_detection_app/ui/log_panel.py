"""
日志面板：以只读文本区域显示系统运行日志。
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QPlainTextEdit


class LogPanel(QWidget):
    """系统日志显示面板"""

    def __init__(self, max_lines: int = 500, parent=None):
        super().__init__(parent)
        self._max_lines = max_lines
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)

        group = QGroupBox("系统日志")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(8, 20, 8, 8)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(self._max_lines)
        self._log_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        group_layout.addWidget(self._log_view)

        layout.addWidget(group)

    # --- 公开接口 ---

    def append(self, text: str):
        self._log_view.appendPlainText(text)
        # 自动滚动到底部
        scrollbar = self._log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        self._log_view.clear()
