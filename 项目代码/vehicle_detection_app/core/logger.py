"""
系统日志模块：提供统一的日志记录接口，支持输出到 UI 组件。
"""

from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal


class SystemLogger(QObject):
    """日志系统，通过 Qt 信号将日志推送到 UI 面板"""

    new_log = pyqtSignal(str)  # 发送单条日志

    def __init__(self):
        super().__init__()
        self._history: list = []

    def info(self, msg: str):
        self._emit("INFO", msg)

    def warn(self, msg: str):
        self._emit("WARN", msg)

    def error(self, msg: str):
        self._emit("ERROR", msg)

    def debug(self, msg: str):
        self._emit("DEBUG", msg)

    def _emit(self, level: str, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {msg}"
        self._history.append(formatted)
        self.new_log.emit(formatted)

    def get_history(self) -> list:
        return self._history.copy()
