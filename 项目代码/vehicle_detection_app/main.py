"""
应用入口：启动 YOLO26 车辆检测与流量统计系统。
"""

import sys
import os

# 确保项目根目录在 sys.path 中
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from utils.config import AppConfig
from ui.app_icon import create_app_icon
from ui.main_window import MainWindow


def resource_path(relative_path: str) -> str:
    """Return a path that works both from source and from a PyInstaller bundle."""
    base_dir = getattr(sys, "_MEIPASS", PROJECT_ROOT)
    return os.path.join(base_dir, relative_path)


def main():
    # 高 DPI 支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Vehicle Detection System")
    app.setWindowIcon(create_app_icon())

    # ONNX 模型路径
    config = AppConfig(
        model_path=resource_path("best_train2.onnx"),
        conf_threshold=0.25,
    )

    window = MainWindow(config)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
