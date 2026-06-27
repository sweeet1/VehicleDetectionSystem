"""
视频显示面板：渲染 OpenCV 帧，叠加边界框、轨迹线、计数线。
使用 PIL 绘制中文标签，避免 OpenCV putText 乱码。
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QEvent, QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ui.stylesheet import COLORS

# 中文字体（Windows 系统自带）
_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"   # 微软雅黑
_FONT_SIZE = 16

# 预创建字体
try:
    _FONT = ImageFont.truetype(_FONT_PATH, _FONT_SIZE)
except Exception:
    _FONT = ImageFont.load_default()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = _hex_to_rgb(hex_color)
    return b, g, r


_ENTRY_BGR = _hex_to_bgr(COLORS["success"])
_EXIT_BGR = _hex_to_bgr(COLORS["danger"])
_ENTRY_RGB = _hex_to_rgb(COLORS["success"])
_EXIT_RGB = _hex_to_rgb(COLORS["danger"])


class VideoPanel(QWidget):
    """视频渲染组件"""

    line_moved = pyqtSignal(str, float)  # ("entry"|"exit", pct)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._label.setMinimumSize(640, 360)
        self._label.setText("等待视频源...")
        self._label.setStyleSheet(
            f"background-color: #030201; border: 1px solid {COLORS['border']}; "
            f"border-radius: 7px; color: {COLORS['text_muted']}; font-size: 15px;"
        )
        self._label.setMouseTracking(True)
        self._label.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._current_frame: np.ndarray | None = None
        self._overlay_objects: list = []
        self._entry_pos: int = 0
        self._exit_pos: int = 0
        self._has_lines: bool = False
        self._line_type: str = "horizontal"
        self._class_colors: dict = {}
        self._dragging_line: str | None = None
        self._drag_threshold_px = 10

    def update_frame(self, frame: np.ndarray):
        self._current_frame = frame.copy()
        self._render()

    def update_overlay(self, tracks: dict, entry_pos: int = 0, exit_pos: int = 0,
                       line_type: str = "horizontal",
                       class_names: dict = None, class_colors: dict = None,
                       track_speeds: dict = None, render: bool = True):
        self._overlay_objects = []
        speeds = track_speeds or {}
        for tid, t in (tracks or {}).items():
            self._overlay_objects.append({
                "track_id": tid,
                "bbox": t["bbox"],
                "class_id": t["class_id"],
                "class_name": (class_names or {}).get(t["class_id"], str(t["class_id"])),
                "confidence": t["confidence"],
                "history": list(t["history"]),
                "speed": speeds.get(tid, 0),
            })
        self._entry_pos = entry_pos
        self._exit_pos = exit_pos
        self._has_lines = entry_pos > 0 or exit_pos > 0
        self._line_type = line_type
        self._class_colors = class_colors or {}
        if render:
            self._render()

    def clear(self):
        self._current_frame = None
        self._overlay_objects.clear()
        self._has_lines = False
        self._dragging_line = None
        self._label.unsetCursor()
        self._label.setText("等待视频源...")

    def eventFilter(self, obj, event):
        if obj is self._label:
            event_type = event.type()
            if event_type == QEvent.MouseButtonPress:
                return self._on_mouse_press(event)
            if event_type == QEvent.MouseMove:
                return self._on_mouse_move(event)
            if event_type == QEvent.MouseButtonRelease:
                return self._on_mouse_release(event)
            if event_type == QEvent.Leave and self._dragging_line is None:
                self._label.unsetCursor()
        return super().eventFilter(obj, event)

    def current_frame_size(self) -> tuple[int, int] | None:
        if self._current_frame is None:
            return None
        h, w = self._current_frame.shape[:2]
        return h, w

    def _on_mouse_press(self, event) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        line = self._nearest_line(event.pos())
        if line is None:
            return False
        self._dragging_line = line
        self._label.setCursor(self._drag_cursor())
        self._move_dragged_line(event.pos())
        return True

    def _on_mouse_move(self, event) -> bool:
        if self._dragging_line is not None:
            self._move_dragged_line(event.pos())
            return True
        self._update_hover_cursor(event.pos())
        return False

    def _on_mouse_release(self, event) -> bool:
        if self._dragging_line is None or event.button() != Qt.LeftButton:
            return False
        self._move_dragged_line(event.pos())
        self._dragging_line = None
        self._update_hover_cursor(event.pos())
        return True

    def _move_dragged_line(self, pos: QPoint):
        if self._dragging_line is None:
            return
        pct = self._pct_from_label_pos(pos, self._line_type)
        if pct is None or self._current_frame is None:
            return
        h, w = self._current_frame.shape[:2]
        dim = h if self._line_type == "horizontal" else w
        frame_pos = int(round(pct * dim))
        if self._dragging_line == "entry":
            self._entry_pos = frame_pos
        else:
            self._exit_pos = frame_pos
        self._has_lines = True
        self._render()
        self.line_moved.emit(self._dragging_line, pct)

    def _update_hover_cursor(self, pos: QPoint):
        if self._nearest_line(pos) is None:
            self._label.unsetCursor()
        else:
            self._label.setCursor(self._drag_cursor())

    def _drag_cursor(self):
        return Qt.SizeVerCursor if self._line_type == "horizontal" else Qt.SizeHorCursor

    def _nearest_line(self, pos: QPoint) -> str | None:
        if not self._has_lines or self._current_frame is None:
            return None

        distances = []
        for name, frame_pos in (("entry", self._entry_pos), ("exit", self._exit_pos)):
            label_pos = self._line_label_pos(frame_pos)
            if label_pos is not None:
                distances.append((abs(self._event_axis(pos) - label_pos), name))

        if not distances:
            return None
        distance, name = min(distances, key=lambda item: item[0])
        return name if distance <= self._drag_threshold_px else None

    def _line_label_pos(self, frame_pos: int) -> int | None:
        rect = self._display_rect()
        if rect is None or self._current_frame is None:
            return None
        x, y, display_w, display_h = rect
        h, w = self._current_frame.shape[:2]
        if self._line_type == "horizontal":
            return int(round(y + frame_pos / max(h, 1) * display_h))
        return int(round(x + frame_pos / max(w, 1) * display_w))

    def _event_axis(self, pos: QPoint) -> int:
        return pos.y() if self._line_type == "horizontal" else pos.x()

    def _pct_from_label_pos(self, pos: QPoint, line_type: str) -> float | None:
        rect = self._display_rect()
        if rect is None:
            return None
        x, y, display_w, display_h = rect
        if line_type == "horizontal":
            raw = (pos.y() - y) / max(display_h, 1)
        else:
            raw = (pos.x() - x) / max(display_w, 1)
        return min(0.95, max(0.05, raw))

    def _display_rect(self) -> tuple[int, int, int, int] | None:
        if self._current_frame is None:
            return None

        h, w = self._current_frame.shape[:2]
        label_w = max(self._label.width(), 1)
        label_h = max(self._label.height(), 1)
        scale = min(label_w / max(w, 1), label_h / max(h, 1))
        display_w = max(1, int(round(w * scale)))
        display_h = max(1, int(round(h * scale)))
        x = (label_w - display_w) // 2
        y = (label_h - display_h) // 2
        return x, y, display_w, display_h

    def _render(self):
        if self._current_frame is None:
            return

        frame = self._current_frame.copy()
        h, w = frame.shape[:2]

        # 先用 OpenCV 画边界框和轨迹线（性能好）
        for obj in self._overlay_objects:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color_bgr = self._class_colors.get(obj["class_id"], (0, 200, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

            history = obj.get("history", [])
            if len(history) >= 2:
                for i in range(1, len(history)):
                    cv2.line(frame,
                             (int(history[i - 1][0]), int(history[i - 1][1])),
                             (int(history[i][0]), int(history[i][1])),
                             color_bgr, 2)

        if self._has_lines:
            if self._line_type == "horizontal":
                cv2.line(frame, (0, self._entry_pos), (w, self._entry_pos), _ENTRY_BGR, 2)
                cv2.line(frame, (0, self._exit_pos), (w, self._exit_pos), _EXIT_BGR, 2)
            else:
                cv2.line(frame, (self._entry_pos, 0), (self._entry_pos, h), _ENTRY_BGR, 2)
                cv2.line(frame, (self._exit_pos, 0), (self._exit_pos, h), _EXIT_BGR, 2)

        # 转换为 PIL 绘制中文标签文字
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        for obj in self._overlay_objects:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color_bgr = self._class_colors.get(obj["class_id"], (0, 200, 255))
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR → RGB

            spd_str = f" {obj.get('speed', 0):.0f}km/h" if obj.get('speed', 0) > 0 else ""
            label = f"#{obj['track_id']} {obj['class_name']} {obj['confidence']:.2f}{spd_str}"
            bbox2 = draw.textbbox((0, 0), label, font=_FONT)
            tw, th = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

            label_y = max(0, y1 - th - 4)
            draw.rectangle([x1, label_y, x1 + tw + 6, y1], fill=color_rgb)
            draw.text((x1 + 3, label_y), label, fill=(0, 0, 0), font=_FONT)

        # 双线标注
        if self._has_lines:
            if self._line_type == "horizontal":
                draw.text((6, max(0, self._entry_pos - 22)), "ENTRY", fill=_ENTRY_RGB, font=_FONT)
                draw.text((6, min(h - 20, self._exit_pos + 2)), "EXIT", fill=_EXIT_RGB, font=_FONT)
            else:
                entry_x = max(0, min(max(w - 60, 0), self._entry_pos + 4))
                exit_x = max(0, min(max(w - 45, 0), self._exit_pos + 4))
                draw.text((entry_x, 6), "ENTRY", fill=_ENTRY_RGB, font=_FONT)
                draw.text((exit_x, 26), "EXIT", fill=_EXIT_RGB, font=_FONT)

        # PIL Image (RGB) → QImage
        result = np.ascontiguousarray(pil_img)
        h2, w2, ch = result.shape
        qimg = QImage(result.data, w2, h2, ch * w2, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(
            self._label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._label.setPixmap(pixmap)
