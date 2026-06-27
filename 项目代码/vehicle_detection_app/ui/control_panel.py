"""
控制面板：视频源选择、检测参数调节、双线计数控制。
"""

from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSlider, QComboBox, QFileDialog,
    QLineEdit, QCheckBox,
)

from ui.stylesheet import COLORS


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event): event.ignore()

class NoWheelSlider(QSlider):
    def wheelEvent(self, event): event.ignore()


class ValueRow(QWidget):
    value_confirmed = pyqtSignal(int)

    def __init__(self, prefix: str, init_val: int, vmin: int, vmax: int, display_fn, parent=None):
        super().__init__(parent)
        self._vmin, self._vmax = vmin, vmax
        self._display_fn, self._current_val = display_fn, init_val
        self._prefix = prefix

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        self._prefix_label = QLabel(prefix + ": "); self._prefix_label.setObjectName("stats_label")
        layout.addWidget(self._prefix_label)

        self._value_edit = QLineEdit(display_fn(init_val))
        self._value_edit.setReadOnly(True); self._value_edit.setFrame(False)
        self._value_edit.setStyleSheet(f"QLineEdit {{ background: transparent; border: none; color: {COLORS['text']}; font-size: 11px; padding: 0; }}")
        self._value_edit.setFixedWidth(50)
        self._value_edit.installEventFilter(self)
        self._value_edit.returnPressed.connect(self._confirm)
        self._value_edit.editingFinished.connect(self._on_editing_finished)
        layout.addWidget(self._value_edit); layout.addStretch()

    def eventFilter(self, obj, event):
        if obj is self._value_edit and event.type() == QEvent.MouseButtonDblClick:
            self._enter_edit(); return True
        return super().eventFilter(obj, event)

    def _enter_edit(self):
        self._value_edit.setReadOnly(False); self._value_edit.setFrame(True); self._value_edit.setFixedWidth(58)
        self._value_edit.setStyleSheet(f"QLineEdit {{ background-color: {COLORS['bg_input']}; border: 1px solid {COLORS['primary']}; border-radius: 3px; padding: 1px 4px; color: {COLORS['text']}; font-size: 11px; }}")
        self._value_edit.selectAll(); self._value_edit.setFocus()

    def _exit_edit(self):
        self._value_edit.setReadOnly(True); self._value_edit.setFrame(False); self._value_edit.setFixedWidth(50)
        self._value_edit.setStyleSheet(f"QLineEdit {{ background: transparent; border: none; color: {COLORS['text']}; font-size: 11px; padding: 0; }}")
        self._value_edit.setText(self._display_fn(self._current_val))

    def _confirm(self):
        text = self._value_edit.text().strip()
        if text.endswith("%"): text = text[:-1]
        try:
            val = float(text)
            v = int(round(val * 100)) if abs(val) < 2 and self._vmax > 10 else int(round(val))
            if v < self._vmin or v > self._vmax:
                self._show_error(f"值 {text} 超出范围 [{self._display_fn(self._vmin)} ~ {self._display_fn(self._vmax)}]")
                self._exit_edit(); return
            self._current_val = v
            self._value_edit.setText(self._display_fn(v)); self._exit_edit()
            self.value_confirmed.emit(v)
        except ValueError:
            self._show_error(f"「{text}」不是有效数字"); self._exit_edit()

    def _on_editing_finished(self):
        if not self._value_edit.isReadOnly(): self._exit_edit()

    def update_value(self, v: int):
        self._current_val = v
        if self._value_edit.isReadOnly(): self._value_edit.setText(self._display_fn(v))

    def _show_error(self, msg: str):
        from PyQt5.QtWidgets import QToolTip
        QToolTip.showText(self._value_edit.mapToGlobal(self._value_edit.rect().bottomLeft()), msg, self._value_edit)


class ControlPanel(QWidget):
    """右侧控制面板"""

    open_video_clicked = pyqtSignal(str)
    open_camera_clicked = pyqtSignal(int)
    export_clicked = pyqtSignal()
    conf_threshold_changed = pyqtSignal(float)
    entry_line_changed = pyqtSignal(float)
    exit_line_changed = pyqtSignal(float)
    line_type_changed = pyqtSignal(str)
    fps_changed = pyqtSignal(int)
    px_to_m_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8); layout.setSpacing(10)

        # ── 视频源 ──
        g1 = QGroupBox("视频源"); l1 = QVBoxLayout(g1)
        self._source_combo = NoWheelComboBox()
        self._source_combo.addItems(["摄像头 (默认)", "打开视频文件..."])
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        l1.addWidget(self._source_combo)
        self._camera_combo = NoWheelComboBox()
        self._camera_combo.addItems([f"Camera {i}" for i in range(3)])
        self._camera_combo.currentIndexChanged.connect(lambda i: self.open_camera_clicked.emit(i))
        l1.addWidget(self._camera_combo)
        layout.addWidget(g1)

        # ── 检测参数 ──
        g2 = QGroupBox("检测参数"); l2 = QVBoxLayout(g2)

        self._conf_row = ValueRow("置信度阈值", 50, 10, 95, display_fn=lambda v: f"{v/100:.2f}")
        self._conf_row.value_confirmed.connect(lambda v: self._conf_slider.setValue(v))
        l2.addWidget(self._conf_row)

        self._conf_slider = NoWheelSlider(Qt.Horizontal)
        self._conf_slider.setRange(10, 95); self._conf_slider.setValue(50)
        self._conf_slider.valueChanged.connect(lambda v: (self._conf_row.update_value(v), self.conf_threshold_changed.emit(v/100.0)))
        l2.addWidget(self._conf_slider)

        # 计数线类型
        lt_label = QLabel("计数线类型:"); lt_label.setObjectName("stats_label")
        l2.addWidget(lt_label)
        self._line_type_combo = NoWheelComboBox()
        self._line_type_combo.addItems(["水平线", "垂直线"])
        self._line_type_combo.currentIndexChanged.connect(
            lambda i: self.line_type_changed.emit("horizontal" if i == 0 else "vertical"))
        l2.addWidget(self._line_type_combo)

        # ENTRY 线
        self._entry_row = ValueRow("ENTRY 线", 33, 5, 95, display_fn=lambda v: f"{v}%")
        self._entry_row.value_confirmed.connect(lambda v: self._entry_slider.setValue(v))
        l2.addWidget(self._entry_row)
        self._entry_slider = NoWheelSlider(Qt.Horizontal)
        self._entry_slider.setRange(5, 95); self._entry_slider.setValue(33)
        self._entry_slider.valueChanged.connect(lambda v: (self._entry_row.update_value(v), self.entry_line_changed.emit(v/100.0)))
        l2.addWidget(self._entry_slider)

        # EXIT 线
        self._exit_row = ValueRow("EXIT 线", 66, 5, 95, display_fn=lambda v: f"{v}%")
        self._exit_row.value_confirmed.connect(lambda v: self._exit_slider.setValue(v))
        l2.addWidget(self._exit_row)
        self._exit_slider = NoWheelSlider(Qt.Horizontal)
        self._exit_slider.setRange(5, 95); self._exit_slider.setValue(66)
        self._exit_slider.valueChanged.connect(lambda v: (self._exit_row.update_value(v), self.exit_line_changed.emit(v/100.0)))
        l2.addWidget(self._exit_slider)

        # FPS
        fps_label = QLabel("视频帧率 (FPS):"); fps_label.setObjectName("stats_label")
        l2.addWidget(fps_label)
        self._fps_combo = NoWheelComboBox()
        self._fps_combo.addItems(["15", "20", "25", "30"])
        self._fps_combo.setCurrentIndex(2)  # 25
        self._fps_combo.currentIndexChanged.connect(lambda i: self.fps_changed.emit(int(self._fps_combo.currentText())))
        l2.addWidget(self._fps_combo)

        # px_to_m 速度标定
        cal_row = QWidget(); cal_hl = QHBoxLayout(cal_row); cal_hl.setContentsMargins(0, 0, 0, 0)

        cal_label = QLabel("标定 px→m:"); cal_label.setObjectName("stats_label")
        cal_hl.addWidget(cal_label)
        self._cal_value = QLabel("0.15"); self._cal_value.setStyleSheet(
            f"color: {COLORS['accent']}; font-weight: bold; font-size: 12px;")
        cal_hl.addWidget(self._cal_value); cal_hl.addStretch()
        l2.addWidget(cal_row)

        self._cal_slider = NoWheelSlider(Qt.Horizontal)
        self._cal_slider.setRange(1, 50); self._cal_slider.setValue(15)  # 15 → 0.15
        self._cal_slider.valueChanged.connect(
            lambda v: (self._cal_value.setText(str(v / 100)),
                       self.px_to_m_changed.emit(v / 100.0)))
        l2.addWidget(self._cal_slider)

        layout.addWidget(g2)

        # ── 导出 ──
        g3 = QGroupBox("数据导出"); l3 = QVBoxLayout(g3)
        self._btn_export = QPushButton("导出统计数据 (CSV)")
        self._btn_export.clicked.connect(self.export_clicked.emit)
        l3.addWidget(self._btn_export)
        layout.addWidget(g3)
        layout.addStretch()

    def sync_from_flow(self, flow: dict):
        """根据车流分析结果同步 UI 控件"""
        if not flow:
            return
        # 线类型下拉框
        lt = flow.get("line_type", "horizontal")
        self._line_type_combo.blockSignals(True)
        self._line_type_combo.setCurrentIndex(0 if lt == "horizontal" else 1)
        self._line_type_combo.blockSignals(False)

        # ENTRY 滑块
        ep = int(flow["entry_pct"] * 100)
        self._entry_slider.blockSignals(True)
        self._entry_slider.setValue(ep)
        self._entry_slider.blockSignals(False)
        self._entry_row.update_value(ep)

        # EXIT 滑块
        xp = int(flow["exit_pct"] * 100)
        self._exit_slider.blockSignals(True)
        self._exit_slider.setValue(xp)
        self._exit_slider.blockSignals(False)
        self._exit_row.update_value(xp)

    def set_entry_pct(self, pct: float):
        self._set_line_pct(self._entry_slider, self._entry_row, pct)

    def set_exit_pct(self, pct: float):
        self._set_line_pct(self._exit_slider, self._exit_row, pct)

    def _set_line_pct(self, slider: QSlider, row: ValueRow, pct: float):
        value = int(round(min(0.95, max(0.05, pct)) * 100))
        slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(False)
        row.update_value(value)

    def _on_source_changed(self, index: int):
        self._camera_combo.setVisible(index == 0)
        if index == 0:
            self.open_camera_clicked.emit(self._camera_combo.currentIndex())
        elif index == 1:
            path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*)")
            if path: self.open_video_clicked.emit(path)
            else: self._source_combo.setCurrentIndex(0)
