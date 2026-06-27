"""
全局 QSS 样式表：定义应用的浅色监控台主题配色。
"""

# 配色常量
COLORS = {
    "bg_main": "#F3F5F7",
    "bg_surface": "#FFFFFF",
    "bg_raised": "#EEF2F5",
    "bg_input": "#F8FAFC",
    "bg_hover": "#E5EBF1",
    "bg_pressed": "#D9E2EA",
    "primary": "#B9903F",
    "primary_hover": "#CBA550",
    "primary_pressed": "#9E772E",
    "on_primary": "#171B1F",
    "accent": "#1F2933",
    "text": "#26323D",
    "text_dim": "#5D6B78",
    "text_muted": "#626E7A",
    "border": "#D4DCE5",
    "border_soft": "#E5EAF0",
    "border_focus": "#B9903F",
    "success": "#087B6B",
    "warning": "#9C812A",
    "danger": "#B84E42",
    "danger_soft": "#FFF0ED",
    "danger_pressed": "#F6D7D0",
    "info": "#3C7EA6",
    "scrollbar": "#C9D3DD",
    "scrollbar_hover": "#AEBCCA",
}

# 全局样式
GLOBAL_STYLE = f"""
/* ===== 全局 ===== */
QWidget {{
    background-color: {COLORS["bg_main"]};
    color: {COLORS["text"]};
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 12px;
}}

/* ===== 主窗口 ===== */
QMainWindow {{
    background-color: {COLORS["bg_main"]};
    border: none;
}}

/* ===== 分组框 ===== */
QGroupBox {{
    background-color: {COLORS["bg_surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 7px;
    margin-top: 15px;
    padding: 15px 11px 11px 11px;
    font-weight: bold;
    font-size: 12px;
    color: {COLORS["text_dim"]};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 4px;
    padding: 0 6px;
    color: {COLORS["text_dim"]};
    font-size: 11px;
    font-weight: bold;
}}

/* ===== 按钮 ===== */
QPushButton {{
    background-color: {COLORS["bg_raised"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 5px;
    padding: 7px 16px;
    color: {COLORS["text"]};
    font-weight: normal;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: {COLORS["bg_hover"]};
    border-color: {COLORS["border_focus"]};
}}

QPushButton:pressed {{
    background-color: {COLORS["bg_pressed"]};
}}

QPushButton:disabled {{
    background-color: {COLORS["bg_surface"]};
    color: {COLORS["text_muted"]};
    border-color: {COLORS["border"]};
}}


/* ===== 标签 ===== */
QLabel {{
    background: transparent;
    color: {COLORS["text"]};
    border: none;
}}

QLabel#title_label {{
    font-size: 18px;
    font-weight: bold;
    color: {COLORS["accent"]};
    padding: 0;
}}

QLabel#subtitle_label {{
    font-size: 10px;
    color: {COLORS["text_muted"]};
    padding-top: 1px;
}}

QLabel#stats_value {{
    font-size: 24px;
    font-weight: bold;
    color: {COLORS["accent"]};
}}

QLabel#stats_label {{
    font-size: 11px;
    color: {COLORS["text_dim"]};
}}

/* ===== 滑块 ===== */
QSlider::groove:horizontal {{
    background: {COLORS["bg_input"]};
    height: 4px;
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {COLORS["primary"]};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {COLORS["accent"]};
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}

QSlider::handle:horizontal:hover {{
    background: {COLORS["primary_hover"]};
}}

/* ===== 文本编辑 ===== */
QTextEdit, QPlainTextEdit {{
    background-color: {COLORS["bg_input"]};
    border: 1px solid {COLORS["border_soft"]};
    border-radius: 5px;
    padding: 8px;
    color: {COLORS["text"]};
    font-size: 12px;
    font-family: "Consolas", "Courier New", monospace;
}}

QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {COLORS["border_focus"]};
}}

/* ===== 下拉框 ===== */
QComboBox {{
    background-color: {COLORS["bg_input"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 5px;
    padding: 6px 10px;
    color: {COLORS["text"]};
    min-height: 20px;
}}

QComboBox:hover {{
    border-color: {COLORS["border_focus"]};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS["bg_raised"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 5px;
    selection-background-color: {COLORS["primary"]};
    selection-color: {COLORS["on_primary"]};
    color: {COLORS["text"]};
}}

/* ===== 滚动条 ===== */
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {COLORS["scrollbar"]};
    border-radius: 3px;
    min-height: 40px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLORS["scrollbar_hover"]};
}}

QScrollBar::handle:vertical:pressed {{
    background: {COLORS["primary"]};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
    background: none;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {COLORS["scrollbar"]};
    border-radius: 3px;
    min-width: 40px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {COLORS["scrollbar_hover"]};
}}

QScrollBar::handle:horizontal:pressed {{
    background: {COLORS["primary"]};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
    background: none;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* ===== 分隔线 ===== */
QFrame#separator {{
    background-color: {COLORS["border"]};
    max-height: 1px;
}}

/* ===== 提示框 ===== */
QToolTip {{
    background-color: {COLORS["bg_raised"]};
    border: 1px solid {COLORS["border"]};
    color: {COLORS["text"]};
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
}}
"""

# 视频面板专用样式
VIDEO_PANEL_STYLE = f"""
    background-color: #000;
    border: 1px solid {COLORS["border"]};
    border-radius: 7px;
"""

# 统计面板样式
STATS_PANEL_STYLE = f"""
    background-color: {COLORS["bg_surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 7px;
    padding: 8px;
"""

# 日志面板专用样式
LOG_PANEL_STYLE = f"""
    background-color: {COLORS["bg_input"]};
    border: 1px solid {COLORS["border_soft"]};
    border-radius: 5px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
    color: {COLORS["text_dim"]};
"""
