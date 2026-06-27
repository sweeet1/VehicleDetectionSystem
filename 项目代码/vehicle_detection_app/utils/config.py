"""
应用配置：定义所有可调参数、类别映射、默认路径等。
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

# ─── 内部类别映射 ───
CLASS_NAMES_CN: Dict[int, str] = {2: "轿车", 3: "其他", 4: "公交车", 5: "货车"}

CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    2: (65, 164, 217), 3: (170, 183, 100), 4: (182, 157, 127), 5: (95, 121, 208),
}

ID_TO_KEY: Dict[int, str] = {2: "car", 3: "moto", 4: "bus", 5: "truck"}

# 模型 ID(0-3) → 内部 ID(2-5)
MODEL_TO_INTERNAL: Dict[int, int] = {0: 2, 1: 5, 2: 4, 3: 3}
MODEL_CLASS_NAMES = {0: "car", 1: "van", 2: "bus", 3: "others"}

# 内部 ID → 模型 ID
INTERNAL_TO_MODEL = {2: 0, 5: 1, 4: 2, 3: 3}
MODEL_INTERNAL_TO_NAME = {0: "轿车", 1: "货车", 2: "公交车", 3: "其他"}


@dataclass
class AppConfig:
    """应用全局配置"""

    # --- 检测参数 ---
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: int = 640
    detect_interval: int = 3

    # --- 跟踪参数 ---
    track_history_length: int = 30
    min_track_frames: int = 1
    max_disappeared: int = 60
    max_distance: int = 120
    track_alpha: float = 0.7

    # --- 计数参数 ---
    entry_pct: float = 0.33
    exit_pct: float = 0.66
    line_type: str = "horizontal"
    min_traj: int = 5
    min_area: int = 500
    fps: int = 25
    px_to_m: float = 0.15
    auto_analyze_frames: int = 30       # 自动分析采样帧数

    # --- 视频源 ---
    camera_id: int = 0
    target_fps: int = 30

    # --- 导出 ---
    export_dir: str = "./exports"

    # --- 模型 ---
    model_path: str = ""

    # --- 日志 ---
    max_log_lines: int = 500

    class_names: Dict[int, str] = field(default_factory=lambda: CLASS_NAMES_CN)
    class_colors: Dict[int, Tuple[int, int, int]] = field(default_factory=lambda: CLASS_COLORS)
