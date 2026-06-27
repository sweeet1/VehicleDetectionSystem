"""
主窗口：组装所有 UI 面板，连接信号槽，协调核心模块。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QMessageBox, QScrollArea, QFileDialog,
)

from core.video_processor import VideoProcessor
from core.detector import Detector
from core.tracker import VehicleTracker
from core.counter import VehicleCounter
from core.traffic_analyzer import analyze_traffic_flow
from core.logger import SystemLogger
from utils.config import AppConfig
from utils.export import StatsExporter
from ui.video_panel import VideoPanel
from ui.video_control_bar import VideoControlBar
from ui.control_panel import ControlPanel
from ui.stats_panel import StatsPanel
from ui.log_panel import LogPanel
from ui.app_icon import create_app_icon
from ui.title_bar import TitleBar
from ui.stylesheet import GLOBAL_STYLE


class MainWindow(QMainWindow):

    def __init__(self, config: AppConfig = None):
        super().__init__()
        self.config = config or AppConfig()

        self.logger = SystemLogger()
        self.video_processor = VideoProcessor()
        self.detector = Detector(
            model_path=self.config.model_path,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold, img_size=self.config.img_size)
        self.tracker = VehicleTracker(
            max_history=self.config.track_history_length,
            min_frames=self.config.min_track_frames,
            iou_threshold=0.3, max_disappeared=self.config.max_disappeared,
            max_distance=self.config.max_distance, alpha=self.config.track_alpha)
        self.counter = VehicleCounter(
            entry_pct=self.config.entry_pct, exit_pct=self.config.exit_pct,
            line_type=self.config.line_type, min_traj=self.config.min_traj,
            min_area=self.config.min_area, fps=self.config.fps, px_to_m=self.config.px_to_m)

        self._running = False; self._paused = False; self._seeking = False
        self._was_playing_before_seek = False; self._is_camera_mode = True
        self._frame_count = 0; self._last_source = None; self._last_is_camera = True
        self._last_detections = []; self._last_track_speeds = {}
        self._cached_tracks = {}                    # 检测帧跟踪结果缓存，中间帧用于显示

        self._init_ui(); self._connect_signals(); self._apply_theme()
        self._on_open_camera(self.control_panel._camera_combo.currentIndex())

    # ── UI ──

    def _init_ui(self):
        self.setWindowTitle("Vehicle Detection System — YOLO26 车辆检测系统")
        self.setWindowIcon(create_app_icon())
        self.setMinimumSize(1280, 720); self.resize(1440, 860)
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(12, 10, 12, 12); root.setSpacing(8)

        self.title_bar = TitleBar(); root.addWidget(self.title_bar)

        body = QHBoxLayout(); body.setSpacing(12)
        va = QVBoxLayout(); va.setSpacing(6); va.setContentsMargins(0, 0, 0, 0)
        self.video_panel = VideoPanel(); va.addWidget(self.video_panel, 1)
        self._control_bar = VideoControlBar(); va.addWidget(self._control_bar)
        body.addLayout(va, 4)

        right_col = QVBoxLayout(); right_col.setContentsMargins(0, 0, 0, 0); right_col.setSpacing(9)
        self.stats_panel = StatsPanel(); right_col.addWidget(self.stats_panel, 0)

        self._right_scroll = QScrollArea()
        self._right_scroll.setWidgetResizable(True)
        self._right_scroll.setMinimumWidth(286)
        self._right_scroll.setMaximumWidth(336)
        self._right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._right_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.control_panel = ControlPanel()
        self._right_scroll.setWidget(self.control_panel)
        right_col.addWidget(self._right_scroll, 1)
        body.addLayout(right_col, 0)
        root.addLayout(body, 8)

        self.log_panel = LogPanel(max_lines=self.config.max_log_lines); root.addWidget(self.log_panel, 2)

    def _apply_theme(self): self.setStyleSheet(GLOBAL_STYLE)

    # ── 信号 ──

    def _connect_signals(self):
        self.control_panel.open_video_clicked.connect(self._on_open_video)
        self.control_panel.open_camera_clicked.connect(self._on_open_camera)
        self.control_panel.export_clicked.connect(self._on_export)
        self.control_panel.conf_threshold_changed.connect(self._on_conf_changed)
        self.control_panel.entry_line_changed.connect(self._on_entry_changed)
        self.control_panel.exit_line_changed.connect(self._on_exit_changed)
        self.control_panel.line_type_changed.connect(self._on_line_type_changed)
        self.control_panel.fps_changed.connect(self._on_fps_changed)
        self.control_panel.px_to_m_changed.connect(self._on_px_to_m_changed)
        self.video_panel.line_moved.connect(self._on_video_line_moved)

        self._control_bar.play_pause_clicked.connect(self._on_play_pause)
        self._control_bar.stop_clicked.connect(self._on_stop)
        self._control_bar.seek_requested.connect(self.video_processor.seek)
        self._control_bar.drag_started.connect(self._on_drag_started)
        self._control_bar.drag_seeking.connect(self._on_drag_seeking)
        self._control_bar.drag_ended.connect(self._on_drag_ended)
        self._control_bar.speed_changed.connect(self.video_processor.set_speed)
        self._control_bar.open_video_clicked.connect(self._on_switch_video)

        self.video_processor.frame_ready.connect(self._on_frame)
        self.video_processor.fps_updated.connect(self.stats_panel.update_fps)
        self.video_processor.source_opened.connect(self._on_source_status)
        self.video_processor.finished.connect(self._on_video_finished)
        self.video_processor.progress.connect(self._control_bar.update_position)

        self.stats_panel.clear_clicked.connect(self._on_clear_stats)
        self.logger.new_log.connect(self.log_panel.append)

    # ── 视频源 ──

    def _on_open_video(self, path: str):
        self._reset(); self._is_camera_mode = False
        self._last_source = path; self._last_is_camera = False
        self._control_bar.set_mode(False)
        self.logger.info(f"打开视频文件: {path}")

        # 自动分析车流
        flow = analyze_traffic_flow(path, self.config.auto_analyze_frames)
        if flow:
            self.config.line_type = flow["line_type"]
            self.config.entry_pct = flow["entry_pct"]
            self.config.exit_pct = flow["exit_pct"]
            self.counter.set_line_type(flow["line_type"])
            self.counter.set_entry_pct(flow["entry_pct"])
            self.counter.set_exit_pct(flow["exit_pct"])
            self.control_panel.sync_from_flow(flow)
            self.logger.info(f"自动分析: 方向={flow['primary']}, 线型={flow['line_type']}, "
                           f"ENTRY={flow['entry_pct']:.0%}, EXIT={flow['exit_pct']:.0%}")
        self.video_processor.open_video(path)

    def _on_open_camera(self, cam_id: int):
        self._reset(); self._is_camera_mode = True
        self._last_source = cam_id; self._last_is_camera = True
        self._control_bar.set_mode(True)
        self.logger.info(f"打开摄像头: {cam_id}")
        self.video_processor.open_camera(cam_id)

    def _on_source_status(self, success: bool, description: str):
        if success:
            self._running = True; self._control_bar.set_playing(True)
            self.logger.info(f"视频源已就绪: {description}")
            if not self._is_camera_mode: self._control_bar.reset()
        else:
            self.logger.error(f"视频源打开失败: {description}")
            QMessageBox.critical(self, "错误", f"无法打开视频源:\n{description}")

    # ── 播放/暂停/停止 ──

    def _on_play_pause(self):
        if not self._running:
            self._running = True; self._frame_count = 0
            self.counter.reset(); self.stats_panel.reset()
            if self._last_is_camera:
                self._is_camera_mode = True; self._control_bar.set_mode(True)
                self.video_processor.open_camera(self._last_source or 0)
            elif self._last_source:
                self._is_camera_mode = False; self._control_bar.set_mode(False)
                self.video_processor.open_video(self._last_source)
            else:
                self.video_processor.open_camera(0)
            return
        self._paused = not self._paused
        if self._paused: self.video_processor.pause()
        else: self.video_processor.resume()
        self._control_bar.set_playing(not self._paused)

    def _on_stop(self):
        self.video_processor.stop()
        self._running = False; self._paused = False; self._seeking = False; self._frame_count = 0
        self._last_detections = []; self._last_track_speeds = {}; self._cached_tracks = {}
        self._control_bar.set_playing(False); self._control_bar.reset()
        self.video_panel.clear(); self.tracker.reset(); self.counter.reset(); self.stats_panel.reset()
        self.logger.info("视频已停止")

    def _on_video_finished(self):
        self.video_processor.stop()
        self._running = False; self._paused = False; self._seeking = False
        self._control_bar.set_playing(False); self.tracker.reset()
        self.logger.info("视频播放结束，保留统计数据")

    def _on_switch_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "",
                                               "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*)")
        if path: self._on_open_video(path)

    # ── 拖拽 ──

    def _on_drag_started(self):
        self._seeking = True; self._was_playing_before_seek = not self._paused
        if self._was_playing_before_seek: self.video_processor.pause()

    def _on_drag_seeking(self, ms: int): self.video_processor.seek(ms)
    def _on_drag_ended(self, ms: int):
        self.video_processor.seek(ms); self._seeking = False
        if self._was_playing_before_seek: self.video_processor.resume()

    # ── 参数 ──

    def _on_export(self):
        stats = self.counter.get_stats()
        stats["fps"] = self.stats_panel.get_last_fps()
        try:
            path = StatsExporter.export_summary(stats)
            self.logger.info(f"统计数据已导出: {path}")
            QMessageBox.information(self, "导出成功", f"统计数据已导出到:\n{path}")
        except Exception as e:
            self.logger.error(f"导出失败: {e}"); QMessageBox.critical(self, "导出失败", str(e))

    def _on_conf_changed(self, value: float): self.detector.update_thresholds(conf=value)
    def _on_entry_changed(self, pct: float):
        self.counter.set_entry_pct(pct); self._refresh_video_overlay()
    def _on_exit_changed(self, pct: float):
        self.counter.set_exit_pct(pct); self._refresh_video_overlay()
    def _on_line_type_changed(self, lt: str):
        self.counter.set_line_type(lt); self._refresh_video_overlay()
    def _on_fps_changed(self, fps: int): self.counter.set_fps(fps)
    def _on_px_to_m_changed(self, v: float): self.counter.set_px_to_m(v)
    def _on_video_line_moved(self, line_name: str, pct: float):
        if line_name == "entry":
            self.counter.set_entry_pct(pct)
            self.control_panel.set_entry_pct(pct)
        elif line_name == "exit":
            self.counter.set_exit_pct(pct)
            self.control_panel.set_exit_pct(pct)

    def _on_clear_stats(self):
        self._frame_count = 0; self.counter.reset(); self.stats_panel.reset()
        self.logger.info("统计数据已清零")

    # ── 帧处理 ──

    def _on_frame(self, frame):
        if not self._running: return
        if self._paused and not self._seeking: return
        if self._seeking: self.video_panel.update_frame(frame); return

        self._frame_count += 1
        h, w = frame.shape[:2]
        is_det_frame = self._frame_count % self.config.detect_interval == 0

        if is_det_frame:
            # 检测帧：完整管线（模型 → tracker → counter）
            self._last_detections = self.detector.detect(frame)
            tracks = self.tracker.update(self._last_detections)
            self._cached_tracks = tracks
            result = self.counter.update(tracks, (h, w), self._frame_count)
            self._last_track_speeds = {tid: self.counter.get_rt_speed(tid) for tid in tracks}
        else:
            # 中间帧：不碰 tracker 和 counter，只复用缓存帧显示
            tracks = self._cached_tracks

        # stats panel 只在检测帧更新（平均速度内部有缓存机制）
        if is_det_frame:
            self.stats_panel.update_stats({
                "total_count": result["total_count"],
                "entry_count": result["entry_count"],
                "exit_count": result["exit_count"],
                "class_counts": result["class_counts"],
                "avg_speed_kmh": self.counter.get_avg_rt_speed(),
            })

        self._refresh_video_overlay(tracks=tracks, render=False)
        self.video_panel.update_frame(frame)

    # ── 内部 ──

    def _reset(self):
        if self._running: self.video_processor.stop(); self.video_processor.wait(1000)
        self._running = False; self._paused = False; self._seeking = False; self._frame_count = 0
        self._last_detections = []; self._last_track_speeds = {}; self._cached_tracks = {}
        self.tracker.reset(); self.counter.reset(); self.stats_panel.reset(); self._control_bar.reset()

    def _refresh_video_overlay(self, tracks: dict | None = None, render: bool = True):
        frame_size = self.video_panel.current_frame_size()
        if frame_size is not None:
            h, w = frame_size
            dim = h if self.counter.line_type == "horizontal" else w
            self.counter.set_dimension(dim)

        self.video_panel.update_overlay(
            tracks=self._cached_tracks if tracks is None else tracks,
            entry_pos=self.counter.entry_pos,
            exit_pos=self.counter.exit_pos,
            line_type=self.counter.line_type,
            class_names=self.config.class_names,
            class_colors=self.config.class_colors,
            track_speeds=self._last_track_speeds,
            render=render)

    def closeEvent(self, event):
        self.video_processor.stop(); self.video_processor.wait(2000)
        self.logger.info("应用关闭"); event.accept()
