"""
视频处理模块：负责视频文件/摄像头帧读取，在独立线程中运行，
通过 Qt 信号将帧数据发送到主线程进行显示和处理。
"""

import time
import cv2
from PyQt5.QtCore import QThread, pyqtSignal


class VideoProcessor(QThread):
    """视频读取线程，持续读取帧并通过信号发送"""

    frame_ready = pyqtSignal(object)
    fps_updated = pyqtSignal(float)
    source_opened = pyqtSignal(bool, str)
    finished = pyqtSignal()
    progress = pyqtSignal(int, int)       # (current_pos_ms, total_duration_ms)

    def __init__(self):
        super().__init__()
        self._source = None
        self._is_camera = False
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._paused = False
        self._target_fps = 30
        self._frame_interval = 1.0 / 30
        self._pending_seek: int | None = None     # 待处理的跳转位置 (ms)

    # --- 公开接口 ---

    def open_video(self, path: str):
        self._source = path
        self._is_camera = False
        self._start_capture()

    def open_camera(self, camera_id: int = 0):
        self._source = camera_id
        self._is_camera = True
        self._start_capture()

    def set_target_fps(self, fps: int):
        self._target_fps = fps
        self._frame_interval = 1.0 / max(fps, 1)

    def set_speed(self, speed: float):
        """设置播放倍速：调整帧间隔 = 1/(base_fps * speed)"""
        self._frame_interval = 1.0 / max(self._target_fps * speed, 0.1)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False
        self._paused = False

    def seek(self, position_ms: int):
        """跳转到指定毫秒位置（仅视频文件有效）"""
        if self._is_camera or self._cap is None:
            return
        self._pending_seek = position_ms

    def is_camera(self) -> bool:
        return self._is_camera

    def run(self):
        if self._cap is None or not self._cap.isOpened():
            return

        self._running = True
        last_time = time.time()

        while self._running:
            # 处理跳转请求
            if self._pending_seek is not None:
                self._cap.set(cv2.CAP_PROP_POS_MSEC, self._pending_seek)
                self._pending_seek = None
                # 跳转后立即读一帧，让画面跟上（即便是暂停状态）
                ret, seek_frame = self._cap.read()
                if ret:
                    self.frame_ready.emit(seek_frame)
                    if not self._is_camera:
                        ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                        total = self._cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(self._cap.get(cv2.CAP_PROP_FPS), 1) * 1000
                        self.progress.emit(int(ms), int(total))

            if self._paused:
                self.msleep(50)
                if self._cap and not self._is_camera:
                    ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                    total = self._cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(self._cap.get(cv2.CAP_PROP_FPS), 1) * 1000
                    self.progress.emit(int(ms), int(total))
                continue

            ret, frame = self._cap.read()
            if not ret:
                if not self._is_camera:
                    self.finished.emit()
                    break
                self.msleep(100)
                continue

            # 帧率控制（倍速通过调整 frame_interval 实现，>1x 不等 sleep）
            elapsed = time.time() - last_time
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))

            now = time.time()
            actual_fps = 1.0 / max(now - last_time, 0.001)
            last_time = now

            self.frame_ready.emit(frame)
            self.fps_updated.emit(actual_fps)

            # 发送进度（仅视频文件）
            if not self._is_camera and self._cap:
                ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                total = self._cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(self._cap.get(cv2.CAP_PROP_FPS), 1) * 1000
                self.progress.emit(int(ms), int(total))

        self._release()

    # --- 内部 ---

    def _start_capture(self):
        self._release()
        self._cap = cv2.VideoCapture(self._source)
        if self._cap.isOpened():
            self.source_opened.emit(True, str(self._source))
            if not self.isRunning():
                self.start()
        else:
            self.source_opened.emit(False, f"无法打开视频源: {self._source}")

    def _release(self):
        self._running = False
        self._pending_seek = None
        if self._cap:
            self._cap.release()
            self._cap = None
