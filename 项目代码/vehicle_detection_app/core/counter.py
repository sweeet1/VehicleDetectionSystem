"""
车辆计数器 — 双线（ENTRY+EXIT）双向计数 + 实时测速 + 穿越测速。
"""

import math
from utils.config import MODEL_INTERNAL_TO_NAME, ID_TO_KEY


class VehicleCounter:
    """双线车辆计数器"""

    ASPECT_RATIOS = {0: (0.8, 2.5), 1: (0.6, 2.0), 2: (1.5, 5.0), 3: (0.5, 5.0)}

    def __init__(self, entry_pct: float = 0.33, exit_pct: float = 0.66,
                 line_type: str = "horizontal", min_traj: int = 5, min_area: int = 500,
                 fps: int = 25, px_to_m: float = 0.15):
        self.entry_pct = entry_pct
        self.exit_pct = exit_pct
        self.line_type = line_type
        self.min_traj = min_traj
        self.min_area = min_area
        self.fps = fps
        self.px_to_m = px_to_m

        self.entry_pos = 0
        self.exit_pos = 0
        self.entry_count = 0
        self.exit_count = 0
        self.trajectories = {}
        self.counted_ids = set()
        self._current_dim = 720
        self.class_counts = {"entry": {0: 0, 1: 0, 2: 0, 3: 0}, "exit": {0: 0, 1: 0, 2: 0, 3: 0}}
        self.speeds = []                              # 穿越测速记录
        self._rt_speeds: dict[int, float] = {}        # 实时速度: track_id → km/h (EMA平滑)
        self._display_speed: float = 0                # 面板显示的缓存均速
        self._speed_counter: int = 0                  # 帧计数，每2s更新一次面板速度

    def set_dimension(self, dim: int):
        self._current_dim = dim
        self.entry_pos = int(self.entry_pct * dim)
        self.exit_pos = int(self.exit_pct * dim)

    def set_entry_pct(self, pct: float):
        self.entry_pct = pct
        self.entry_pos = int(pct * self._current_dim)

    def set_exit_pct(self, pct: float):
        self.exit_pct = pct
        self.exit_pos = int(pct * self._current_dim)

    def set_line_type(self, line_type: str):
        self.line_type = line_type

    def set_fps(self, fps: int):
        self.fps = fps

    def set_px_to_m(self, px_to_m: float):
        self.px_to_m = px_to_m

    def update(self, tracks: dict, frame_size: tuple, frame_id: int = 0) -> dict:
        h, w = frame_size
        dim = h if self.line_type == "horizontal" else w
        if dim != self._current_dim:
            self._current_dim = dim
            self.entry_pos = int(self.entry_pct * dim)
            self.exit_pos = int(self.exit_pct * dim)

        frame_events = []
        self._rt_speeds.clear()

        for tid, t in tracks.items():
            bbox = t["bbox"]
            class_id = t["class_id"]
            internal_cls = self._internal_to_model(class_id)
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            result = self._update_single(tid, cx, cy, internal_cls, area, bw, bh, frame_id)
            if result:
                frame_events.append({
                    "track_id": tid, "class_id": class_id,
                    "direction": result[1],
                    "class_name": ID_TO_KEY.get(class_id, "unknown"),
                    "frame_id": frame_id,
                })

        stats = self.get_stats()
        return {
            "events": frame_events,
            "total_count": stats["total_count"],
            "entry_count": stats["entry_count"],
            "exit_count": stats["exit_count"],
            "class_counts": self._ui_class_counts(),
        }

    def _update_single(self, track_id, cx, cy, cls, area, bw, bh, frame_idx):
        if track_id not in self.trajectories:
            self.trajectories[track_id] = {"points": [], "classes": [], "areas": [],
                                           "bboxes": [], "frames": [],
                                           "entry_frame": None, "exit_frame": None,
                                           "last_cx": cx, "last_cy": cy}

        traj = self.trajectories[track_id]
        traj["points"].append((cx, cy))
        traj["classes"].append(cls)
        traj["areas"].append(area)
        traj["bboxes"].append((bw, bh))
        traj["frames"].append(frame_idx)

        # ── 实时速度：相邻帧位移 → EMA 平滑的 km/h ──
        lx = traj.get("last_cx", cx)
        ly = traj.get("last_cy", cy)
        pixel_dist = math.hypot(cx - lx, cy - ly)
        instant_kmh = (pixel_dist * self.px_to_m) * self.fps * 3.6
        old_spd = self._rt_speeds.get(track_id, instant_kmh)
        self._rt_speeds[track_id] = round(old_spd * 0.7 + instant_kmh * 0.3, 1)
        traj["last_cx"] = cx
        traj["last_cy"] = cy

        self._check_crossing(traj, frame_idx, cx, cy)
        self._calc_speed(track_id)

        n = len(traj["points"])
        if track_id in self.counted_ids or n < self.min_traj or area < self.min_area:
            return None

        # 必须穿过至少一条检测线
        crossed_entry = traj.get("entry_frame") is not None
        crossed_exit = traj.get("exit_frame") is not None
        if not crossed_entry and not crossed_exit:
            return None

        # 方向由车辆实际运动决定：delta > 0 → 向下(IN)，delta < 0 → 向上(OUT)
        first, last = traj["points"][0], traj["points"][-1]
        delta = last[1] - first[1] if self.line_type == "horizontal" else last[0] - first[0]
        if abs(delta) < 5:
            return None

        self.counted_ids.add(track_id)
        refined_cls = self._refine_class(track_id, cls)
        class_name = MODEL_INTERNAL_TO_NAME.get(refined_cls, "unknown")

        if delta > 0:
            self.entry_count += 1
            self.class_counts["entry"][refined_cls] = self.class_counts["entry"].get(refined_cls, 0) + 1
            return (True, "down", class_name, None)
        else:
            self.exit_count += 1
            self.class_counts["exit"][refined_cls] = self.class_counts["exit"].get(refined_cls, 0) + 1
            return (True, "up", class_name, None)

    def _check_crossing(self, traj, frame_idx, cx, cy):
        pts = traj["points"]
        if len(pts) < 2:
            return
        prev_v = pts[-2][1] if self.line_type == "horizontal" else pts[-2][0]
        curr_v = cy if self.line_type == "horizontal" else cx
        for attr, line_pos in [("entry_frame", self.entry_pos), ("exit_frame", self.exit_pos)]:
            if prev_v < line_pos <= curr_v or prev_v > line_pos >= curr_v:
                if traj.get(attr) is None:
                    traj[attr] = frame_idx

    def _calc_speed(self, track_id):
        """穿越测速：车辆穿过两线后，用线间距算平均速度（更准确但触发晚）"""
        if any(s["track_id"] == track_id for s in self.speeds):
            return
        traj = self.trajectories[track_id]
        ef = traj.get("entry_frame"); xf = traj.get("exit_frame")
        if ef is None or xf is None:
            return
        frame_span = abs(xf - ef)
        pixel_dist = abs(self.exit_pos - self.entry_pos)
        if frame_span <= 0:
            return
        time_sec = frame_span / max(self.fps, 1)
        if time_sec < 0.2:
            return
        real_dist = pixel_dist * self.px_to_m
        if real_dist < 0.5:
            return
        speed_kmh = real_dist / time_sec * 3.6
        vote = {}
        for c in traj["classes"]:
            vote[c] = vote.get(c, 0) + 1
        dominant = max(vote, key=vote.get)
        direction = "down" if traj["points"][-1][1] > traj["points"][0][1] else "up"
        self.speeds.append({
            "track_id": track_id, "class": MODEL_INTERNAL_TO_NAME.get(dominant, "unknown"),
            "direction": direction, "real_dist_m": round(real_dist, 2),
            "time_sec": round(time_sec, 3), "speed_kmh": round(speed_kmh, 1),
        })

    def get_avg_rt_speed(self) -> float:
        """每 ~2s 更新一次的面板均速，使用移动平均平滑"""
        self._speed_counter += 1
        update_interval = max(self.fps * 2, 20)  # 2 秒对应的帧数，最少 20 帧
        if self._speed_counter % update_interval == 0 and self._rt_speeds:
            self._display_speed = round(sum(self._rt_speeds.values()) / len(self._rt_speeds), 1)
        return self._display_speed

    def get_rt_speed(self, track_id: int) -> float:
        """获取指定 track 的实时速度 (km/h)，用于显示在检测框上"""
        return self._rt_speeds.get(track_id, 0)

    def _refine_class(self, track_id, cls):
        traj = self.trajectories[track_id]
        bboxes = traj["bboxes"]
        avg_w = sum(b[0] for b in bboxes) / len(bboxes)
        avg_h = sum(b[1] for b in bboxes) / len(bboxes)
        if avg_h == 0:
            return cls
        ar = avg_w / avg_h
        lo, hi = self.ASPECT_RATIOS.get(cls, (0, 999))
        if lo <= ar <= hi:
            return cls
        best, best_score = cls, -1
        for cid, (lo2, hi2) in self.ASPECT_RATIOS.items():
            if lo2 <= ar <= hi2:
                count = traj["classes"].count(cid)
                if count > best_score:
                    best_score = count; best = cid
        return best

    def get_stats(self) -> dict:
        total = self.entry_count + self.exit_count
        avg_speed = round(sum(s["speed_kmh"] for s in self.speeds) / len(self.speeds), 1) if self.speeds else 0
        return {
            "entry_count": self.entry_count, "exit_count": self.exit_count,
            "total_count": total, "class_counts": self.class_counts,
            "avg_speed_kmh": avg_speed, "speeds": self.speeds,
        }

    def _ui_class_counts(self) -> dict:
        result = {"car": 0, "moto": 0, "bus": 0, "truck": 0}
        model_to_key = {0: "car", 1: "truck", 2: "bus", 3: "moto"}
        for direction in ("entry", "exit"):
            for model_id, count in self.class_counts.get(direction, {}).items():
                key = model_to_key.get(model_id, "moto")
                result[key] = result.get(key, 0) + count
        return result

    @staticmethod
    def _internal_to_model(internal_id: int) -> int:
        mapping = {2: 0, 5: 1, 4: 2, 3: 3}
        return mapping.get(internal_id, 3)

    def reset(self):
        self.entry_count = 0; self.exit_count = 0
        self.trajectories.clear(); self.counted_ids.clear()
        self.class_counts = {"entry": {0: 0, 1: 0, 2: 0, 3: 0}, "exit": {0: 0, 1: 0, 2: 0, 3: 0}}
        self.speeds.clear()
        self._rt_speeds.clear()
        self._display_speed = 0; self._speed_counter = 0
