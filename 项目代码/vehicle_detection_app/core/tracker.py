"""
多目标跟踪器 — 速度预测 + 轻量匹配 + EMA 平滑。
检测间隔中无检测帧由速度向量预测位置，框平滑跟随。
"""

import numpy as np


def _compute_iou_matrix(boxes_a, boxes_b):
    """向量化 IoU 矩阵计算"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    boxes_a = np.array(boxes_a, dtype=np.float32)
    boxes_b = np.array(boxes_b, dtype=np.float32)
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def _score_match(score_matrix, threshold=0.3):
    """Greedy one-to-one matching without an extra SciPy runtime dependency."""
    if score_matrix.size == 0:
        return []

    candidates = []
    rows, cols = score_matrix.shape
    for row in range(rows):
        for col in range(cols):
            score = score_matrix[row, col]
            if score >= threshold:
                candidates.append((float(score), row, col))

    matches = []
    used_rows = set()
    used_cols = set()
    for score, row, col in sorted(candidates, reverse=True):
        if row in used_rows or col in used_cols:
            continue
        matches.append((row, col))
        used_rows.add(row)
        used_cols.add(col)
    return matches


class VehicleTracker:
    """速度预测 + IoU/距离混合匹配 + EMA 平滑，接口兼容旧版"""

    def __init__(self, max_history: int = 30, min_frames: int = 1,
                 iou_threshold: float = 0.3, max_disappeared: int = 60,
                 max_distance: int = 120, alpha: float = 0.7):
        self.max_history = max_history
        self.min_frames = min_frames
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.alpha = alpha
        self._tracks = {}

    def reset(self):
        self._tracks.clear()

    def update(self, detections: list) -> dict:
        for tid in list(self._tracks.keys()):
            self._tracks[tid]["matched"] = False

        if not detections:
            for tid in list(self._tracks.keys()):
                self._tracks[tid]["disappeared"] += 1
                if self._tracks[tid]["disappeared"] > self.max_disappeared:
                    del self._tracks[tid]
            return self._get_results()

        track_ids = list(self._tracks.keys())

        # 用速度预测位置做匹配
        pred_boxes = []
        for tid in track_ids:
            t = self._tracks[tid]
            b = t["bbox"]
            vx = t.get("vx", 0)
            vy = t.get("vy", 0)
            pred_boxes.append([b[0] + vx, b[1] + vy, b[2] + vx, b[3] + vy])

        det_boxes = [d["bbox"] for d in detections]
        iou_mat = _compute_iou_matrix(pred_boxes, det_boxes)

        # 距离混合评分
        if track_ids and detections:
            pred_ctr = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in pred_boxes])
            det_ctr = np.array([[(d["bbox"][0]+d["bbox"][2])/2, (d["bbox"][1]+d["bbox"][3])/2] for d in detections])
            dx = pred_ctr[:, 0:1] - det_ctr[:, 0:1].T
            dy = pred_ctr[:, 1:2] - det_ctr[:, 1:2].T
            dist = np.sqrt(dx*dx + dy*dy)
            dist_score = np.clip(1.0 - dist / self.max_distance, 0, 1)
            combined = 0.4 * iou_mat + 0.6 * dist_score
        else:
            combined = iou_mat

        matches = _score_match(combined, threshold=0.2)

        matched_t = set()
        for ti, di in matches:
            tid = track_ids[ti]
            det = detections[di]
            old = np.array(self._tracks[tid]["bbox"])
            new = np.array(det["bbox"])
            smoothed = old * self.alpha + new * (1 - self.alpha)
            self._tracks[tid]["bbox"] = tuple(map(int, smoothed.tolist()))
            self._tracks[tid]["class_id"] = det["class_id"]
            self._tracks[tid]["class_name"] = det["class_name"]
            self._tracks[tid]["confidence"] = det["confidence"]

            # 更新速度向量
            ocx = (old[0] + old[2]) / 2; ocy = (old[1] + old[3]) / 2
            ncx = (new[0] + new[2]) / 2; ncy = (new[1] + new[3]) / 2
            vx = (ncx - ocx) * 0.5 + self._tracks[tid].get("vx", 0) * 0.5
            vy = (ncy - ocy) * 0.5 + self._tracks[tid].get("vy", 0) * 0.5
            self._tracks[tid]["vx"] = vx; self._tracks[tid]["vy"] = vy
            self._tracks[tid]["disappeared"] = 0
            self._tracks[tid]["age"] += 1
            self._tracks[tid]["matched"] = True

            # 历史轨迹
            cx = (self._tracks[tid]["bbox"][0] + self._tracks[tid]["bbox"][2]) / 2
            cy = (self._tracks[tid]["bbox"][1] + self._tracks[tid]["bbox"][3]) / 2
            if "history" in self._tracks[tid]:
                self._tracks[tid]["history"].append((cx, cy))
            matched_t.add(ti)

        # 未匹配 tracker 超时
        for ti in range(len(track_ids)):
            if ti not in matched_t:
                tid = track_ids[ti]
                self._tracks[tid]["disappeared"] += 1
                if self._tracks[tid]["disappeared"] > self.max_disappeared:
                    del self._tracks[tid]

        # 新 detection 建 track
        matched_d = {di for _, di in matches}
        for di, det in enumerate(detections):
            if di not in matched_d:
                bbox = det["bbox"]
                cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
                from collections import deque
                history = deque(maxlen=self.max_history)
                history.append((cx, cy))
                tid = max(self._tracks.keys()) + 1 if self._tracks else 0
                self._tracks[tid] = {
                    "bbox": bbox, "class_id": det["class_id"],
                    "class_name": det["class_name"], "confidence": det["confidence"],
                    "disappeared": 0, "age": 1, "vx": 0, "vy": 0,
                    "matched": True, "history": history,
                }

        return self._get_results()

    def _get_results(self) -> dict:
        results = {}
        for tid, t in self._tracks.items():
            if not t.get("matched", False):
                continue
            if t.get("age", 0) >= self.min_frames:
                results[tid] = {
                    "bbox": t["bbox"],
                    "class_id": t["class_id"],
                    "class_name": t["class_name"],
                    "confidence": t["confidence"],
                    "history": list(t.get("history", [])),
                }
        return results
