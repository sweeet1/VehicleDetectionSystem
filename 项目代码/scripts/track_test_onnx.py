"""
ONNX版 YOLO26 多目标跟踪 + 车流量统计
不依赖 Ultralytics, 使用 onnxruntime 推理 + 轻量级 IoU 跟踪器

运行方式:
    python scripts/track_test_onnx.py

依赖:
    pip install onnxruntime opencv-python numpy
"""

import os
import sys
import json
import csv
import cv2
import numpy as np

# ======================== 配置区 ========================
ONNX_MODEL = "runs/train/yolo26n_ua_detrac/weights/best_train2.onnx"
VIDEO_SOURCE = "test1.mp4"

CONF_THRESH = 0.25
IOU_THRESH = 0.45
NMS_IOU_THRESH = 0.3
IMG_SIZE = 640

OUTPUT_DIR = "runs/track_test_onnx"
OUTPUT_VIDEO = "tracked_output_onnx.mp4"
SAVE_VIDEO = True
SHOW_PREVIEW = True

ENTRY_PCT = 0.33
EXIT_PCT = 0.66
MIN_TRAJ_LENGTH = 5
MIN_AREA = 500
FPS = 25
PIXELS_TO_METERS = 0.15

MAX_DISAPPEARED = 30
MAX_DISTANCE = 80
ANALYSIS_FRAMES = 30
# ========================================================


def nms_per_class(detections, iou_thresh=0.3):
    """按类别做 NMS, 避免不同类的框被误删"""
    if not detections:
        return []

    det_arr = np.array(detections)
    classes = np.unique(det_arr[:, 5].astype(int))

    keep = []
    for cls in classes:
        cls_mask = det_arr[:, 5].astype(int) == cls
        cls_dets = det_arr[cls_mask]
        cls_boxes = cls_dets[:, :4].astype(np.float32)
        cls_scores = cls_dets[:, 4].astype(np.float32)

        x1, y1, x2, y2 = cls_boxes[:, 0], cls_boxes[:, 1], cls_boxes[:, 2], cls_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = cls_scores.argsort()[::-1]

        while len(order) > 0:
            i = order[0]
            keep.append(cls_dets[i].tolist())
            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

    return keep


def analyze_traffic_flow(video_path, n_frames=30):
    """
    分析前N帧光流, 自动检测:
    1. 车流主方向 (vertical/horizontal)
    2. 车辆运动方向 (up/down/left/right)
    3. 检测线建议位置
    返回: {
        "direction": "vertical" or "horizontal",
        "flow_x": float, "flow_y": float,
        "entry_pct": float, "exit_pct": float,
        "line_type": "horizontal" or "vertical"
    }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_step = max(1, total // (n_frames + 1))
    flows = []

    prev_gray = None
    frame_count = 0
    for i in range(min(n_frames * 3, total)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * sample_step)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = flow.shape[:2]
            cy, cx = h // 2, w // 2
            region = flow[cy-h//4:cy+h//4, cx-w//4:cx+w//4]
            fx = region[:,:,0].mean()
            fy = region[:,:,1].mean()
            if abs(fx) > 0.5 or abs(fy) > 0.5:
                flows.append([fx, fy])
                frame_count += 1
                if frame_count >= n_frames:
                    break
        prev_gray = gray

    cap.release()

    if len(flows) < 5:
        return None

    avg_fx = sum(f[0] for f in flows) / len(flows)
    avg_fy = sum(f[1] for f in flows) / len(flows)

    abs_fx = abs(avg_fx)
    abs_fy = abs(avg_fy)

    if abs_fy > abs_fx * 1.2:
        direction = "vertical"
        if avg_fy > 0:
            entry_pct = 0.25
            exit_pct = 0.75
            primary = "down"
        else:
            entry_pct = 0.75
            exit_pct = 0.25
            primary = "up"
        line_type = "horizontal"
    elif abs_fx > abs_fy * 1.2:
        direction = "horizontal"
        if avg_fx > 0:
            entry_pct = 0.25
            exit_pct = 0.75
            primary = "right"
        else:
            entry_pct = 0.75
            exit_pct = 0.25
            primary = "left"
        line_type = "vertical"
    else:
        direction = "diagonal"
        if abs_fy >= abs_fx:
            if avg_fy > 0:
                entry_pct = 0.25
                exit_pct = 0.75
                primary = "down"
            else:
                entry_pct = 0.75
                exit_pct = 0.25
                primary = "up"
            line_type = "horizontal"
        else:
            if avg_fx > 0:
                entry_pct = 0.25
                exit_pct = 0.75
                primary = "right"
            else:
                entry_pct = 0.75
                exit_pct = 0.25
                primary = "left"
            line_type = "vertical"

    return {
        "direction": direction,
        "flow_x": avg_fx,
        "flow_y": avg_fy,
        "entry_pct": entry_pct,
        "exit_pct": exit_pct,
        "line_type": line_type,
        "primary": primary,
        "width": width,
        "height": height,
    }


class ONNXDetector:
    """ONNX YOLO26 检测器"""

    def __init__(self, model_path, img_size=640, conf_thresh=0.25):
        try:
            import onnxruntime as ort
        except ImportError:
            print("[错误] 未安装 onnxruntime，请执行: pip install onnxruntime")
            sys.exit(1)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.class_names = {0: "car", 1: "van", 2: "bus", 3: "others"}

    def preprocess(self, img):
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        dx = (self.img_size - new_w) // 2
        dy = (self.img_size - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        return blob, scale, dx, dy, w, h

    def postprocess(self, output, scale, dx, dy, orig_w, orig_h):
        detections = output[0]
        boxes = []
        for det in detections:
            conf = det[4]
            if conf < self.conf_thresh:
                continue
            cls_id = int(det[5])
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            x1 = (x1 - dx) / scale
            y1 = (y1 - dy) / scale
            x2 = (x2 - dx) / scale
            y2 = (y2 - dy) / scale
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            boxes.append([x1, y1, x2, y2, conf, cls_id])
        boxes = nms_per_class(boxes, NMS_IOU_THRESH)
        return boxes

    def detect(self, img):
        blob, scale, dx, dy, orig_w, orig_h = self.preprocess(img)
        output = self.session.run(None, {self.input_name: blob})[0]
        return self.postprocess(output, scale, dx, dy, orig_w, orig_h)


def compute_iou_matrix(boxes_a, boxes_b):
    """批量计算 IoU 矩阵, 比逐对计算快很多"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    boxes_a = np.array(boxes_a)
    boxes_b = np.array(boxes_b)

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-6)


def hungarian_match(iou_matrix, threshold=0.3):
    """匈牙利匹配, 返回匹配对列表"""
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= threshold:
                matches.append((r, c))
        return matches
    except ImportError:
        matches = []
        used_rows = set()
        used_cols = set()
        for _ in range(min(iou_matrix.shape[0], iou_matrix.shape[1])):
            max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if iou_matrix[max_idx] < threshold:
                break
            r, c = max_idx
            matches.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
            iou_matrix[r, :] = -1
            iou_matrix[:, c] = -1
        return matches


class SimpleTracker:
    """
    轻量级 IoU 跟踪器 (带速度预测)

    改进:
    1. 向量化 IoU 计算
    2. 匈牙利最优匹配
    3. 速度预测: 用最近几帧的运动趋势预测下一帧位置
    4. 轨迹平滑: 指数移动平均
    5. 长时间遮挡容忍
    """

    def __init__(self, max_disappeared=60, max_distance=120):
        self.next_id = 0
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.alpha = 0.7

    def update(self, detections):
        for tid in self.tracks:
            self.tracks[tid]["matched_this_frame"] = False

        if len(detections) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return self._get_results()

        if len(self.tracks) == 0:
            for det in detections:
                self._create_track(det)
            return self._get_results()

        track_ids = list(self.tracks.keys())
        track_boxes = []
        for tid in track_ids:
            info = self.tracks[tid]
            bbox = info["bbox"]
            vx = info.get("vx", 0)
            vy = info.get("vy", 0)
            pred_bbox = [
                bbox[0] + vx, bbox[1] + vy,
                bbox[2] + vx, bbox[3] + vy
            ]
            track_boxes.append(pred_bbox)

        det_boxes = [d[:4] for d in detections]
        iou_matrix = compute_iou_matrix(track_boxes, det_boxes)

        dist_matrix = np.zeros_like(iou_matrix)
        track_centers = np.array([
            [(b[0]+b[2])/2, (b[1]+b[3])/2] for b in track_boxes
        ])
        det_centers = np.array([
            [(d[0]+d[2])/2, (d[1]+d[3])/2] for d in det_boxes
        ])
        if len(track_centers) > 0 and len(det_centers) > 0:
            dx = track_centers[:, 0:1] - det_centers[:, 0:1].T
            dy = track_centers[:, 1:2] - det_centers[:, 1:2].T
            dist_matrix = np.sqrt(dx*dx + dy*dy)
            max_dist = self.max_distance
            dist_matrix = np.clip(1.0 - dist_matrix / max_dist, 0, 1)

        combined = 0.4 * iou_matrix + 0.6 * dist_matrix
        matches = hungarian_match(combined, threshold=0.2)

        matched_tracks = set()
        matched_dets = set()

        for ti, di in matches:
            tid = track_ids[ti]
            old_bbox = np.array(self.tracks[tid]["bbox"])
            new_bbox = np.array(detections[di][:4])
            smoothed = old_bbox * self.alpha + new_bbox * (1 - self.alpha)
            self.tracks[tid]["bbox"] = smoothed.tolist()

            old_cx = (old_bbox[0] + old_bbox[2]) / 2
            old_cy = (old_bbox[1] + old_bbox[3]) / 2
            new_cx = (new_bbox[0] + new_bbox[2]) / 2
            new_cy = (new_bbox[1] + new_bbox[3]) / 2
            vx = (new_cx - old_cx) * 0.5 + self.tracks[tid].get("vx", 0) * 0.5
            vy = (new_cy - old_cy) * 0.5 + self.tracks[tid].get("vy", 0) * 0.5
            self.tracks[tid]["vx"] = vx
            self.tracks[tid]["vy"] = vy

            self.tracks[tid]["class"] = int(detections[di][5])
            self.tracks[tid]["conf"] = float(detections[di][4])
            self.tracks[tid]["disappeared"] = 0
            self.tracks[tid]["age"] += 1
            self.tracks[tid]["matched_this_frame"] = True
            matched_tracks.add(tid)
            matched_dets.add(di)

        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det)

        return self._get_results()

    def _create_track(self, det):
        self.tracks[self.next_id] = {
            "bbox": det[:4],
            "class": int(det[5]),
            "conf": float(det[4]),
            "disappeared": 0,
            "age": 1,
            "vx": 0,
            "vy": 0,
            "matched_this_frame": True,
        }
        self.next_id += 1

    def _get_results(self):
        results = []
        for tid, info in self.tracks.items():
            if not info.get("matched_this_frame", False):
                continue
            x1, y1, x2, y2 = info["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            area = w * h
            results.append({
                "track_id": tid,
                "bbox": info["bbox"],
                "center": (cx, cy),
                "class": info["class"],
                "conf": info["conf"],
                "area": area,
                "w": w,
                "h": h,
                "age": info["age"],
            })
        return results


class VehicleCounter:
    """
    车辆计数器 — 自适应水平/垂直检测线

    通过轨迹方向判断车辆行驶方向, 实现双向计数
    支持水平线(垂直运动)和垂直线(水平运动)两种模式
    """

    ASPECT_RATIOS = {
        0: (0.8, 2.5),
        1: (0.6, 2.0),
        2: (1.5, 5.0),
        3: (0.5, 5.0),
    }

    def __init__(self, entry_pos, exit_pos, line_type="horizontal",
                 min_traj=5, min_area=500, fps=25, px_to_m=0.15,
                 extra_line_pct=0.9):
        self.entry_pos = entry_pos
        self.exit_pos = exit_pos
        self.line_type = line_type
        self.min_traj = min_traj
        self.min_area = min_area
        self.fps = fps
        self.px_to_m = px_to_m
        self.extra_line_pct = extra_line_pct
        self.extra_line_pos = None

        self.entry_count = 0
        self.exit_count = 0
        self.trajectories = {}
        self.counted_ids = set()
        self.class_names = {0: "car", 1: "van", 2: "bus", 3: "others"}
        self.class_counts = {
            "entry": {0: 0, 1: 0, 2: 0, 3: 0},
            "exit": {0: 0, 1: 0, 2: 0, 3: 0},
        }
        self.speeds = []

    def set_frame_size(self, width, height):
        if self.line_type == "horizontal":
            self.extra_line_pos = int(self.extra_line_pct * height)
        else:
            self.extra_line_pos = int(self.extra_line_pct * width)

    def _get_crossing(self, traj, frame_idx, center_x, center_y):
        n = len(traj["points"])
        if n < 2:
            return

        prev_x, prev_y = traj["points"][-2]
        curr_x, curr_y = center_x, center_y

        if self.line_type == "horizontal":
            prev_pos = prev_y
            curr_pos = curr_y
        else:
            prev_pos = prev_x
            curr_pos = curr_x

        lines = [
            ("entry_frame", self.entry_pos),
            ("exit_frame", self.exit_pos),
        ]
        if self.extra_line_pos is not None:
            lines.append(("extra_frame", self.extra_line_pos))

        for attr, line_pos in lines:
            if prev_pos < line_pos and curr_pos >= line_pos:
                if traj.get(attr) is None:
                    traj[attr] = frame_idx
            if prev_pos > line_pos and curr_pos <= line_pos:
                if traj.get(attr) is None:
                    traj[attr] = frame_idx

        if n == 1:
            for attr, line_pos in lines:
                if abs(curr_pos - line_pos) < 5:
                    traj[attr] = frame_idx

    def _calc_speed(self, track_id):
        traj = self.trajectories[track_id]
        entry_f = traj.get("entry_frame")
        exit_f = traj.get("exit_frame")
        extra_f = traj.get("extra_frame")

        if entry_f is not None and exit_f is not None:
            frame_span = abs(exit_f - entry_f)
            pixel_dist = abs(self.exit_pos - self.entry_pos)
        elif entry_f is not None and extra_f is not None:
            frame_span = abs(extra_f - entry_f)
            pixel_dist = abs(self.extra_line_pos - self.entry_pos)
        elif exit_f is not None and extra_f is not None:
            frame_span = abs(extra_f - exit_f)
            pixel_dist = abs(self.extra_line_pos - self.exit_pos)
        else:
            return

        if frame_span <= 0:
            return

        time_sec = frame_span / self.fps if self.fps > 0 else 0
        if time_sec < 1.0:
            return

        real_dist = pixel_dist * self.px_to_m
        if real_dist < 0.5:
            return

        speed_kmh = real_dist / time_sec * 3.6
        vote = {}
        for c in traj["classes"]:
            vote[c] = vote.get(c, 0) + 1
        dominant_cls = max(vote, key=vote.get)
        pts = traj["points"]
        first_pt, last_pt = pts[0], pts[-1]
        if self.line_type == "horizontal":
            direction = "down" if last_pt[1] > first_pt[1] else "up"
        else:
            direction = "right" if last_pt[0] > first_pt[0] else "left"
        self.speeds.append({
            "track_id": track_id,
            "class": self.class_names.get(dominant_cls, "unknown"),
            "direction": direction,
            "real_dist_m": round(real_dist, 2),
            "time_sec": round(time_sec, 3),
            "speed_kmh": round(speed_kmh, 1),
        })

    def update(self, track_id, center_x, center_y, cls, bbox_area, bbox_w, bbox_h, frame_idx):
        if track_id not in self.trajectories:
            self.trajectories[track_id] = {
                "points": [], "classes": [], "areas": [],
                "bboxes": [], "frames": [],
                "entry_frame": None, "exit_frame": None, "extra_frame": None,
            }

        traj = self.trajectories[track_id]
        traj["points"].append((center_x, center_y))
        traj["classes"].append(cls)
        traj["areas"].append(bbox_area)
        traj["bboxes"].append((bbox_w, bbox_h))
        traj["frames"].append(frame_idx)

        self._get_crossing(traj, frame_idx, center_x, center_y)

        n = len(traj["points"])
        already_speeded = any(s["track_id"] == track_id for s in self.speeds)
        if not already_speeded:
            self._calc_speed(track_id)

        if track_id in self.counted_ids:
            return None
        if n < self.min_traj:
            return None
        if bbox_area < self.min_area:
            return None

        first = traj["points"][0]
        last = traj["points"][-1]
        if self.line_type == "horizontal":
            delta = last[1] - first[1]
        else:
            delta = last[0] - first[0]
        if abs(delta) < 10:
            return None

        crossed_entry = traj.get("entry_frame") is not None
        crossed_exit = traj.get("exit_frame") is not None
        crossed_extra = traj.get("extra_frame") is not None
        if not (crossed_entry or crossed_exit or crossed_extra):
            return None

        self.counted_ids.add(track_id)
        final_cls = self._refine_class(track_id, cls)

        if delta > 0:
            self.entry_count += 1
            self.class_counts["entry"][final_cls] += 1
            return (True, "down", self.class_names[final_cls], None)
        else:
            self.exit_count += 1
            self.class_counts["exit"][final_cls] += 1
            return (True, "up", self.class_names[final_cls], None)

    def _vote_class(self, track_id):
        traj = self.trajectories[track_id]
        vote = {}
        for c in traj["classes"]:
            vote[c] = vote.get(c, 0) + 1
        return max(vote, key=vote.get)

    def _refine_class(self, track_id, detected_cls):
        traj = self.trajectories[track_id]
        bboxes = traj["bboxes"]
        avg_w = sum(b[0] for b in bboxes) / len(bboxes)
        avg_h = sum(b[1] for b in bboxes) / len(bboxes)
        if avg_h == 0:
            return detected_cls
        aspect_ratio = avg_w / avg_h
        if detected_cls in self.ASPECT_RATIOS:
            lo, hi = self.ASPECT_RATIOS[detected_cls]
            if lo <= aspect_ratio <= hi:
                return detected_cls
        best_cls = detected_cls
        best_score = -1
        for cls_id, (lo, hi) in self.ASPECT_RATIOS.items():
            if lo <= aspect_ratio <= hi:
                count = traj["classes"].count(cls_id)
                if count > best_score:
                    best_score = count
                    best_cls = cls_id
        return best_cls

    def get_stats(self):
        return {
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "total_count": self.entry_count + self.exit_count,
            "class_counts": self.class_counts,
            "avg_speed_kmh": round(
                sum(s["speed_kmh"] for s in self.speeds) / len(self.speeds), 1
            ) if self.speeds else 0,
            "speeds": self.speeds,
        }

    def save_trajectories(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "trajectories.json")
        export_data = {}
        for tid, traj in self.trajectories.items():
            export_data[str(tid)] = {
                "points": [(int(p[0]), int(p[1])) for p in traj["points"]],
                "classes": [int(c) for c in traj["classes"]],
                "class_names": [self.class_names.get(int(c), "unknown") for c in traj["classes"]],
                "areas": [int(a) for a in traj["areas"]],
                "bboxes": [(int(b[0]), int(b[1])) for b in traj["bboxes"]],
                "length": len(traj["points"]),
                "max_area": int(max(traj["areas"])) if traj["areas"] else 0,
            }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        csv_path = os.path.join(output_dir, "trajectory_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "track_id", "trajectory_length", "max_area",
                "dominant_class", "direction", "entry", "exit",
                "real_dist_m", "time_sec", "speed_kmh"
            ])
            speed_map = {s["track_id"]: s for s in self.speeds}
            for tid, traj in self.trajectories.items():
                classes = traj["classes"]
                vote = {}
                for c in classes:
                    vote[c] = vote.get(c, 0) + 1
                dominant = self.class_names.get(max(vote, key=vote.get), "unknown")
                first_y = traj["points"][0][1]
                last_y = traj["points"][-1][1]
                direction = "down" if last_y > first_y else "up"
                entered = 1 if tid in self.counted_ids and last_y > first_y else 0
                exited = 1 if tid in self.counted_ids and last_y < first_y else 0
                spd = speed_map.get(tid, {})
                writer.writerow([
                    tid, len(traj["points"]),
                    int(max(traj["areas"])) if traj["areas"] else 0,
                    dominant, direction, entered, exited,
                    spd.get("real_dist_m", ""),
                    spd.get("time_sec", ""),
                    spd.get("speed_kmh", ""),
                ])
        print(f"  轨迹数据已保存: {json_path}, {csv_path}")


def process_frame(frame, detector, tracker, counter, frame_idx, flow_info=None):
    detections = detector.detect(frame)
    tracked = tracker.update(detections)

    annotated = frame.copy()
    h, w = frame.shape[:2]

    if flow_info and flow_info["line_type"] == "vertical":
        entry_x = int(flow_info["entry_pct"] * w)
        exit_x = int(flow_info["exit_pct"] * w)
        extra_x = counter.extra_line_pos if counter.extra_line_pos else int(0.9 * w)
        cv2.line(annotated, (entry_x, 0), (entry_x, h), (0, 255, 0), 2)
        cv2.putText(annotated, "ENTRY", (entry_x + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.line(annotated, (exit_x, 0), (exit_x, h), (0, 0, 255), 2)
        cv2.putText(annotated, "EXIT", (exit_x + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.line(annotated, (extra_x, 0), (extra_x, h), (0, 165, 255), 2)
        cv2.putText(annotated, "AUX", (extra_x + 5, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        entry_y = int((flow_info["entry_pct"] if flow_info else ENTRY_PCT) * h)
        exit_y = int((flow_info["exit_pct"] if flow_info else EXIT_PCT) * h)
        extra_y = counter.extra_line_pos if counter.extra_line_pos else int(0.9 * h)
        cv2.line(annotated, (0, entry_y), (w, entry_y), (0, 255, 0), 2)
        cv2.putText(annotated, "ENTRY", (10, entry_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.line(annotated, (0, exit_y), (w, exit_y), (0, 0, 255), 2)
        cv2.putText(annotated, "EXIT", (10, exit_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.line(annotated, (0, extra_y), (w, extra_y), (0, 165, 255), 2)
        cv2.putText(annotated, "AUX", (10, extra_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for obj in tracked:
        tid = obj["track_id"]
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        cx, cy = int(obj["center"][0]), int(obj["center"][1])
        cls = obj["class"]
        conf = obj["conf"]
        area = obj["area"]
        bw, bh = obj["w"], obj["h"]

        color = colors[cls % len(colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = "ID:{} {:.2f}".format(tid, conf)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(annotated, (cx, cy), 4, color, -1)

        counter.update(tid, cx, cy, cls, area, bw, bh, frame_idx)

    stats = counter.get_stats()
    info = "IN: {}  OUT: {}  TOTAL: {}".format(
        stats['entry_count'], stats['exit_count'], stats['total_count'])
    cv2.putText(annotated, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    y_offset = 60
    cc = stats["class_counts"]
    for cls_id, name in counter.class_names.items():
        in_n = cc["entry"].get(cls_id, 0)
        out_n = cc["exit"].get(cls_id, 0)
        cls_info = "{}: IN={} OUT={}".format(name, in_n, out_n)
        cv2.putText(annotated, cls_info, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 22

    speed_info = "Avg Speed: {} km/h".format(stats['avg_speed_kmh'])
    cv2.putText(annotated, speed_info, (10, y_offset + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return annotated


def main():
    print("=" * 60)
    print("YOLO26 ONNX版 多目标跟踪 + 车流量统计")
    print("=" * 60)

    if not os.path.exists(ONNX_MODEL):
        print("[错误] ONNX模型文件不存在: {}".format(ONNX_MODEL))
        return

    print("\n[1/4] 加载ONNX模型: {}".format(ONNX_MODEL))
    detector = ONNXDetector(ONNX_MODEL, IMG_SIZE, CONF_THRESH)
    print("  推理设备: {}".format(detector.session.get_providers()[0]))

    print("\n[2/4] 打开视频源: {}".format(VIDEO_SOURCE))
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[错误] 无法打开视频源: {}".format(VIDEO_SOURCE))
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("  分辨率: {}x{}, FPS: {}, 总帧数: {}".format(width, height, fps, total_frames))

    print("\n[2.5/4] 分析车流方向...")
    flow_info = analyze_traffic_flow(VIDEO_SOURCE, ANALYSIS_FRAMES)
    if flow_info:
        print("  车流类型: {}".format(flow_info["direction"]))
        print("  主方向: {} (flow_x={:.1f}, flow_y={:.1f})".format(
            flow_info["primary"], flow_info["flow_x"], flow_info["flow_y"]))
        print("  检测线类型: {}".format(flow_info["line_type"]))
        entry_pos = int(flow_info["entry_pct"] * (width if flow_info["line_type"] == "vertical" else height))
        exit_pos = int(flow_info["exit_pct"] * (width if flow_info["line_type"] == "vertical" else height))
        print("  检测线位置: ENTRY={}, EXIT={}".format(entry_pos, exit_pos))
    else:
        print("  [警告] 无法分析车流, 使用默认水平线")
        flow_info = {
            "direction": "vertical", "line_type": "horizontal",
            "entry_pct": ENTRY_PCT, "exit_pct": EXIT_PCT,
            "primary": "down", "flow_x": 0, "flow_y": 0,
        }

    print("  速度标定: {} 米/像素".format(PIXELS_TO_METERS))

    video_writer = None
    if SAVE_VIDEO:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print("  输出视频: {}".format(output_path))

    tracker = SimpleTracker(MAX_DISAPPEARED, MAX_DISTANCE)
    entry_pos_val = flow_info["entry_pct"] * (width if flow_info["line_type"] == "vertical" else height)
    exit_pos_val = flow_info["exit_pct"] * (width if flow_info["line_type"] == "vertical" else height)
    counter = VehicleCounter(entry_pos_val, exit_pos_val, flow_info["line_type"],
                             MIN_TRAJ_LENGTH, MIN_AREA, fps, PIXELS_TO_METERS)
    counter.set_frame_size(width, height)
    print("  辅助线位置: {}".format(counter.extra_line_pos))

    print("\n[3/4] 处理视频中...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = process_frame(frame, detector, tracker, counter, frame_count, flow_info)

        if video_writer is not None:
            video_writer.write(annotated_frame)

        if SHOW_PREVIEW:
            preview = cv2.resize(annotated_frame, None, fx=0.7, fy=0.7)
            cv2.imshow("YOLO26 ONNX Tracking", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        if frame_count % 100 == 0:
            stats = counter.get_stats()
            print("  [{}/{}] IN:{} OUT:{} TOTAL:{}".format(
                frame_count, total_frames,
                stats['entry_count'], stats['exit_count'], stats['total_count']))

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    stats = counter.get_stats()
    print("\n[4/4] 处理完成!")
    print("  总帧数: {}".format(frame_count))
    print("  上行车辆(IN): {}".format(stats['entry_count']))
    print("  下行车辆(OUT): {}".format(stats['exit_count']))
    print("  总计: {}".format(stats['total_count']))
    print("  平均车速: {} km/h".format(stats['avg_speed_kmh']))
    print("  各车型统计:")
    for cls_id, name in counter.class_names.items():
        in_n = stats["class_counts"]["entry"].get(cls_id, 0)
        out_n = stats["class_counts"]["exit"].get(cls_id, 0)
        print("    {}: IN={}, OUT={}".format(name, in_n, out_n))

    if stats["speeds"]:
        print("\n  各车辆速度:")
        for s in stats["speeds"]:
            print("    ID:{} [{}] {} {}m / {}s = {} km/h".format(
                s['track_id'], s['class'], s['direction'],
                s['real_dist_m'], s['time_sec'], s['speed_kmh']))

    counter.save_trajectories(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("跟踪测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
