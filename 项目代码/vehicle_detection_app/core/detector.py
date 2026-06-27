"""
ONNX 目标检测器 — onnxruntime 推理，不依赖 ultralytics/torch。
按类别分别做 NMS，避免不同类检测框互相误删。
"""

import cv2
import numpy as np
from typing import List

from utils.config import MODEL_TO_INTERNAL, MODEL_CLASS_NAMES


class Detector:

    def __init__(self, model_path: str = "", conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, img_size: int = 640):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self._session = None
        self._input_name = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available else ['CPUExecutionProvider']
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, options, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self.model_path = model_path

    def is_loaded(self) -> bool:
        return self._session is not None

    def update_thresholds(self, conf: float = None, iou: float = None):
        if conf is not None:
            self.conf_threshold = conf
        if iou is not None:
            self.iou_threshold = iou

    def detect(self, frame: np.ndarray) -> List[dict]:
        if self._session is None:
            return []

        h0, w0 = frame.shape[:2]
        blob, scale, dx, dy = self._preprocess(frame)
        raw = self._session.run(None, {self._input_name: blob})[0]
        boxes = self._postprocess(raw, scale, dx, dy, w0, h0)

        detections = []
        for box in boxes:
            model_id = int(box[5])
            internal_id = MODEL_TO_INTERNAL.get(model_id)
            if internal_id is None:
                continue
            detections.append({
                "bbox": (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                "class_id": internal_id,
                "class_name": MODEL_CLASS_NAMES.get(model_id, str(model_id)),
                "confidence": float(box[4]),
            })
        return detections

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        dx = (self.img_size - new_w) // 2
        dy = (self.img_size - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, 0), scale, dx, dy

    def _postprocess(self, raw, scale, dx, dy, orig_w, orig_h):
        output = raw[0]
        mask = output[:, 4] > self.conf_threshold
        dets = output[mask]
        if len(dets) == 0:
            return []

        boxes = dets[:, :4].copy()
        boxes[:, [0, 2]] -= dx
        boxes[:, [1, 3]] -= dy
        boxes /= scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
        scores = dets[:, 4]
        cls_ids = dets[:, 5].astype(int)
        return self._nms_per_class(boxes, scores, cls_ids)

    def _nms_per_class(self, boxes, scores, cls_ids):
        """按类别分别做 NMS"""
        result = []
        for cls_id in np.unique(cls_ids):
            mask = cls_ids == cls_id
            c_boxes = boxes[mask]
            c_scores = scores[mask]
            keep = self._nms(c_boxes, c_scores)
            for i in keep:
                result.append([*c_boxes[i], c_scores[i], cls_id])
        return result

    def _nms(self, boxes, scores):
        x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou <= self.iou_threshold)[0] + 1]
        return keep
