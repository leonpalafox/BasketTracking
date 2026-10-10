"""Player detectors. Each detector maps a BGR frame to `sv.Detections` of players only."""

import os

import numpy as np
import supervision as sv

# Roboflow's public basketball model (RF-DETR-S fine-tuned on 10 classes), from
# https://blog.roboflow.com/identify-basketball-players/
ROBOFLOW_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
ROBOFLOW_PLAYER_CLASSES = {
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
}


class RoboflowDetector:
    """Roboflow-hosted basketball model. Needs ROBOFLOW_API_KEY and access to detect.roboflow.com."""

    def __init__(self, model_id=ROBOFLOW_MODEL_ID, confidence=0.4, api_key=None):
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient

        api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise SystemExit("Set ROBOFLOW_API_KEY (free key from https://app.roboflow.com/settings/api).")
        self.client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)
        self.client.configure(InferenceConfiguration(confidence_threshold=confidence))
        self.model_id = model_id

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        result = self.client.infer(frame, model_id=self.model_id)
        det = sv.Detections.from_inference(result)
        if len(det) == 0:
            return det
        keep = np.isin(det.data["class_name"], list(ROBOFLOW_PLAYER_CLASSES))
        return det[keep]


class YoloDetector:
    """Generic COCO YOLO 'person' detector. Also picks up refs, coaches and spectators."""

    def __init__(self, weights="yolo11m.pt", confidence=0.25, imgsz=960):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.confidence = confidence
        self.imgsz = imgsz

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        result = self.model(frame, classes=[0], conf=self.confidence, imgsz=self.imgsz, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)


def make_detector(name: str, **kwargs):
    if name == "roboflow":
        return RoboflowDetector(**kwargs)
    if name == "yolo":
        return YoloDetector(**kwargs)
    raise ValueError(f"unknown detector {name!r}")
