import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Generator, Union

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.tracker import VehicleTracker
from pipeline.detector import YOLOSignDetector, SignClassifier
from pipeline.traffic_light_classifier import TrafficLightClassifier
from core.logger import logger


class TrafficVisionEngine:
    """
    Central orchestrator for the Traffic Vision Pipeline.
    Supports multi-light consensus and stream ingestion.
    (Decoupled from rule evaluation)
    """

    def __init__(
        self,
        camera_config: Optional[Dict[str, Any]] = None,
        stream_fps: Optional[int] = None
    ):
        logger.info("Initializing Traffic Vision Engine...")
        
        config = camera_config or {}
        self.camera_id = config.get("camera_id", "cam_default")
        self.stream_fps = stream_fps or config.get("stream_fps", 30)

        # 1. Multi-Traffic Light ROIs
        raw_rois = config.get("traffic_light_rois") or config.get("traffic_light_roi") or []
        self.light_rois = []
        if raw_rois:
            if isinstance(raw_rois[0], (int, float)):
                self.light_rois = [raw_rois]
            elif len(raw_rois) == 4 and isinstance(raw_rois[0], list) and len(raw_rois[0]) == 2:
                self.light_rois = [raw_rois]
            else:
                self.light_rois = raw_rois

        self.light_classifier = TrafficLightClassifier() if self.light_rois else None

        # 2. State-Aware Vehicle Tracker
        self.vehicle_tracker = VehicleTracker(
            weights_path="yolov8n.pt",
            conf_threshold=0.40,
            imgsz=960
        )

        # 3. Sign Perception
        sign_weights = PROJECT_ROOT / "runs" / "traffic_sign_detector" / "yolov8n_640_full2700_prod" / "weights" / "best.pt"
        if not sign_weights.exists():
            sign_weights = PROJECT_ROOT / "runs" / "traffic_sign_detector" / "yolov8n_640_300_test" / "weights" / "best.pt"

        self.sign_detector = YOLOSignDetector(weights_path=str(sign_weights), conf_threshold=0.25, imgsz=1280)
        self.sign_classifier = SignClassifier(weights_dir="weights", conf_threshold=0.60)

        # 4. Multi-Tier Temporal Skipping
        self.light_update_interval = max(1, int(1.0 * self.stream_fps))   # Every ~1 second
        self.sign_update_interval = max(1, int(5.0 * self.stream_fps))    # Every ~5 seconds

        self.cached_light_state = "UNKNOWN"
        self.cached_signs: List[Dict[str, Any]] = []

        logger.info(f"Engine ready for [{self.camera_id}] (Base FPS: {self.stream_fps}).")

    def _get_rectified_crop(self, frame: np.ndarray, quad: list, target_w: int = 24, target_h: int = 72) -> np.ndarray:
        """Warps a 4-point quadrilateral [TL, TR, BR, BL] into an upright rectangular patch."""
        src_pts = np.array(quad, dtype=np.float32)
        dst_pts = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(frame, matrix, (target_w, target_h))

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[str] = None,
        frame_idx: int = 0
    ) -> Dict[str, Any]:
        """Processes a single frame array."""
        fid = frame_id or f"{self.camera_id}_{frame_idx}"
        
        if frame is None or frame.size == 0:
            logger.error(f"Received empty or invalid frame: {fid}")
            return self._build_empty_payload(fid)

        h, w = frame.shape[:2]
        current_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # STAGE 1: High-Frequency Vehicle Tracking
        tracked_vehicles = self.vehicle_tracker.track(frame)
        vehicle_payloads = [
            {
                "track_id": v.track_id,
                "class_name": v.class_name,
                "confidence": v.confidence,
                "bbox": v.bbox,
                "bottom_center": v.bottom_center,
                "history": getattr(v, "history", []),
                "direction": getattr(v, "direction", "UNKNOWN"),
                "velocity_px_frame": getattr(v, "velocity_px_frame", 0.0)
            }
            for v in tracked_vehicles
        ]

        # STAGE 2: Multi-Light Consensus (~1s interval)
        if self.light_classifier and self.light_rois:
            if frame_idx % self.light_update_interval == 0 or self.cached_light_state == "UNKNOWN":
                detected_states = []
                for roi in self.light_rois:
                    if len(roi) == 4 and isinstance(roi[0], (list, tuple)):
                        crop = self._get_rectified_crop(frame, roi)
                    elif len(roi) == 4 and isinstance(roi[0], (int, float)):
                        x1, y1, x2, y2 = roi
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w, int(x2)), min(h, int(y2))
                        crop = frame[y1:y2, x1:x2]
                    else:
                        continue

                    if crop.size > 0:
                        detected_states.append(self.light_classifier.get_state(crop))

                if "RED" in detected_states:
                    self.cached_light_state = "RED"
                elif "YELLOW" in detected_states:
                    self.cached_light_state = "YELLOW"
                elif "GREEN" in detected_states:
                    self.cached_light_state = "GREEN"
                else:
                    self.cached_light_state = "UNKNOWN"
            current_light = self.cached_light_state
        else:
            current_light = None

        # STAGE 3: Sign Recognition (~5s interval)
        if frame_idx % self.sign_update_interval == 0 or not self.cached_signs:
            sign_candidates = self.sign_detector.detect(frame)
            self.cached_signs = self.sign_classifier.classify(sign_candidates)

        return {
            "camera_id": self.camera_id,
            "frame_id": fid,
            "frame_idx": frame_idx,
            "timestamp": current_timestamp,
            "traffic_light_state": current_light,
            "telemetry": {
                "vehicles": vehicle_payloads,
                "signs": self.cached_signs
            }
        }

    def process_stream(
        self,
        video_source: Union[str, Path, int]
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Ingests a video file or stream directly, yields telemetry per frame,
        and ensures video capture resources are freed.
        """
        cap = cv2.VideoCapture(str(video_source))
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {video_source}")
            return

        src_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if src_fps > 0 and src_fps != self.stream_fps:
            self.stream_fps = src_fps
            self.light_update_interval = max(1, int(1.0 * src_fps))
            self.sign_update_interval = max(1, int(5.0 * src_fps))

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                yield self.process_frame(
                    frame=frame,
                    frame_id=f"{self.camera_id}_{frame_idx}",
                    frame_idx=frame_idx
                )
                frame_idx += 1
        finally:
            cap.release()
            logger.info(f"Released video source: {video_source}. Processed {frame_idx} frames.")

    def _build_empty_payload(self, frame_id: str) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_id": frame_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "traffic_light_state": None,
            "telemetry": {"vehicles": [], "signs": []},
            "error": "Invalid frame data"
        }