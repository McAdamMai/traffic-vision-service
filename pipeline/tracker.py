from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.logger import logger


@dataclass
class TrackedVehicle:
    track_id: int
    class_name: str
    confidence: float
    bbox: Dict[str, int]  # {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    bottom_center: Tuple[int, int]
    history: List[Tuple[int, int]] = field(default_factory=list)


class VehicleTracker:
    """
    Stateful multi-object tracker wrapping YOLOv8 ByteTrack.
    Maintains active track buffers and trajectory history across frames.
    """

    # COCO Class IDs: 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
    VEHICLE_CLASS_MAP = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        conf_threshold: float = 0.40,
        imgsz: int = 960,
        max_history: int = 30,
        device: Optional[str] = None,
    ):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.max_history = max_history

        logger.info(f"Loading YOLOv8 Tracker from {weights_path} on {self.device}...")
        self.model = YOLO(weights_path)
        
        # Per-track trajectory buffer: {track_id: deque([(x, y), ...], maxlen=max_history)}
        self.trajectory_buffer: Dict[int, deque] = {}

    def track(self, frame: np.ndarray) -> List[TrackedVehicle]:
        """
        Runs YOLO + ByteTrack on a single video frame.
        Maintains trajectory state between calls using persist=True.
        """
        if frame is None or frame.size == 0:
            return []

        # Run native ByteTrack inside Ultralytics
        results = self.model.track(
            source=frame,
            persist=True,               
            tracker="bytetrack.yaml",   
            classes=list(self.VEHICLE_CLASS_MAP.keys()),
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        tracked_vehicles: List[TrackedVehicle] = []
        
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return tracked_vehicles

        boxes = results[0].boxes
        track_ids = boxes.id.int().cpu().tolist()
        coords = boxes.xyxy.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.float().cpu().tolist()

        active_ids_this_frame = set()

        for track_id, coord, cls_id, conf in zip(track_ids, coords, class_ids, confidences):
            x1, y1, x2, y2 = coord
            active_ids_this_frame.add(track_id)

            # Calculate bottom-center contact patch on the road
            bottom_center = (int((x1 + x2) / 2), int(y2))

            # Update rolling trajectory buffer
            if track_id not in self.trajectory_buffer:
                self.trajectory_buffer[track_id] = deque(maxlen=self.max_history)
            
            self.trajectory_buffer[track_id].append(bottom_center)

            vehicle_obj = TrackedVehicle(
                track_id=track_id,
                class_name=self.VEHICLE_CLASS_MAP.get(cls_id, "vehicle"),
                confidence=round(conf, 3),
                bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                bottom_center=bottom_center,
                history=list(self.trajectory_buffer[track_id]),
            )
            tracked_vehicles.append(vehicle_obj)

        # Evict dead tracks if memory buffer grows excessively
        if len(self.trajectory_buffer) > 200:
            stale_keys = [k for k in self.trajectory_buffer if k not in active_ids_this_frame]
            for k in stale_keys:
                del self.trajectory_buffer[k]

        return tracked_vehicles