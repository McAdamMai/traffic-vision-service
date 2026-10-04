import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

from core.logger import logger


@dataclass
class SignCandidate:
    """Data structure for candidate regions extracted by YOLO."""
    bbox: dict  # {"x1": int, "y1": int, "x2": int, "y2": int}
    confidence: float  # YOLO bounding box confidence
    region: np.ndarray  # Cropped BGR image for EfficientNet
    color: str = "auto"  # Legacy field to maintain downstream compatibility
    shape: str = "auto"  # Legacy field to maintain downstream compatibility


# ============================================================================
# MODULE 1: Sign Extractor (YOLOv8) 
# ============================================================================
class YOLOSignDetector:
    """
    Dedicated road-sign localizer using YOLOv8.
    """

    def __init__(
            self,
            weights_path: str = "runs/traffic_sign_detector/yolov8n_640_300_test/weights/best.pt",
            conf_threshold: float = 0.25,
            imgsz: int = 1280,  # High resolution preserves tiny distant signs
            device: str = None
    ):
        logger.info(f"Loading YOLO Sign Detector weights from {weights_path}...")
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        self.model = YOLO(weights_path)
        if device:
            self.model.to(device)

        logger.info(f"YOLO Sign Detector initialized on device: {self.model.device}")

    def detect(self, image_array: np.ndarray) -> List[SignCandidate]:
        candidates = []
        if image_array is None or image_array.size == 0:
            return candidates

        h, w = image_array.shape[:2]

        # Run inference using the trained best.pt model
        results = self.model(
            image_array,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return candidates

        for box in results.boxes:
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())

            # Clamp coordinates strictly to image boundaries
            x1 = max(0, int(coords[0]))
            y1 = max(0, int(coords[1]))
            x2 = min(w, int(coords[2]))
            y2 = min(h, int(coords[3]))

            # Crop the region to send to EfficientNet
            region = image_array[y1:y2, x1:x2].copy()

            # Skip invalid or impossibly small crops
            if region.size == 0 or region.shape[0] < 5 or region.shape[1] < 5:
                continue

            candidates.append(SignCandidate(
                bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                confidence=round(conf, 3),
                region=region
            ))

        logger.debug(f"YOLO Sign Extractor found {len(candidates)} candidates.")
        return candidates


# ============================================================================
# MODULE 2: Sign Classifier (EfficientNet)
# ============================================================================
class SignClassifier:
    def __init__(
            self,
            weights_dir: str = "weights",
            conf_threshold: float = 0.70,
            device: str = None
    ):
        self.conf_threshold = conf_threshold
        self.device = torch.device(device) if device else torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Resolve absolute paths based on project root
        project_root = Path(__file__).resolve().parent.parent
        self.weights_dir = project_root / weights_dir

        # 1. Load the dynamic class mapping
        class_map_path = self.weights_dir / "class_map.json"
        logger.info(f"Loading class map from {class_map_path}...")

        with open(class_map_path, "r", encoding="utf-8") as f:
            # json stores keys as strings, convert them back to integers
            self.class_map = {int(k): v for k, v in json.load(f).items()}

        num_classes = len(self.class_map)
        logger.info(f"Initialized classifier for {num_classes} traffic sign classes.")

        # 2. Initialize EfficientNet Architecture
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(1280, num_classes)

        # 3. Load Trained Weights
        weights_path = self.weights_dir / "efficientnet_tt100k.pth"
        logger.info(f"Loading EfficientNet weights from {weights_path}...")

        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            logger.info(f"EfficientNet initialized successfully on device: {self.device}")
        except FileNotFoundError:
            logger.error(f"Weights {weights_path} not found! Run train_classifier.py first.")

        self.model.to(self.device)
        self.model.eval()

        # 4. Standard ImageNet Transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify(self, candidates: List[SignCandidate]) -> list:
        signs = []
        if not candidates:
            return signs

        for candidate in candidates:
            # Convert BGR (OpenCV) to RGB (PyTorch/EfficientNet)
            rgb_region = cv2.cvtColor(candidate.region, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb_region).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                pred_class_idx = torch.argmax(probs).item()
                confidence = probs[0][pred_class_idx].item()

            if confidence >= self.conf_threshold:
                # Map the integer index back to the TT100K string label (e.g., "pl100")
                sign_label = self.class_map.get(pred_class_idx, "unknown")

                signs.append({
                    "bbox": candidate.bbox,
                    "confidence": round(confidence, 3),
                    "class_name": sign_label,
                    "cv_metadata": {"color": candidate.color, "shape": candidate.shape}
                })

        return signs