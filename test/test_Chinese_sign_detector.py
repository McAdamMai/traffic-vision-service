import sys
import json
from pathlib import Path
import cv2
import numpy as np

# Ensure traffic-vision-service root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.detector import YOLOSignDetector

# Input and Output Directories
BATCH_DIR = PROJECT_ROOT.parent / "raw_data" / "tt100k_2021" / "batch_test"
OUTPUT_DIR = PROJECT_ROOT.parent / "raw_data" / "tt100k_2021" / "batch_test_output"

# BGR color mapping for drawing bounding boxes (YOLO defaults to 'auto')
DRAW_COLORS = {
    "auto": (0, 255, 0),  # Green for YOLO detections
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255)
}


def test_sign_detector_batch():
    assert BATCH_DIR.exists(), f"Directory not found: {BATCH_DIR}"

    # Create output directory for visualization
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve path to your newly trained best.pt weights
    weights_path = PROJECT_ROOT / "runs" / "traffic_sign_detector" / "yolov8n_640_300_test" / "weights" / "best.pt"
    assert weights_path.exists(), f"Trained weights not found at {weights_path}"

    # Initialize YOLO Sign Extractor
    detector = YOLOSignDetector(
        weights_path=str(weights_path),
        conf_threshold=0.25,
        imgsz=1280
    )

    image_files = [f for f in BATCH_DIR.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    assert len(image_files) > 0, f"No images found in {BATCH_DIR}"
    print(f"Testing YOLO sign detector on {len(image_files)} images...")
    print(f"Visualized outputs will be saved to: {OUTPUT_DIR}\n")

    total_detections = 0
    total_images_with_signs = 0

    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to read image: {img_path.name}"
        h, w = img.shape[:2]
        vis_img = img.copy()

        # Run inference
        candidates = detector.detect(img)

        # Statistics
        total_detections += len(candidates)
        if len(candidates) > 0:
            total_images_with_signs += 1

        # Validate data contract
        assert isinstance(candidates, list), f"Expected list output, got {type(candidates)}"

        json_ready_candidates = []
        for cand in candidates:
            # 1. Check all attributes exist in the new dataclass
            assert hasattr(cand, "bbox"), "Missing 'bbox' in SignCandidate"
            assert hasattr(cand, "confidence"), "Missing 'confidence' in SignCandidate"
            assert hasattr(cand, "region"), "Missing 'region' in SignCandidate"
            assert hasattr(cand, "color"), "Missing legacy 'color' field"
            assert hasattr(cand, "shape"), "Missing legacy 'shape' field"

            # 2. Validate bounding box (dictionary format)
            bbox = cand.bbox
            assert isinstance(bbox, dict), f"bbox must be a dictionary, got: {type(bbox)}"
            assert all(k in bbox for k in ("x1", "y1", "x2", "y2")), f"Missing keys in bbox: {bbox}"

            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

            assert x2 >= x1 and y2 >= y1, f"Invalid bbox coordinates: {bbox}"
            assert 0 <= x1 <= w and 0 <= x2 <= w, f"bbox x-coords out of bounds: {bbox}"
            assert 0 <= y1 <= h and 0 <= y2 <= h, f"bbox y-coords out of bounds: {bbox}"

            # 3. Validate logical values
            assert 0.0 <= cand.confidence <= 1.0, f"Confidence out of range: {cand.confidence}"

            # 4. Validate the cropped region image array
            assert isinstance(cand.region, np.ndarray), "Region must be a NumPy array"
            assert cand.region.size > 0, "Region array cannot be empty"
            expected_h = y2 - y1
            expected_w = x2 - x1
            assert cand.region.shape[0] == expected_h, \
                f"Region height mismatch. Expected {expected_h}, got {cand.region.shape[0]}"
            assert cand.region.shape[1] == expected_w, \
                f"Region width mismatch. Expected {expected_w}, got {cand.region.shape[1]}"

            json_ready_candidates.append({
                "bbox": bbox,
                "color": cand.color,
                "shape": cand.shape,
                "confidence": round(cand.confidence, 3)
            })

            # ==========================================
            # VISUALIZATION DRAWING
            # ==========================================
            box_color = DRAW_COLORS.get(cand.color, (0, 255, 0))

            # Draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 3)

            # Draw label background for readability
            label = f"sign {cand.confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis_img, (x1, y1 - text_h - 10), (x1 + text_w, y1), box_color, -1)

            # Draw text
            cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Save visualized image
        output_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(output_path), vis_img)

        # Print JSON payload
        print(f"\n[{idx}/{len(image_files)}] Payload for {img_path.name} (candidates: {len(candidates)})")
        print(json.dumps(json_ready_candidates, indent=2))

    print(f"\n{'=' * 50}")
    print(f"Batch Detection Summary:")
    print(f"  Total images tested: {len(image_files)}")
    print(f"  Images with candidate signs: {total_images_with_signs}")
    print(f"  Total sign candidates extracted: {total_detections}")
    print(f"  Visualizations saved to: {OUTPUT_DIR}")
    print(f"{'=' * 50}")
    print("\nAll YOLO extractor batch assertions passed.")


def test_sign_detector_single_image():
    """Test detailed payload formatting for a single image."""
    image_files = [f for f in BATCH_DIR.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(image_files) == 0:
        return

    weights_path = PROJECT_ROOT / "runs" / "traffic_sign_detector" / "yolov8n_640_300_test" / "weights" / "best.pt"

    detector = YOLOSignDetector(
        weights_path=str(weights_path),
        conf_threshold=0.25,
        imgsz=1280
    )

    test_img_path = image_files[0]
    img = cv2.imread(str(test_img_path))
    candidates = detector.detect(img)

    json_ready_candidates = []
    for c in candidates:
        json_ready_candidates.append({
            "bbox": c.bbox,
            "cv_metadata": {
                "color": c.color,
                "shape": c.shape,
                "confidence": round(c.confidence, 3)
            },
            "status": "pending_classification"
        })

    payload = {
        "frame_id": test_img_path.stem,
        "timestamp": "2026-09-02T11:35:36Z",
        "telemetry": {
            "vehicles": [],  # To be filled by VehicleDetector
            "signs_candidates": json_ready_candidates  # To be classified by SignClassifier
        }
    }

    print(f"\n===== Complete Payload Example for {test_img_path.name} =====")
    print(json.dumps(payload, indent=2))

    # Validate payload structure
    assert "frame_id" in payload
    assert "telemetry" in payload
    assert "signs_candidates" in payload["telemetry"]
    assert "vehicles" in payload["telemetry"]
    assert isinstance(payload["telemetry"]["signs_candidates"], list)


if __name__ == "__main__":
    test_sign_detector_batch()

    print("\n" + "=" * 60)
    test_sign_detector_single_image()