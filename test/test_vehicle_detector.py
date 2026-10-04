import sys
import json
from pathlib import Path
import cv2

# Ensure traffic-vision-service root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.detector import VehicleDetector

# Input and Output Directories
BATCH_DIR = PROJECT_ROOT.parent / "raw_data" / "tt100k_2021" / "batch_test"
OUTPUT_DIR = PROJECT_ROOT.parent / "raw_data" / "tt100k_2021" / "batch_test_vehicle_output"

# 预期检测的车辆类别（COCO class IDs）
EXPECTED_VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

# BGR color mapping for drawing vehicle bounding boxes
DRAW_COLORS = {
    "car": (255, 144, 30),     # Deep Sky Blue
    "bus": (0, 165, 255),      # Orange
    "truck": (255, 0, 255)     # Magenta
}


def test_vehicle_detector_batch():
    assert BATCH_DIR.exists(), f"Directory not found: {BATCH_DIR}"

    # Create output directory for visualization
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化车辆检测器
    detector = VehicleDetector(
        weights_path="yolov8n.pt",
        conf_threshold=0.5,
        imgsz=960  # 提升小目标检测能力
    )

    image_files = [f for f in BATCH_DIR.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    assert len(image_files) > 0, f"No images found in {BATCH_DIR}"
    print(f"Testing vehicle detector on {len(image_files)} images from: {BATCH_DIR}")
    print(f"Visualized outputs will be saved to: {OUTPUT_DIR}\n")

    total_detections = 0
    total_images_with_vehicles = 0

    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to read image: {img_path.name}"
        h, w = img.shape[:2]
        vis_img = img.copy()

        # Run inference
        detections = detector.detect(img)

        # 统计
        total_detections += len(detections)
        if len(detections) > 0:
            total_images_with_vehicles += 1

        # Validate data contract
        assert isinstance(detections, list), f"Expected list output, got {type(detections)}"

        for det in detections:
            # 检查字段完整性
            assert "bbox" in det, f"Missing 'bbox' in detection: {det}"
            assert "confidence" in det, f"Missing 'confidence' in detection: {det}"
            assert "class_id" in det, f"Missing 'class_id' in detection: {det}"
            assert "class_name" in det, f"Missing 'class_name' in detection: {det}"

            # 检查bbox结构
            bbox = det["bbox"]
            assert all(k in bbox for k in ("x1", "y1", "x2", "y2")), \
                f"Invalid bbox structure: {bbox}"

            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

            # 检查bbox合法性
            assert x2 >= x1 and y2 >= y1, \
                f"Invalid bbox coordinates: {bbox}"

            # 检查坐标在图像范围内
            assert 0 <= x1 <= w and 0 <= x2 <= w, \
                f"bbox x-coordinates out of image bounds: {bbox}"
            assert 0 <= y1 <= h and 0 <= y2 <= h, \
                f"bbox y-coordinates out of image bounds: {bbox}"

            # 检查类别是否为车辆
            assert det["class_id"] in EXPECTED_VEHICLE_CLASSES, \
                f"Unexpected class_id: {det['class_id']}, expected vehicle classes only"

            # 检查confidence范围
            assert 0.0 <= det["confidence"] <= 1.0, \
                f"Confidence out of range: {det['confidence']}"

            # ==========================================
            # VISUALIZATION DRAWING
            # ==========================================
            class_name = det["class_name"]
            box_color = DRAW_COLORS.get(class_name, (0, 255, 0))

            # Draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 2)

            # Draw label background for readability
            label = f"{class_name} {det['confidence']:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_img, (x1, y1 - text_h - 8), (x1 + text_w, y1), box_color, -1)

            # Draw text
            text_color = (0, 0, 0) if class_name == "yellow" else (255, 255, 255)
            cv2.putText(vis_img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Save visualized image
        output_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(output_path), vis_img)

        # Print the formatted JSON payload for visual inspection
        print(f"[{idx}/{len(image_files)}] Payload for {img_path.name} "
              f"(detections: {len(detections)})")
        print(json.dumps(detections, indent=2))

    # 输出统计结果
    print(f"\n{'=' * 50}")
    print(f"Batch Detection Summary:")
    print(f"  Total images tested: {len(image_files)}")
    print(f"  Images with vehicles: {total_images_with_vehicles}")
    print(f"  Total vehicle detections: {total_detections}")
    print(f"  Average detections per image: {total_detections / len(image_files):.2f}")
    print(f"  Visualizations saved to: {OUTPUT_DIR}")
    print(f"{'=' * 50}")
    print("\nAll batch detection assertions passed.")


def test_vehicle_detector_single_image():
    """测试单张图片的详细输出格式"""
    # 找一个测试图片
    image_files = [f for f in BATCH_DIR.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(image_files) == 0:
        print("No images available for single image test.")
        return

    detector = VehicleDetector(weights_path="yolov8n.pt", conf_threshold=0.5)
    test_img_path = image_files[0]

    img = cv2.imread(str(test_img_path))
    detections = detector.detect(img)

    # 打印完整JSON格式（模拟最终输出）
    payload = {
        "frame_id": test_img_path.stem,
        "timestamp": "2026-09-02T11:35:36Z",
        "telemetry": {
            "vehicles": detections,
            "signs": []  # 路标由其他模块填充
        }
    }

    print(f"\n===== Complete Payload Example for {test_img_path.name} =====")
    print(json.dumps(payload, indent=2))

    # 验证payload结构
    assert "frame_id" in payload
    assert "telemetry" in payload
    assert "vehicles" in payload["telemetry"]
    assert isinstance(payload["telemetry"]["vehicles"], list)


if __name__ == "__main__":
    # 运行批量测试
    test_vehicle_detector_batch()

    # 可选：运行单图测试
    print("\n" + "=" * 60)
    test_vehicle_detector_single_image()