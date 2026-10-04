import sys
import cv2
import yaml
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.traffic_light_classifier import TrafficLightClassifier
from core.logger import logger


def get_rectified_crop(frame: np.ndarray, quad: list, target_w: int = 24, target_h: int = 72) -> np.ndarray:
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


def debug_traffic_lights(
    config_file: str = "cam_00.yaml",
    target_fps: int = 1,          # Sample at 1 FPS
    max_seconds: int = 30,        # Test up to 30 seconds of video
    save_all_sampled_frames: bool = True
):
    # 1. Resolve paths
    config_path = PROJECT_ROOT / "configs" / "cameras" / config_file
    video_dir = PROJECT_ROOT / "raw_data" / "videos"
    output_dir = PROJECT_ROOT / "test" / "output" / "debug_lights"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        logger.error(f"Config not found at: {config_path}")
        return

    video_files = sorted(list(video_dir.glob("*.mp4")))
    if not video_files:
        logger.error(f"No video files found in: {video_dir}")
        return

    video_path = video_files[0]
    logger.info(f"Analyzing Video: {video_path.name}")
    logger.info(f"Loading Config:  {config_path.name}")

    # 2. Parse Camera Configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_rois = config.get("traffic_light_rois") or config.get("traffic_light_roi") or []
    light_rois = []
    if raw_rois:
        if isinstance(raw_rois[0], (int, float)):
            light_rois = [raw_rois]
        elif len(raw_rois) == 4 and isinstance(raw_rois[0], list) and len(raw_rois[0]) == 2:
            light_rois = [raw_rois]
        else:
            light_rois = raw_rois

    stop_line = config.get("stop_line")
    restricted_zones = config.get("restricted_zones", {})

    logger.info(f"Found {len(light_rois)} traffic light ROI(s) in config.")

    # 3. Initialize classifier and capture
    classifier = TrafficLightClassifier()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    frame_step = max(1, int(native_fps / target_fps))
    logger.info(f"Stream: {src_w}x{src_h} @ {native_fps} FPS. Sampling every {frame_step} frames (~{target_fps} FPS).")

    raw_frame_idx = 0
    sampled_sec = 0

    while cap.isOpened() and sampled_sec < max_seconds:
        ret, frame = cap.read()
        if not ret:
            break

        current_idx = raw_frame_idx
        raw_frame_idx += 1

        if current_idx % frame_step != 0:
            continue

        annotated_frame = frame.copy()

        # -----------------------------------------------------------------
        # STEP A: Evaluate and Draw Traffic Light ROIs
        # -----------------------------------------------------------------
        for idx, roi in enumerate(light_rois):
            is_quad = (len(roi) == 4 and isinstance(roi[0], (list, tuple)))
            
            if is_quad:
                crop = get_rectified_crop(frame, roi)
                label_pt = (int(roi[0][0]), max(15, int(roi[0][1]) - 5))
            elif len(roi) == 4 and isinstance(roi[0], (int, float)):
                x1, y1, x2, y2 = [int(v) for v in roi]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(src_w, x2), min(src_h, y2)
                crop = frame[y1:y2, x1:x2]
                label_pt = (x1, max(15, y1 - 5))
            else:
                continue

            if crop is not None and crop.size > 0:
                predicted_state = classifier.get_state(crop)
                
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                mask_r1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                mask_r2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
                red_px = cv2.countNonZero(mask_r1) + cv2.countNonZero(mask_r2)
                yellow_px = cv2.countNonZero(cv2.inRange(hsv, np.array([15, 70, 50]), np.array([35, 255, 255])))
                green_px = cv2.countNonZero(cv2.inRange(hsv, np.array([40, 70, 50]), np.array([90, 255, 255])))
            else:
                predicted_state = "INVALID_CROP"
                red_px = yellow_px = green_px = 0

            logger.info(
                f"[Sec {sampled_sec:02d} | Frame {current_idx:04d} | Light #{idx}] "
                f"State: {predicted_state:<7} | "
                f"Counts: R={red_px:<3}, Y={yellow_px:<3}, G={green_px:<3}"
            )

            if sampled_sec == 0 and crop.size > 0:
                zoom_crop = cv2.resize(crop, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
                zoom_path = output_dir / f"sec0_light_{idx}_zoom4x.jpg"
                cv2.imwrite(str(zoom_path), zoom_crop)

            box_color = (0, 0, 255) if predicted_state == "RED" else (0, 255, 0) if predicted_state == "GREEN" else (0, 255, 255)
            
            if is_quad:
                pts = np.array(roi, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], isClosed=True, color=box_color, thickness=2)
            else:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                
            cv2.putText(annotated_frame, f"L{idx}:{predicted_state}", label_pt,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # -----------------------------------------------------------------
        # STEP B: Draw Stop Line
        # -----------------------------------------------------------------
        if stop_line and len(stop_line) == 2:
            p1 = tuple(int(v) for v in stop_line[0])
            p2 = tuple(int(v) for v in stop_line[1])
            cv2.line(annotated_frame, p1, p2, (0, 0, 255), 3)
            cv2.putText(annotated_frame, "STOP LINE", (p1[0], max(20, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # -----------------------------------------------------------------
        # STEP C: Draw Restricted Zones
        # -----------------------------------------------------------------
        for zone_name, polygon in restricted_zones.items():
            if len(polygon) >= 3:
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
                cv2.putText(annotated_frame, zone_name, tuple(pts[0][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        if save_all_sampled_frames:
            snapshot_path = output_dir / f"overlay_sec_{sampled_sec:02d}.jpg"
            cv2.imwrite(str(snapshot_path), annotated_frame)

        sampled_sec += 1

    cap.release()
    logger.info(f"Diagnostic complete. Processed {sampled_sec} seconds (1 frame/sec). Results saved in '{output_dir}'.")


if __name__ == "__main__":
    debug_traffic_lights(target_fps=1, max_seconds=30)