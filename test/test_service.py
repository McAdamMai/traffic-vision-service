from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import logger
from pipeline.engine import TrafficVisionEngine


def run_video_export(input_video_path: str | Path | None = None):
    # 1. Resolve paths
    config_path = PROJECT_ROOT / "configs" / "cameras" / "cam_01.yaml"
    video_dir = PROJECT_ROOT / "raw_data" / "videos"
    output_dir = PROJECT_ROOT / "test" / "output"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        logger.error(f"Config not found at: {config_path}")
        return

    # Determine target video from argument or fallback to directory
    if input_video_path:
        target_video = Path(input_video_path).resolve()
        if not target_video.exists():
            logger.error(f"Target video not found at: {target_video}")
            return
    else:
        video_files = sorted(list(video_dir.glob("*.mp4")))
        if not video_files:
            logger.error(f"No video files found in: {video_dir}")
            return
        target_video = video_files[0]

    output_jsonl = output_dir / f"{target_video.stem}_telemetry.jsonl"
    output_mp4 = output_dir / f"{target_video.stem}_annotated.mp4"

    logger.info(f"Target Video: {target_video.name}")
    logger.info(f"Output Video: {output_mp4.name}")

    # 2. Load config and init engine
    with open(config_path, "r", encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)

    engine = TrafficVisionEngine(camera_config=camera_config)

    # 3. Setup Video Capture and Writer
    cap = cv2.VideoCapture(str(target_video))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {target_video}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (width, height))

    # Test limit: Process first 30 seconds to save time. Set to None for full video.
    max_frames = fps * 30
    frame_idx = 0

    logger.info(f"Generating video... (Limit: {max_frames} frames)")

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame through engine
            telemetry = engine.process_frame(frame, frame_idx=frame_idx)
            out_f.write(json.dumps(telemetry, ensure_ascii=False) + "\n")

            annotated = frame.copy()

            # --- A. Draw Geometries from Config ---
            # Traffic Light ROIs
            for roi in engine.light_rois:
                if len(roi) == 4 and isinstance(roi[0], (list, tuple)):
                    pts = np.array(roi, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated, [pts], isClosed=True, color=(0, 255, 255), thickness=2
                    )
                elif len(roi) == 4 and isinstance(roi[0], (int, float)):
                    x1, y1, x2, y2 = [int(v) for v in roi]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Stop Line
            stop_line = camera_config.get("stop_line", [])
            if len(stop_line) == 2:
                p1 = tuple(int(v) for v in stop_line[0])
                p2 = tuple(int(v) for v in stop_line[1])
                cv2.line(annotated, p1, p2, (0, 0, 255), 3)

            # Restricted Zones
            for z_name, z_pts in camera_config.get("restricted_zones", {}).items():
                if len(z_pts) >= 3:
                    pts = np.array(z_pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated, [pts], isClosed=True, color=(255, 0, 255), thickness=2
                    )

            # --- B. Draw Vehicles ---
            vehicles = telemetry.get("telemetry", {}).get("vehicles", [])
            for v in vehicles:
                box = v["bbox"]
                x1, y1, x2, y2 = (
                    int(box["x1"]),
                    int(box["y1"]),
                    int(box["x2"]),
                    int(box["y2"]),
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 150, 0), 2)

                label = f"#{v['track_id']} {v['class_name']}"
                if v.get("velocity_px_frame", 0) > 0:
                    label += f" ({v['direction']})"

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 150, 0),
                    2,
                )

            # --- C. Draw Global Light State ---
            light_state = telemetry.get("traffic_light_state", "UNKNOWN")
            color = (
                (0, 0, 255)
                if light_state == "RED"
                else (0, 255, 0)
                if light_state == "GREEN"
                else (0, 255, 255)
            )

            cv2.rectangle(annotated, (20, 20), (450, 70), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                f"SYSTEM: {light_state}",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3,
            )

            # Write to video file
            video_writer.write(annotated)

            if frame_idx % fps == 0:
                logger.info(f"Rendered {frame_idx // fps}s / {max_frames // fps}s")

            frame_idx += 1

    cap.release()
    video_writer.release()
    logger.info(f"Video export complete. Saved to: {output_mp4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export annotated traffic vision video.")
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="Path to the input video file (.mp4). Defaults to the first video in raw_data/videos.",
    )
    args = parser.parse_args()

    run_video_export(args.video_path)