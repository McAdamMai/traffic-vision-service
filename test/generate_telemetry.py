import sys
import json
import cv2
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.engine import TrafficVisionEngine

def generate_telemetry_headless(config_name="cam_00.yaml", max_frames=None):
    # Resolve Paths
    config_path = PROJECT_ROOT / "configs" / "cameras" / config_name
    video_dir = PROJECT_ROOT / "raw_data" / "videos"
    output_dir = PROJECT_ROOT / "test" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(video_dir.glob("*.mp4")))
    if not video_files:
        print(f"Error: No .mp4 files found in {video_dir}")
        return

    target_video = video_files[0]
    output_file = output_dir / f"{target_video.stem}_telemetry.jsonl"

    # Initialize Engine
    with open(config_path, "r", encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)

    engine = TrafficVisionEngine(camera_config=camera_config)
    cap = cv2.VideoCapture(str(target_video))
    
    frame_idx = 0
    print(f"Starting headless extraction: {target_video.name}")
    print(f"Outputting to: {output_file}")

    # Process and stream straight to JSONL
    with open(output_file, "w", encoding="utf-8") as out_f:
        while cap.isOpened():
            if max_frames and frame_idx >= max_frames:
                break
                
            ret, frame = cap.read()
            if not ret:
                break

            # Pure inference, no drawing
            telemetry = engine.process_frame(frame, frame_idx=frame_idx)
            out_f.write(json.dumps(telemetry, ensure_ascii=False) + "\n")
            
            # Lightweight progress ping (every ~10 seconds of 30fps video)
            if frame_idx % 300 == 0 and frame_idx > 0:
                print(f"Processed {frame_idx} frames...")
                
            frame_idx += 1

    cap.release()
    print(f"Extraction complete. {frame_idx} frames saved to {output_file.name}")

if __name__ == "__main__":
    # Remove max_frames limit to process the entire video
    generate_telemetry_headless(max_frames=900)