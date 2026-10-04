import sys
import os
import tempfile
from pathlib import Path
import yaml
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.engine import TrafficVisionEngine
from pipeline.detector import YOLOSignDetector, SignClassifier

app = FastAPI(title="Traffic Vision Analysis API")
CONFIG_DIR = PROJECT_ROOT / "configs" / "cameras"

# ---------------------------------------------------------------------------
# GLOBAL MODEL INITIALIZATION
# Load sign models once at startup so they don't reload on every HTTP request
# ---------------------------------------------------------------------------
sign_weights = PROJECT_ROOT / "runs" / "traffic_sign_detector" / "yolov8n_640_full2700_prod" / "weights" / "best.pt"
if not sign_weights.exists():
    sign_weights = PROJECT_ROOT / "runs" / "traffic_sign_detector" / "yolov8n_640_300_test" / "weights" / "best.pt"

global_sign_detector = YOLOSignDetector(weights_path=str(sign_weights), conf_threshold=0.25, imgsz=1280)
global_sign_classifier = SignClassifier(weights_dir="weights", conf_threshold=0.60)


@app.post("/api/v1/analyze/video")
async def analyze_video_violations(
    camera_id: str = Form(...),
    video: UploadFile = File(...)
):
    """
    Accepts an MP4 video file and a camera_id. 
    Returns all traffic violations detected in the clip.
    """
    config_path = CONFIG_DIR / f"{camera_id}.yaml"

    # 1. Validate configuration exists
    if not config_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Configuration for camera '{camera_id}' not found."
        )

    with open(config_path, "r") as f:
        camera_config = yaml.safe_load(f)

    # 2. Save the uploaded video to a temporary file
    # OpenCV's VideoCapture cannot read video streams directly from HTTP memory buffers
    suffix = Path(video.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        content = await video.read()
        temp_video.write(content)
        temp_video_path = temp_video.name

    try:
        # 3. Initialize Engine and Video Capture
        engine = TrafficVisionEngine(camera_config=camera_config)
        cap = cv2.VideoCapture(temp_video_path)
        
        all_violations = []
        frame_idx = 0

        # 4. Process the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            payload = engine.process_frame(
                frame=frame, 
                frame_id=f"frame_{frame_idx}", 
                frame_idx=frame_idx
            )
            
            # If violations occurred in this frame, save them
            violations = payload["telemetry"].get("violations", [])
            if violations:
                all_violations.append({
                    "frame_idx": frame_idx,
                    "timestamp": payload["timestamp"],
                    "events": violations
                })
                
            frame_idx += 1

        cap.release()

        # 5. Return aggregated results
        return {
            "status": "success",
            "camera_id": camera_id,
            "frames_processed": frame_idx,
            "total_violation_events": len(all_violations),
            "violations": all_violations
        }

    finally:
        # 6. Cleanup: Delete the temporary video file from the server
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


@app.post("/api/v1/analyze/sign")
async def analyze_traffic_sign(image: UploadFile = File(...)):
    """
    Accepts an image file (JPG/PNG).
    Locates and classifies all traffic signs in the image.
    """
    # 1. Read the image bytes directly into memory
    contents = await image.read()
    np_array = np.frombuffer(contents, np.uint8)
    
    # 2. Decode the bytes into an OpenCV BGR array
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    # 3. Run the two-stage sign pipeline using the globally loaded models
    sign_candidates = global_sign_detector.detect(frame)
    classified_signs = global_sign_classifier.classify(sign_candidates)

    return {
        "status": "success",
        "signs_detected": len(classified_signs),
        "results": classified_signs
    }