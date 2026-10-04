import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from core.logger import logger
from pipeline.engine import TrafficVisionEngine
import cv2

router = APIRouter(prefix="/jobs", tags=["Jobs"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = PROJECT_ROOT / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def process_video_background(video_path: Path, config: Dict[str, Any], job_id: str):
    """Worker task executed asynchronously off the main request thread."""
    logger.info(f"Starting pipeline execution for job: {job_id}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    engine = TrafficVisionEngine(camera_config=config, stream_fps=fps)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = f"{job_id}_{frame_idx}"
        telemetry = engine.process_frame(frame, frame_id=frame_id, frame_idx=frame_idx)
        
        # In practice: stream telemetry over WebSocket or append to a DuckDB/JSON file
        frame_idx += 1

    cap.release()
    logger.info(f"Job {job_id} processing completed. Processed {frame_idx} frames.")


@router.post("/upload")
async def upload_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Raw video file (.mp4, .avi, etc.)"),
    camera_config: str = Form(..., description="JSON string containing camera perception config")
):
    # 1. Parse JSON configuration
    try:
        config_dict = json.loads(camera_config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Malformed camera_config JSON: {str(exc)}")

    # 2. Assign unique Job ID and set target destination
    job_id = f"{config_dict.get('camera_id', 'cam')}_{uuid.uuid4().hex[:8]}"
    file_extension = Path(video.filename).suffix or ".mp4"
    dest_path = UPLOAD_DIR / f"{job_id}{file_extension}"

    # 3. Stream binary directly to disk
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write file to disk: {str(exc)}")
    finally:
        await video.close()

    # 4. Enqueue background execution
    background_tasks.add_task(process_video_background, dest_path, config_dict, job_id)

    return {
        "status": "QUEUED",
        "job_id": job_id,
        "saved_path": str(dest_path.relative_to(PROJECT_ROOT)),
        "applied_config": config_dict
    }