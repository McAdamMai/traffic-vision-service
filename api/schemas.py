import time
from pydantic import BaseModel, Field
from typing import List, Optional

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectedObject(BaseModel):
    category: str
    confidence: float
    bbox: BoundingBox

class TrafficLightStatus(BaseModel):
    state: str = Field(..., description="RED, YELLOW, GREEN, or UNKNOWN")
    confidence: float

class InferenceResponse(BaseModel):
    status: str = Field(default="success", description="'success' or 'error'")
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    frame_id: Optional[str] = None
    traffic_light: Optional[TrafficLightStatus] = None
    signs: List[DetectedObject] = []
    vehicles: List[DetectedObject] = []
    inference_time_ms: float