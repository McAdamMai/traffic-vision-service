import sys
import cv2
from pathlib import Path
from typing import Generator, Tuple

import numpy as np

# Inject project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class VideoStreamer:
    """
    Pure ingestion layer. Opens a video source and yields frames as a generator.
    Zero knowledge of neural networks, engines, or visualization.
    """
    def __init__(self, input_path: str):
        self.input_path = str(input_path)
        self.cap = cv2.VideoCapture(self.input_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.input_path}")

        # Expose metadata for downstream writers
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def stream(self, process_every_n_frames: int) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yields (frame_index, frame_array)."""
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % process_every_n_frames == 0:
                yield frame_idx, frame
            frame_idx += 1
            
        self.cap.release()