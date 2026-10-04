import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from core.logger import logger


class PipelineTransforms:
    def __init__(self):
        logger.debug("Initializing inference transforms (Resize 224x224, Normalize).")
        self.inference_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def crop_bbox(image_array: np.ndarray, bbox: dict) -> np.ndarray:
        y1 = max(0, bbox["y1"])
        y2 = min(image_array.shape[0], bbox["y2"])
        x1 = max(0, bbox["x1"])
        x2 = min(image_array.shape[1], bbox["x2"])

        # Log a warning if the bounding box math collapsed or fell entirely outside the image
        if x1 >= x2 or y1 >= y2:
            logger.warning(f"Invalid crop dimensions generated for bbox: {bbox}. Returning empty array.")
            return np.array([])

        return image_array[y1:y2, x1:x2]

    def prepare_for_classification(self, cropped_array: np.ndarray) -> torch.Tensor:
        try:
            rgb_crop = cv2.cvtColor(cropped_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_crop)
            tensor = self.inference_transform(pil_image)
            return tensor.unsqueeze(0)
        except Exception as e:
            logger.error(f"Failed to transform cropped image: {str(e)}")
            raise