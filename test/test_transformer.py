import numpy as np
from pipeline.transforms import PipelineTransforms


def test_cropping_and_transforms():
    print("Testing Transforms...")
    transforms = PipelineTransforms()

    # Arrange: Mock a 1000x1000 image and a bounding box
    dummy_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    mock_bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}

    # Act 1: Test Cropping
    cropped = transforms.crop_bbox(dummy_image, mock_bbox)

    # Assert 1: Validate exact slice dimensions
    assert cropped.shape == (100, 100, 3), f"Expected 100x100 crop, got {cropped.shape}"

    # Act 2: Test PyTorch Preparation
    tensor = transforms.prepare_for_classification(cropped)

    # Assert 2: EfficientNet requires exactly [1, 3, 224, 224] shape
    assert list(tensor.shape) == [1, 3, 224, 224], f"Expected tensor shape [1, 3, 224, 224], got {list(tensor.shape)}"
    print("Transforms test passed.")


if __name__ == "__main__":
    test_cropping_and_transforms()