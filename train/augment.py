from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_dataloaders(data_dir: str, batch_size: int = 32):
    # Standard transform for EfficientNet/ResNet
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        # Simulate varying weather/lighting conditions required by the task book
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),

        # Simulate slight camera angle offsets
        transforms.RandomRotation(degrees=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),

        transforms.ToTensor(),
        # Standard ImageNet normalization parameters
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # The dataset automatically maps folder names (e.g., 'p11') to integer class IDs
    dataset = ImageFolder(root=data_dir, transform=train_transform)

    # Note: For your 5-fold CV requirement, you will split this dataset inside train_detector.py
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Save the class mapping for your inference engine later
    class_mapping = {v: k for k, v in dataset.class_to_idx.items()}

    return dataloader, class_mapping