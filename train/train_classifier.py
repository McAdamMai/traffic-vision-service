import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ============================================================
# Path Resolution & Hardware Config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "train"
OUTPUT_DIR = PROJECT_ROOT / "weights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
VAL_SPLIT = 0.20
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_gpu_mem(device):
    """Safely poll Apple Silicon memory."""
    try:
        if str(device) == "mps":
            return f"{torch.mps.current_allocated_memory() / 1E9:.2f}G"
    except Exception:
        pass
    return "N/A"


def train_classifier():
    print("\n" + "=" * 70)
    print("TT100K EFFICIENTNET SIGN CLASSIFIER")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")

    assert DATA_DIR.exists(), f"Directory not found: {DATA_DIR}"

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset & Split
    raw_dataset = datasets.ImageFolder(root=str(DATA_DIR))
    total_samples = len(raw_dataset)
    num_classes = len(raw_dataset.classes)

    val_size = int(total_samples * VAL_SPLIT)
    train_size = total_samples - val_size
    base_train, base_val = random_split(raw_dataset, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(42))

    train_dataset = TransformedSubset(base_train, transform=train_transforms)
    val_dataset = TransformedSubset(base_val, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model Setup
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nImage sizes 224 train, 224 val")
    print(f"Using 0 dataloader workers")
    print(f"Starting training for {EPOCHS} epochs...\n")

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ---------------------------------------------------------
        # TRAINING PHASE
        # ---------------------------------------------------------
        print(f"{'Epoch':>11} {'GPU_mem':>10} {'train_loss':>12} {'Instances':>11} {'Size':>10}")

        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Native Progress Bar Math
            progress = (i + 1) / total_batches
            filled = int(12 * progress)
            bar = '━' * filled + ' ' * (12 - filled)

            ep_str = f"{epoch + 1}/{EPOCHS}"
            mem_str = get_gpu_mem(DEVICE)
            loss_str = f"{loss.item():.4f}"

            # Overwrite terminal line using \r
            sys.stdout.write(
                f"\r{ep_str:>11} {mem_str:>10} {loss_str:>12} {inputs.size(0):>11} {'224:':>5} {int(progress * 100):3d}% ╸{bar}╸ {i + 1}/{total_batches}")
            sys.stdout.flush()

        train_loss = running_loss / len(train_dataset)
        print()  # Move to new line after loop finishes

        # ---------------------------------------------------------
        # VALIDATION PHASE
        # ---------------------------------------------------------
        print(f"{'Class':>11} {'Images':>10} {'Instances':>12} {'val_loss':>11} {'Accuracy':>10}")

        model.eval()
        val_loss, correct_val = 0.0, 0
        total_val_batches = len(val_loader)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()

                # Native Progress Bar Math
                progress = (i + 1) / total_val_batches
                filled = int(12 * progress)
                bar = '━' * filled + ' ' * (12 - filled)

                acc_current = correct_val / max(1, ((i + 1) * BATCH_SIZE))
                v_loss_str = f"{loss.item():.4f}"

                # CRITICAL FIX: Changed {acc_current:.4f:>10} to {acc_current:>10.4f}
                sys.stdout.write(
                    f"\r{'all':>11} {len(val_dataset):>10} {inputs.size(0):>12} {v_loss_str:>11} {acc_current:>10.4f} {int(progress * 100):3d}% ╸{bar}╸ {i + 1}/{total_val_batches}")
                sys.stdout.flush()

        val_loss = val_loss / len(val_dataset)
        val_acc = correct_val / len(val_dataset)

        # CRITICAL FIX: Changed {val_acc:.4f:>10} to {val_acc:>10.4f}
        sys.stdout.write(
            f"\r{'all':>11} {len(val_dataset):>10} {len(val_dataset):>12} {val_loss:>11.4f} {val_acc:>10.4f} 100% ╸{'━' * 12}╸ {total_val_batches}/{total_val_batches}\n")
        sys.stdout.flush()
        print()  # Blank line separator

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(OUTPUT_DIR / "efficientnet_tt100k.pth"))

    # Save Class Map
    class_map = {idx: name for name, idx in raw_dataset.class_to_idx.items()}
    with open(OUTPUT_DIR / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Weights saved to: {OUTPUT_DIR / 'efficientnet_tt100k.pth'}")


if __name__ == "__main__":
    train_classifier()