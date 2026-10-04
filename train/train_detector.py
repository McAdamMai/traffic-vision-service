from pathlib import Path
import shutil

from ultralytics import YOLO


# ============================================================
# Project paths
# ============================================================

# train/train_detector.py
# project root is one level above this file

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT.parent
    / "raw_data"
    / "tt100k_yolo"
)

ORIGINAL_DATASET_YAML = (
    DATASET_ROOT
    / "dataset.yaml"
)

PRETRAINED_MODEL = (
    PROJECT_ROOT
    / "yolov8n.pt"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "traffic_sign_detector"
)


# ============================================================
# Quick-test dataset configuration
# ============================================================

# Number of training images to use for this experiment.
TRAIN_IMAGES = 300

# Number of validation images.
#
# Your existing dataset already contains 300 validation images,
# so we use all 300 of them.
VAL_IMAGES = 300

QUICK_DATASET_DIR = (
    DATASET_ROOT
    / "quick_300"
)

QUICK_DATASET_YAML = (
    QUICK_DATASET_DIR
    / "dataset.yaml"
)


# ============================================================
# Training configuration — Apple M2 Pro
# ============================================================

# This is a QUICK PIPELINE TEST.
#
# It is NOT the final training configuration.
EPOCHS = 20

# 640 is faster than 768 and is sufficient for a quick test.
IMAGE_SIZE = 640

# Conservative batch size for M2.
BATCH_SIZE = 8

# Apple Silicon GPU.
DEVICE = "mps"

# Keep workers low on macOS.
WORKERS = 2

EXPERIMENT_NAME = "yolov8n_640_300_test"


# ============================================================
# Create quick 300-image dataset
# ============================================================

def create_quick_dataset():

    print("\n")
    print("=" * 70)
    print("CREATING QUICK 300-IMAGE DATASET")
    print("=" * 70)

    original_train_images = (
        DATASET_ROOT
        / "images"
        / "train"
    )

    original_train_labels = (
        DATASET_ROOT
        / "labels"
        / "train"
    )

    original_val_images = (
        DATASET_ROOT
        / "images"
        / "val"
    )

    original_val_labels = (
        DATASET_ROOT
        / "labels"
        / "val"
    )

    # --------------------------------------------------------
    # Validate source directories
    # --------------------------------------------------------

    if not original_train_images.exists():
        raise FileNotFoundError(
            f"Training images not found:\n"
            f"{original_train_images}"
        )

    if not original_train_labels.exists():
        raise FileNotFoundError(
            f"Training labels not found:\n"
            f"{original_train_labels}"
        )

    if not original_val_images.exists():
        raise FileNotFoundError(
            f"Validation images not found:\n"
            f"{original_val_images}"
        )

    if not original_val_labels.exists():
        raise FileNotFoundError(
            f"Validation labels not found:\n"
            f"{original_val_labels}"
        )

    # --------------------------------------------------------
    # Remove previous quick dataset
    # --------------------------------------------------------

    if QUICK_DATASET_DIR.exists():

        print(
            "\nRemoving previous quick dataset:"
        )

        print(
            f"  {QUICK_DATASET_DIR}"
        )

        shutil.rmtree(
            QUICK_DATASET_DIR
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    quick_train_images = (
        QUICK_DATASET_DIR
        / "images"
        / "train"
    )

    quick_train_labels = (
        QUICK_DATASET_DIR
        / "labels"
        / "train"
    )

    quick_val_images = (
        QUICK_DATASET_DIR
        / "images"
        / "val"
    )

    quick_val_labels = (
        QUICK_DATASET_DIR
        / "labels"
        / "val"
    )

    quick_train_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    quick_train_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    quick_val_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    quick_val_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find training images
    # --------------------------------------------------------

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }

    train_images = sorted(
        [
            path
            for path in original_train_images.iterdir()
            if (
                path.is_file()
                or path.is_symlink()
            )
            and path.suffix.lower()
            in image_extensions
        ]
    )

    if len(train_images) < TRAIN_IMAGES:

        raise RuntimeError(
            f"Only {len(train_images)} training images "
            f"found, but {TRAIN_IMAGES} requested."
        )

    # --------------------------------------------------------
    # Select 300 training images
    # --------------------------------------------------------

    selected_train_images = (
        train_images[
            :TRAIN_IMAGES
        ]
    )

    print(
        f"\nSelecting "
        f"{len(selected_train_images)} "
        f"training images."
    )

    # --------------------------------------------------------
    # Link training images + labels
    # --------------------------------------------------------

    for image_path in selected_train_images:

        destination_image = (
            quick_train_images
            / image_path.name
        )

        label_path = (
            original_train_labels
            / f"{image_path.stem}.txt"
        )

        destination_label = (
            quick_train_labels
            / label_path.name
        )

        # Create image symlink
        destination_image.symlink_to(
            image_path.resolve()
        )

        # Create label symlink
        if label_path.exists():

            destination_label.symlink_to(
                label_path.resolve()
            )

    # --------------------------------------------------------
    # Use all existing validation images
    # --------------------------------------------------------

    val_images = sorted(
        [
            path
            for path in original_val_images.iterdir()
            if (
                path.is_file()
                or path.is_symlink()
            )
            and path.suffix.lower()
            in image_extensions
        ]
    )

    selected_val_images = (
        val_images[
            :VAL_IMAGES
        ]
    )

    print(
        f"Using "
        f"{len(selected_val_images)} "
        f"validation images."
    )

    # --------------------------------------------------------
    # Link validation images + labels
    # --------------------------------------------------------

    for image_path in selected_val_images:

        destination_image = (
            quick_val_images
            / image_path.name
        )

        label_path = (
            original_val_labels
            / f"{image_path.stem}.txt"
        )

        destination_label = (
            quick_val_labels
            / label_path.name
        )

        destination_image.symlink_to(
            image_path.resolve()
        )

        if label_path.exists():

            destination_label.symlink_to(
                label_path.resolve()
            )

    # --------------------------------------------------------
    # Create dataset.yaml
    # --------------------------------------------------------

    dataset_yaml = (
        "# Quick 300-image TT100K experiment\n"
        "\n"
        f"path: {QUICK_DATASET_DIR.as_posix()}\n"
        "\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: traffic_sign\n"
    )

    QUICK_DATASET_YAML.write_text(
        dataset_yaml,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        "\nQuick dataset created:"
    )

    print(
        f"  Train: "
        f"{len(selected_train_images)}"
    )

    print(
        f"  Val:   "
        f"{len(selected_val_images)}"
    )

    print(
        f"\nDataset YAML:"
    )

    print(
        f"  {QUICK_DATASET_YAML}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TT100K TRAFFIC SIGN DETECTOR")
    print("300-IMAGE QUICK TRAINING TEST")
    print("=" * 70)

    print("\nProject root:")
    print(f"  {PROJECT_ROOT}")

    print("\nOriginal dataset:")
    print(f"  {ORIGINAL_DATASET_YAML}")

    print("\nQuick dataset:")
    print(f"  {QUICK_DATASET_YAML}")

    print("\nPretrained model:")
    print(f"  {PRETRAINED_MODEL}")

    print("\nTraining configuration:")
    print(f"  Training images: {TRAIN_IMAGES}")
    print(f"  Validation images: {VAL_IMAGES}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device: {DEVICE}")
    print(f"  Workers: {WORKERS}")

    print("\nOutput:")
    print(
        f"  {RUNS_DIR / EXPERIMENT_NAME}"
    )

    # --------------------------------------------------------
    # Validate pretrained model
    # --------------------------------------------------------

    if not PRETRAINED_MODEL.exists():

        raise FileNotFoundError(
            f"\nPretrained model not found:\n"
            f"{PRETRAINED_MODEL}"
        )

    # --------------------------------------------------------
    # Create quick dataset
    # --------------------------------------------------------

    create_quick_dataset()

    # --------------------------------------------------------
    # Check MPS
    # --------------------------------------------------------

    try:

        import torch

        print("\nPyTorch:")
        print(
            f"  Version: "
            f"{torch.__version__}"
        )

        print(
            f"  MPS available: "
            f"{torch.backends.mps.is_available()}"
        )

        if DEVICE == "mps":

            if not torch.backends.mps.is_available():

                raise RuntimeError(
                    "MPS was requested but "
                    "is not available."
                )

    except ImportError:

        raise RuntimeError(
            "PyTorch is not installed."
        )

    # --------------------------------------------------------
    # Load pretrained model
    # --------------------------------------------------------

    print(
        "\nLoading pretrained YOLOv8..."
    )

    model = YOLO(
        str(PRETRAINED_MODEL)
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nStarting quick training..."
    )

    model.train(

        # Dataset
        data=str(
            QUICK_DATASET_YAML
        ),

        # Training duration
        epochs=EPOCHS,

        # Image resolution
        imgsz=IMAGE_SIZE,

        # Batch size
        batch=BATCH_SIZE,

        # Apple Silicon GPU
        device=DEVICE,

        # Data loading
        workers=WORKERS,

        # Experiment output
        project=str(RUNS_DIR),
        name=EXPERIMENT_NAME,

        # Checkpoints
        save=True,
        save_period=5,

        # Validation
        val=True,

        # Don't cache dataset
        cache=False,

        # Stop if validation stops improving
        patience=5,

        # Reproducibility
        seed=42,

        # Pretrained initialization
        pretrained=True,

        # Verbose output
        verbose=True,
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    best_model = (
        RUNS_DIR
        / EXPERIMENT_NAME
        / "weights"
        / "best.pt"
    )

    last_model = (
        RUNS_DIR
        / EXPERIMENT_NAME
        / "weights"
        / "last.pt"
    )

    print("\n")
    print("=" * 70)
    print("QUICK TRAINING COMPLETE")
    print("=" * 70)

    print("\nBest model:")
    print(f"  {best_model}")

    print("\nLast model:")
    print(f"  {last_model}")

    print("\nQuick dataset:")
    print(f"  {QUICK_DATASET_DIR}")

    print("\nThis was a 300-image pipeline test.")
    print(
        "For final training, use the full "
        "2,700-image training dataset."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
