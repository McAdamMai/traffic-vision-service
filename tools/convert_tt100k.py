from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


# ============================================================
# Configuration
# ============================================================

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

JSON_PATH = "../raw_data/tt100k_2021/annotations_all.json"
SOURCE_DIR = "../raw_data/tt100k_2021"
OUTPUT_DIR = "../raw_data/tt100k_yolo"


# ------------------------------------------------------------
# Dataset selection
# ------------------------------------------------------------

# Total number of ORIGINAL TT100K training images to select.
#
# These will later be split into:
#
#   2700 train
#    300 val
#
MAX_IMAGES = 3000

VAL_RATIO = 0.10

RANDOM_SEED = 42


# ------------------------------------------------------------
# YOLO configuration
# ------------------------------------------------------------

# One-class detector.
CLASS_ID = 0
CLASS_NAME = "traffic_sign"


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

CREATE_VISUALIZATION = True
NUM_VISUALIZATION_IMAGES = 30


# ------------------------------------------------------------
# Output behavior
# ------------------------------------------------------------

# IMPORTANT:
#
# True  = delete existing output directory first
# False = preserve existing files
#
# For a new 3000-image dataset, True is recommended.
OVERWRITE = True


# ============================================================
# Utility functions
# ============================================================

def safe_link_or_copy(
    source: Path,
    destination: Path,
) -> None:
    """
    Create a symlink to the source image.

    On macOS/Linux this avoids duplicating the actual image data.

    If symlinking fails, fall back to copying the file.
    """

    if destination.exists() or destination.is_symlink():
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        destination.symlink_to(
            source.resolve()
        )

    except (OSError, NotImplementedError):

        shutil.copy2(
            source,
            destination,
        )


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Clamp a value to a range.
    """

    return max(
        minimum,
        min(value, maximum),
    )


# ============================================================
# Bounding-box conversion
# ============================================================

def convert_bbox_to_yolo(
    bbox: Dict[str, float],
    image_width: int,
    image_height: int,
) -> Tuple[
    float,
    float,
    float,
    float,
] | None:
    """
    Convert TT100K:

        xmin, ymin, xmax, ymax

    into YOLO:

        x_center, y_center, width, height

    Coordinates are normalized to [0, 1].

    Returns None if the bounding box is invalid.
    """

    required_keys = {
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }

    if not required_keys.issubset(bbox):
        return None

    xmin = float(bbox["xmin"])
    ymin = float(bbox["ymin"])
    xmax = float(bbox["xmax"])
    ymax = float(bbox["ymax"])

    # --------------------------------------------------------
    # Clip to image boundaries
    # --------------------------------------------------------

    xmin = clamp(
        xmin,
        0,
        image_width,
    )

    xmax = clamp(
        xmax,
        0,
        image_width,
    )

    ymin = clamp(
        ymin,
        0,
        image_height,
    )

    ymax = clamp(
        ymax,
        0,
        image_height,
    )

    width = xmax - xmin
    height = ymax - ymin

    # --------------------------------------------------------
    # Reject invalid boxes
    # --------------------------------------------------------

    if width <= 0:
        return None

    if height <= 0:
        return None

    # --------------------------------------------------------
    # Center coordinates
    # --------------------------------------------------------

    x_center = (
        xmin + xmax
    ) / 2.0

    y_center = (
        ymin + ymax
    ) / 2.0

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    x_center /= image_width
    y_center /= image_height

    width /= image_width
    height /= image_height

    return (
        x_center,
        y_center,
        width,
        height,
    )


# ============================================================
# YOLO label validation
# ============================================================

def validate_yolo_line(
    line: str,
) -> bool:
    """
    Validate one YOLO annotation line.

    Expected format:

        class x_center y_center width height
    """

    parts = line.strip().split()

    if len(parts) != 5:
        return False

    try:

        class_id = int(parts[0])

        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

    except ValueError:

        return False

    if class_id != CLASS_ID:
        return False

    if not (
        0.0 <= x <= 1.0
    ):
        return False

    if not (
        0.0 <= y <= 1.0
    ):
        return False

    if not (
        0.0 < w <= 1.0
    ):
        return False

    if not (
        0.0 < h <= 1.0
    ):
        return False

    return True


# ============================================================
# Category utilities
# ============================================================

def get_image_categories(
    img_info: dict,
) -> set[str]:
    """
    Return the unique TT100K categories appearing
    in one image.

    Categories are used ONLY for selecting a balanced
    subset.

    They are NOT used as YOLO classes.
    """

    categories = set()

    for obj in img_info.get(
        "objects",
        [],
    ):

        category = obj.get(
            "category"
        )

        if category:
            categories.add(
                str(category)
            )

    return categories


# ============================================================
# Balanced subset selection
# ============================================================

def select_balanced_subset(
    items: List[Tuple[str, dict]],
    max_images: int,
    seed: int = 42,
) -> List[Tuple[str, dict]]:
    """
    Select a balanced subset of images.

    The goal is to avoid a subset dominated by only the
    most common TT100K categories.

    Strategy:

    1. Group images by the TT100K categories they contain.
    2. Calculate category frequencies.
    3. Give rarer categories higher selection priority.
    4. Iteratively select images that cover categories
       that have received fewer samples.
    5. Use deterministic randomness for tie-breaking.

    The resulting YOLO dataset remains one-class:

        0 = traffic_sign
    """

    if max_images >= len(items):

        print(
            "\nRequested subset is larger than "
            "the available training set."
        )

        return list(items)

    rng = random.Random(seed)

    # --------------------------------------------------------
    # Build category -> image mapping
    # --------------------------------------------------------

    category_to_images = defaultdict(list)

    image_categories = {}

    for img_id, img_info in items:

        categories = get_image_categories(
            img_info
        )

        image_categories[img_id] = categories

        for category in categories:

            category_to_images[
                category
            ].append(
                img_id
            )

    # --------------------------------------------------------
    # Category frequencies
    # --------------------------------------------------------

    category_frequency = {
        category: len(image_ids)
        for category, image_ids
        in category_to_images.items()
    }

    print(
        "\nOriginal training category statistics:"
    )

    for category, count in sorted(
        category_frequency.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:30]:

        print(
            f"  {category:<15} "
            f"{count:>6,} images"
        )

    # --------------------------------------------------------
    # Shuffle candidate lists
    # --------------------------------------------------------

    for image_ids in category_to_images.values():

        rng.shuffle(image_ids)

    # --------------------------------------------------------
    # Create lookup
    # --------------------------------------------------------

    item_lookup = {
        img_id: (
            img_id,
            img_info,
        )
        for img_id, img_info in items
    }

    # --------------------------------------------------------
    # Selection state
    # --------------------------------------------------------

    selected_ids = set()

    category_selected = Counter()

    # --------------------------------------------------------
    # Calculate target representation
    # --------------------------------------------------------
    #
    # We want rare categories to have a chance to appear,
    # but we don't want a rare category to consume the
    # entire dataset.
    #
    # The target is proportional to:
    #
    #       sqrt(category frequency)
    #
    # This compresses the difference between very common
    # and very rare categories.
    # --------------------------------------------------------

    raw_weights = {}

    for category, frequency in (
        category_frequency.items()
    ):

        raw_weights[category] = (
            frequency ** 0.5
        )

    total_weight = sum(
        raw_weights.values()
    )

    category_targets = {}

    for category, weight in (
        raw_weights.items()
    ):

        target = int(
            max_images
            * weight
            / total_weight
        )

        category_targets[category] = max(
            1,
            target,
        )

    # --------------------------------------------------------
    # First pass:
    #
    # Guarantee coverage of categories whenever possible.
    # --------------------------------------------------------

    categories_sorted = sorted(
        category_frequency,
        key=lambda category:
            category_frequency[category],
    )

    for category in categories_sorted:

        candidates = (
            category_to_images[category]
        )

        for img_id in candidates:

            if len(selected_ids) >= max_images:
                break

            if img_id in selected_ids:
                continue

            selected_ids.add(img_id)

            categories = (
                image_categories[img_id]
            )

            for covered_category in categories:

                category_selected[
                    covered_category
                ] += 1

            break

    # --------------------------------------------------------
    # Second pass:
    #
    # Fill remaining slots.
    #
    # Prefer images containing categories that are currently
    # furthest below their target.
    # --------------------------------------------------------

    remaining_ids = [
        img_id
        for img_id, _ in items
        if img_id not in selected_ids
    ]

    rng.shuffle(
        remaining_ids
    )

    while (
        len(selected_ids)
        < max_images
        and remaining_ids
    ):

        best_score = None
        best_candidates = []

        # Examine a random candidate pool rather than every
        # remaining image. This keeps selection fast.
        candidate_pool = remaining_ids[
            : min(500, len(remaining_ids))
        ]

        for img_id in candidate_pool:

            categories = (
                image_categories[img_id]
            )

            if not categories:

                score = 0.0

            else:

                deficits = []

                for category in categories:

                    target = (
                        category_targets[
                            category
                        ]
                    )

                    current = (
                        category_selected[
                            category
                        ]
                    )

                    deficit = max(
                        0,
                        target - current,
                    )

                    deficits.append(
                        deficit
                    )

                # Prefer images covering categories
                # with the largest deficits.
                score = max(
                    deficits,
                    default=0,
                )

                # Small bonus for covering multiple
                # underrepresented categories.
                score += (
                    0.25
                    * sum(
                        1
                        for deficit
                        in deficits
                        if deficit > 0
                    )
                )

            if (
                best_score is None
                or score > best_score
            ):

                best_score = score
                best_candidates = [
                    img_id
                ]

            elif score == best_score:

                best_candidates.append(
                    img_id
                )

        if not best_candidates:
            break

        selected_id = rng.choice(
            best_candidates
        )

        selected_ids.add(
            selected_id
        )

        categories = (
            image_categories[
                selected_id
            ]
        )

        for category in categories:

            category_selected[
                category
            ] += 1

        remaining_ids.remove(
            selected_id
        )

    # --------------------------------------------------------
    # Convert IDs back to items
    # --------------------------------------------------------

    selected_items = [
        item_lookup[img_id]
        for img_id in selected_ids
    ]

    # --------------------------------------------------------
    # Shuffle final subset
    # --------------------------------------------------------

    rng.shuffle(
        selected_items
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        "\nBalanced subset selected:"
    )

    print(
        f"  Requested: "
        f"{max_images:,}"
    )

    print(
        f"  Selected:  "
        f"{len(selected_items):,}"
    )

    print(
        "\nCategory representation "
        "in selected subset:"
    )

    for category, count in (
        category_selected.most_common()
    ):

        print(
            f"  {category:<15} "
            f"{count:>6,} images"
        )

    return selected_items


# ============================================================
# Visualization
# ============================================================

def create_visualizations(
    output_dir: Path,
    split: str,
    num_images: int,
    seed: int = 42,
) -> None:
    """
    Draw YOLO bounding boxes onto random images.
    """

    from PIL import ImageDraw

    image_dir = (
        output_dir
        / "images"
        / split
    )

    label_dir = (
        output_dir
        / "labels"
        / split
    )

    visualization_dir = (
        output_dir
        / "visualization"
        / split
    )

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = [
        p
        for p in image_dir.glob("*")
        if p.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }
    ]

    rng = random.Random(seed)

    rng.shuffle(
        image_files
    )

    image_files = image_files[
        :num_images
    ]

    for image_path in image_files:

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as exc:

            print(
                f"[WARN] Cannot open "
                f"{image_path}: {exc}"
            )

            continue

        draw = ImageDraw.Draw(
            image
        )

        image_width, image_height = (
            image.size
        )

        if label_path.exists():

            lines = (
                label_path
                .read_text(
                    encoding="utf-8"
                )
                .splitlines()
            )

            for line in lines:

                parts = line.split()

                if len(parts) != 5:
                    continue

                try:

                    _, xc, yc, bw, bh = (
                        map(
                            float,
                            parts,
                        )
                    )

                except ValueError:

                    continue

                xc *= image_width
                yc *= image_height

                bw *= image_width
                bh *= image_height

                xmin = (
                    xc - bw / 2
                )

                ymin = (
                    yc - bh / 2
                )

                xmax = (
                    xc + bw / 2
                )

                ymax = (
                    yc + bh / 2
                )

                draw.rectangle(
                    [
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                    ],
                    outline="red",
                    width=max(
                        2,
                        image_width // 500,
                    ),
                )

        output_path = (
            visualization_dir
            / image_path.name
        )

        image.save(
            output_path
        )

    print(
        "\nVisualization saved to:"
    )

    print(
        f"  {visualization_dir}"
    )


# ============================================================
# Main converter
# ============================================================

def convert_tt100k_to_yolo(
    json_path: str = JSON_PATH,
    source_dir: str = SOURCE_DIR,
    output_dir: str = OUTPUT_DIR,
    val_ratio: float = VAL_RATIO,
    max_images: int = MAX_IMAGES,
    seed: int = RANDOM_SEED,
    overwrite: bool = OVERWRITE,
) -> None:

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    json_path = Path(
        json_path
    ).resolve()

    source_dir = Path(
        source_dir
    ).resolve()

    output_dir = Path(
        output_dir
    ).resolve()

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not json_path.exists():

        raise FileNotFoundError(
            f"\nAnnotation file not found:\n"
            f"{json_path}"
        )

    if not source_dir.exists():

        raise FileNotFoundError(
            f"\nSource directory not found:\n"
            f"{source_dir}"
        )

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    if not (
        0.0 < val_ratio < 1.0
    ):

        raise ValueError(
            "VAL_RATIO must be between 0 and 1."
        )

    if max_images <= 0:

        raise ValueError(
            "MAX_IMAGES must be greater than 0."
        )

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    if (
        output_dir.exists()
        and overwrite
    ):

        print(
            f"Removing existing output:\n"
            f"  {output_dir}"
        )

        shutil.rmtree(
            output_dir
        )

    for split in [
        "train",
        "val",
        "test",
    ]:

        (
            output_dir
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            output_dir
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 70)
    print(
        "TT100K -> YOLO BALANCED SUBSET CONVERTER"
    )
    print("=" * 70)

    print(
        "\nConfiguration:"
    )

    print(
        f"  Maximum images: "
        f"{max_images:,}"
    )

    print(
        f"  Validation ratio: "
        f"{val_ratio:.0%}"
    )

    print(
        f"  Random seed: "
        f"{seed}"
    )

    print(
        f"  YOLO classes: "
        f"1"
    )

    print(
        f"  Class 0: "
        f"{CLASS_NAME}"
    )

    print(
        "\nAnnotation file:"
    )

    print(
        f"  {json_path}"
    )

    print(
        "\nSource directory:"
    )

    print(
        f"  {source_dir}"
    )

    print(
        "\nOutput directory:"
    )

    print(
        f"  {output_dir}"
    )

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    print(
        "\nLoading annotations..."
    )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if "imgs" not in data:

        raise ValueError(
            "Expected 'imgs' key in "
            "TT100K annotations."
        )

    images = data["imgs"]

    print(
        f"Found "
        f"{len(images):,} images."
    )

    # --------------------------------------------------------
    # Separate original TT100K splits
    # --------------------------------------------------------

    original_train = []
    original_test = []

    for img_id, img_info in (
        images.items()
    ):

        relative_path = img_info[
            "path"
        ]

        relative_path = (
            relative_path
            .replace("\\", "/")
        )

        original_split = (
            relative_path
            .split("/")[0]
        )

        if original_split == "train":

            original_train.append(
                (
                    img_id,
                    img_info,
                )
            )

        elif original_split == "test":

            original_test.append(
                (
                    img_id,
                    img_info,
                )
            )

    print(
        "\nOriginal TT100K split:"
    )

    print(
        f"  Train: "
        f"{len(original_train):,}"
    )

    print(
        f"  Test:  "
        f"{len(original_test):,}"
    )

    # --------------------------------------------------------
    # Select balanced subset
    # --------------------------------------------------------

    selected_items = (
        select_balanced_subset(
            original_train,
            max_images=max_images,
            seed=seed,
        )
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    rng = random.Random(seed)

    rng.shuffle(
        selected_items
    )

    val_count = int(
        len(selected_items)
        * val_ratio
    )

    # Safety:
    # At least one validation image.
    val_count = max(
        1,
        val_count,
    )

    val_items = (
        selected_items[
            :val_count
        ]
    )

    train_items = (
        selected_items[
            val_count:
        ]
    )

    print(
        "\nFinal dataset split:"
    )

    print(
        f"  Train: "
        f"{len(train_items):,}"
    )

    print(
        f"  Val:   "
        f"{len(val_items):,}"
    )

    print(
        f"  Total: "
        f"{len(selected_items):,}"
    )

    print(
        f"  Test:  "
        f"{len(original_test):,}"
    )

    # ========================================================
    # Statistics
    # ========================================================

    statistics = {
        "train": {
            "images": 0,
            "objects": 0,
            "missing_images": 0,
            "invalid_boxes": 0,
            "empty_images": 0,
        },
        "val": {
            "images": 0,
            "objects": 0,
            "missing_images": 0,
            "invalid_boxes": 0,
            "empty_images": 0,
        },
        "test": {
            "images": 0,
            "objects": 0,
            "missing_images": 0,
            "invalid_boxes": 0,
            "empty_images": 0,
        },
    }

    # --------------------------------------------------------
    # Original category counts
    # --------------------------------------------------------

    original_category_counts = Counter()

    for _, img_info in (
        images.items()
    ):

        for obj in img_info.get(
            "objects",
            [],
        ):

            category = obj.get(
                "category",
                "unknown",
            )

            original_category_counts[
                category
            ] += 1

    # --------------------------------------------------------
    # Selected category counts
    # --------------------------------------------------------

    selected_category_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    # ========================================================
    # Process split
    # ========================================================

    def process_split(
        items: List[Tuple[str, dict]],
        split: str,
    ) -> None:

        stats = statistics[
            split
        ]

        print(
            f"\nProcessing {split}..."
        )

        for index, (
            img_id,
            img_info,
        ) in enumerate(items):

            relative_path = (
                img_info["path"]
                .replace(
                    "\\",
                    "/",
                )
            )

            source_image = (
                source_dir
                / relative_path
            )

            # ------------------------------------------------
            # Check image
            # ------------------------------------------------

            if not source_image.exists():

                stats[
                    "missing_images"
                ] += 1

                print(
                    f"[WARN] Missing image:"
                    f"\n  {source_image}"
                )

                continue

            # ------------------------------------------------
            # Read dimensions
            # ------------------------------------------------

            try:

                with Image.open(
                    source_image
                ) as image:

                    image_width, image_height = (
                        image.size
                    )

            except Exception as exc:

                stats[
                    "missing_images"
                ] += 1

                print(
                    f"[WARN] Cannot read "
                    f"{source_image}: "
                    f"{exc}"
                )

                continue

            if (
                image_width <= 0
                or image_height <= 0
            ):

                stats[
                    "missing_images"
                ] += 1

                continue

            # ------------------------------------------------
            # Track original categories
            # ------------------------------------------------

            for obj in img_info.get(
                "objects",
                [],
            ):

                category = obj.get(
                    "category"
                )

                if category:

                    selected_category_counts[
                        split
                    ][category] += 1

            # ------------------------------------------------
            # Destination image
            # ------------------------------------------------

            image_name = (
                source_image.name
            )

            destination_image = (
                output_dir
                / "images"
                / split
                / image_name
            )

            # ------------------------------------------------
            # Collision detection
            # ------------------------------------------------

            if (
                destination_image.exists()
                or destination_image.is_symlink()
            ):

                try:

                    existing_source = (
                        destination_image.resolve()
                    )

                    actual_source = (
                        source_image.resolve()
                    )

                    if (
                        existing_source
                        != actual_source
                    ):

                        raise RuntimeError(
                            "\nFilename collision "
                            "detected:\n"
                            f"  Existing: "
                            f"{existing_source}\n"
                            f"  New: "
                            f"{actual_source}"
                        )

                except OSError:

                    pass

            # ------------------------------------------------
            # Link/copy image
            # ------------------------------------------------

            safe_link_or_copy(
                source_image,
                destination_image,
            )

            # ------------------------------------------------
            # YOLO label
            # ------------------------------------------------

            label_path = (
                output_dir
                / "labels"
                / split
                / f"{source_image.stem}.txt"
            )

            lines = []

            objects = (
                img_info.get(
                    "objects",
                    [],
                )
            )

            for obj in objects:

                bbox = obj.get(
                    "bbox"
                )

                if bbox is None:

                    stats[
                        "invalid_boxes"
                    ] += 1

                    continue

                yolo_bbox = (
                    convert_bbox_to_yolo(
                        bbox,
                        image_width,
                        image_height,
                    )
                )

                if yolo_bbox is None:

                    stats[
                        "invalid_boxes"
                    ] += 1

                    continue

                (
                    x_center,
                    y_center,
                    width,
                    height,
                ) = yolo_bbox

                lines.append(
                    f"{CLASS_ID} "
                    f"{x_center:.8f} "
                    f"{y_center:.8f} "
                    f"{width:.8f} "
                    f"{height:.8f}"
                )

            # ------------------------------------------------
            # Write label file
            # ------------------------------------------------

            label_path.write_text(
                "\n".join(lines),
                encoding="utf-8",
            )

            stats[
                "images"
            ] += 1

            stats[
                "objects"
            ] += len(lines)

            if not lines:

                stats[
                    "empty_images"
                ] += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                index + 1
            ) % 500 == 0:

                print(
                    f"  {index + 1:,} / "
                    f"{len(items):,}"
                )

    # --------------------------------------------------------
    # Process all splits
    # --------------------------------------------------------

    process_split(
        train_items,
        "train",
    )

    process_split(
        val_items,
        "val",
    )

    process_split(
        original_test,
        "test",
    )

    # ========================================================
    # Validate labels
    # ========================================================

    print(
        "\nValidating YOLO labels..."
    )

    validation_errors = []

    for split in [
        "train",
        "val",
        "test",
    ]:

        label_dir = (
            output_dir
            / "labels"
            / split
        )

        for label_file in (
            label_dir.glob("*.txt")
        ):

            lines = (
                label_file
                .read_text(
                    encoding="utf-8"
                )
                .splitlines()
            )

            for line_number, line in enumerate(
                lines,
                start=1,
            ):

                if not validate_yolo_line(
                    line
                ):

                    validation_errors.append(
                        (
                            split,
                            label_file.name,
                            line_number,
                            line,
                        )
                    )

    if validation_errors:

        print(
            f"[ERROR] Found "
            f"{len(validation_errors)} "
            f"invalid labels."
        )

        for error in validation_errors[
            :20
        ]:

            print(
                " ",
                error,
            )

        raise RuntimeError(
            "YOLO label validation failed."
        )

    print(
        "All generated labels are valid."
    )

    # ========================================================
    # Dataset YAML
    # ========================================================

    dataset_yaml = (
        "# TT100K balanced subset\n"
        "# One-class traffic sign detector\n"
        "\n"
        f"path: {output_dir.as_posix()}\n"
        "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )

    yaml_path = (
        output_dir
        / "dataset.yaml"
    )

    yaml_path.write_text(
        dataset_yaml,
        encoding="utf-8",
    )

    # ========================================================
    # Summary JSON
    # ========================================================

    summary = {
        "source": {
            "annotation_file": str(
                json_path
            ),
            "source_directory": str(
                source_dir
            ),
        },

        "output_directory": str(
            output_dir
        ),

        "class": {
            "id": CLASS_ID,
            "name": CLASS_NAME,
        },

        "configuration": {
            "max_images": max_images,
            "validation_ratio": val_ratio,
            "random_seed": seed,
            "selection_method": (
                "balanced_category_subset"
            ),
        },

        "splits": statistics,

        "original_tt100k_categories": {
            "number_of_categories": len(
                original_category_counts
            ),

            "number_of_objects": sum(
                original_category_counts.values()
            ),
        },

        "selected_tt100k_categories": {
            split: dict(
                counts
            )
            for split, counts
            in selected_category_counts.items()
        },
    }

    summary_path = (
        output_dir
        / "conversion_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Visualization
    # ========================================================

    if CREATE_VISUALIZATION:

        print(
            "\nCreating visualization samples..."
        )

        create_visualizations(
            output_dir,
            split="train",
            num_images=NUM_VISUALIZATION_IMAGES,
            seed=seed,
        )

        create_visualizations(
            output_dir,
            split="val",
            num_images=min(
                10,
                len(val_items),
            ),
            seed=seed,
        )

    # ========================================================
    # Final report
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CONVERSION COMPLETE"
    )
    print("=" * 70)

    for split in [
        "train",
        "val",
        "test",
    ]:

        stats = statistics[
            split
        ]

        print(
            f"\n{split.upper()}"
        )

        print(
            f"  Images:        "
            f"{stats['images']:,}"
        )

        print(
            f"  Objects:       "
            f"{stats['objects']:,}"
        )

        print(
            f"  Empty images:  "
            f"{stats['empty_images']:,}"
        )

        print(
            f"  Missing:       "
            f"{stats['missing_images']:,}"
        )

        print(
            f"  Invalid boxes: "
            f"{stats['invalid_boxes']:,}"
        )

    print(
        "\nDataset YAML:"
    )

    print(
        f"  {yaml_path}"
    )

    print(
        "\nSummary:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        "\nVisualization:"
    )

    print(
        f"  {output_dir / 'visualization'}"
    )

    print(
        "\nDataset:"
    )

    print(
        f"  {output_dir}"
    )

    print(
        "\nClass:"
    )

    print(
        f"  0 = {CLASS_NAME}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  Inspect the visualization images "
        "before starting training."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    convert_tt100k_to_yolo()
