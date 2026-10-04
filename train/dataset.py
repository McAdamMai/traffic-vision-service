import json
import cv2
import os
from pathlib import Path
from collections import defaultdict


def extract_and_trim_tt100k(json_path: str, image_dir: str, output_dir: str, top_n_classes: int = 45,
                            max_per_class: int = 400):
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Step 1: Count class frequencies
    class_counts = defaultdict(int)
    for img_id, img_data in data['imgs'].items():
        for obj in img_data.get('objects', []):
            class_counts[obj['category']] += 1

    # Keep the top N most frequent classes to satisfy the thesis requirement
    valid_classes = set(sorted(class_counts, key=class_counts.get, reverse=True)[:top_n_classes])
    saved_counts = defaultdict(int)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 2: Iterate, crop, and save
    for img_id, img_data in data['imgs'].items():
        # The 'path' in the JSON usually looks like "train/1234.jpg"
        img_path = os.path.join(image_dir, img_data['path'])

        if not os.path.exists(img_path) or 'objects' not in img_data:
            continue

        img = None

        for idx, obj in enumerate(img_data['objects']):
            cat = obj['category']

            if cat not in valid_classes or saved_counts[cat] >= max_per_class:
                continue

            if img is None:
                img = cv2.imread(img_path)

            bbox = obj['bbox']
            xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
            xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])

            # Crop the bounding box
            crop = img[max(0, ymin):min(img.shape[0], ymax), max(0, xmin):min(img.shape[1], xmax)]

            if crop.size > 0:
                class_folder = Path(output_dir) / cat
                class_folder.mkdir(exist_ok=True)

                save_path = class_folder / f"{img_id}_{idx}.jpg"
                cv2.imwrite(str(save_path), crop)
                saved_counts[cat] += 1

    print(f"Extraction complete. Total classes processed: {len(valid_classes)}")
    print(f"Total isolated sign images generated: {sum(saved_counts.values())}")


if __name__ == "__main__":
    # The ../ tells the script to look outside the traffic-vision-service folder
    extract_and_trim_tt100k(
        json_path="../raw_data/tt100k_2021/annotations_all.json",
        image_dir="../raw_data/tt100k_2021/",
        output_dir="data/train/"
    )