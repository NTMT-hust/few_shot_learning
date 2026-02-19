from pathlib import Path

import numpy as np


def load_dataset_from_folder(dataset_path):
    """Load dataset from folder structure with class subfolders"""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    class_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    if len(class_folders) == 0:
        raise ValueError(f"No class folders found in {dataset_path}")

    class_names = [folder.name for folder in class_folders]
    num_classes = len(class_names)

    print(f"\nFound {num_classes} classes: {class_names}")

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    image_paths = []
    labels = []

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    for class_folder in class_folders:
        class_name = class_folder.name
        class_idx = class_to_idx[class_name]

        class_images = [
            str(img_path) for img_path in class_folder.iterdir()
            if img_path.suffix.lower() in image_extensions
        ]

        print(f"  Class '{class_name}' (label {class_idx}): {len(class_images)} images")

        if len(class_images) == 0:
            print(f"    ⚠️  Warning: No images found in {class_folder}")

        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    print(f"\nTotal images loaded: {len(image_paths)}")
    print(f"Class distribution: {np.bincount(labels)}")

    return image_paths, labels, class_names, num_classes
