# prepare_dataset.py

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Paths
SOURCE_TRAIN_GOOD = Path("data/bottle/train/good")
SOURCE_TEST_DIR = Path("data/bottle/test")
DEST_DIR = Path("bottle_split_dataset")

# Define all classes
all_classes = ['good', 'broken_large', 'broken_small', 'contamination']

# Create destination folders
for split in ['train', 'val']:
    for cls in all_classes:
        (DEST_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def collect_images():
    data = []

    # Add 'good' images from train/good
    for img in SOURCE_TRAIN_GOOD.glob("*.png"):
        data.append((img, 'good'))

    # Add test images from each defect category
    for defect_cls in SOURCE_TEST_DIR.iterdir():
        if defect_cls.name in all_classes:
            for img in defect_cls.glob("*.png"):
                data.append((img, defect_cls.name))

    return data

def split_and_copy(data, train_ratio=0.8):
    class_to_images = {cls: [] for cls in all_classes}

    # Group images by class
    for img_path, label in data:
        class_to_images[label].append(img_path)

    # Split and copy
    for label, img_paths in class_to_images.items():
        random.shuffle(img_paths)
        split_idx = int(len(img_paths) * train_ratio)
        train_imgs = img_paths[:split_idx]
        val_imgs = img_paths[split_idx:]

        # Copy files
        for img in train_imgs:
            shutil.copy(img, DEST_DIR / "train" / label / img.name)
        for img in val_imgs:
            shutil.copy(img, DEST_DIR / "val" / label / img.name)

        print(f"✅ {label}: {len(train_imgs)} train, {len(val_imgs)} val")

if __name__ == "__main__":
    data = collect_images()
    split_and_copy(data)
    print("Dataset preparation complete!")
