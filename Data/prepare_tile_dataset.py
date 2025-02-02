import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
# DATASET_PATH = "dataset"  # Original dataset
# OUTPUT_PATH = "dataset_tiled"  # Output tiled dataset
DATASET_PATH = "/home/roy.o@uveye.local/projects/clearml/Dataset/2025-01-29_12-26-44"  # Original dataset
NEW_DATASET_PATH = "/home/roy.o@uveye.local/projects/Data/tile_1024"  # Output tiled dataset
IMG_SIZE = 1024  # Tile size
OVERLAP = 200  # Overlap pixels
SAVE_ONLY_LABELED_TILES = True  # Set to True to save only images with labels

def pad_tile(tile):
    """Ensures that every tile is exactly 1024x1024 (pads if needed)."""
    h, w, c = tile.shape
    pad_bottom = max(IMG_SIZE - h, 0)
    pad_right = max(IMG_SIZE - w, 0)

    if pad_bottom > 0 or pad_right > 0:
        tile = cv2.copyMakeBorder(tile, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return tile

def clean_labels(labels):
    """Sanitizes label data to prevent errors."""
    valid_labels = []
    for line in labels:
        values = line.strip().split()
        if len(values) != 5:
            print(f"⚠️ Warning: Incorrect label format (skipped): {line}")
            continue
        try:
            cls, x_center, y_center, width, height = map(float, values)
            if 0 <= cls and 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                valid_labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            else:
                print(f"⚠️ Warning: Out-of-range label values (skipped): {line}")
        except ValueError:
            print(f"⚠️ Warning: Non-numeric label values (skipped): {line}")
    return valid_labels

def split_image(image, img_name, label_path, save_img_dir, save_label_dir, w, h):
    """Splits an image into overlapping 1024x1024 tiles and updates labels."""
    stride = IMG_SIZE - OVERLAP

    # Read YOLO labels (if available)
    label_file = os.path.join(label_path, img_name.replace(".png", ".txt"))
    labels = []
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            labels = clean_labels(f.readlines())

    for y in range(0, h - OVERLAP, stride):
        for x in range(0, w - OVERLAP, stride):
            tile = image[y:y + IMG_SIZE, x:x + IMG_SIZE]

            # Ensure tile is exactly 1024x1024
            tile = pad_tile(tile)

            tile_name = f"{img_name.replace('.png', '')}_{x}_{y}.png"

            # Adjust bounding boxes
            new_labels = []
            for label in labels:
                cls, x_center, y_center, width, height = map(float, label.split())

                # Convert to absolute coordinates
                x_abs = x_center * w
                y_abs = y_center * h
                w_abs = width * w
                h_abs = height * h

                # Check if bbox is within the tile
                if (x_abs + w_abs / 2 > x and x_abs - w_abs / 2 < x + IMG_SIZE and
                        y_abs + h_abs / 2 > y and y_abs - h_abs / 2 < y + IMG_SIZE):

                    # Convert bbox to tile coordinate system
                    new_x_center = (x_abs - x) / IMG_SIZE
                    new_y_center = (y_abs - y) / IMG_SIZE
                    new_width = w_abs / IMG_SIZE
                    new_height = h_abs / IMG_SIZE

                    # Ensure bbox is inside tile
                    if 0 <= new_x_center <= 1 and 0 <= new_y_center <= 1:
                        new_labels.append(f"{cls} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}")

            # Save tile **only if it contains labels**
            if SAVE_ONLY_LABELED_TILES and len(new_labels) == 0:
                print(f"❌ Skipping tile {tile_name} (no labels).")
                continue  # Skip this tile

            # Save the tiled image
            cv2.imwrite(os.path.join(save_img_dir, tile_name), tile, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            print(f"✅ Saved tile {tile_name} - Size: {tile.shape[1]}x{tile.shape[0]}")

            # Save updated labels
            label_file_new = os.path.join(save_label_dir, tile_name.replace(".png", ".txt"))
            with open(label_file_new, "w") as f:
                f.write("\n".join(new_labels))

def process_dataset(dataset_path, new_dataset_path):
    """Processes the dataset, ensuring proper tiling and label adjustments."""
    for split in ["train", "val"]:
        img_dir = os.path.join(dataset_path, "images", split)
        label_dir = os.path.join(dataset_path, "labels", split)

        save_img_dir = os.path.join(new_dataset_path, "images", split)
        save_label_dir = os.path.join(new_dataset_path, "labels", split)
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_label_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(img_dir), desc=f"Processing {split} set"):
            if img_name.endswith(".png"):
                img_path = os.path.join(img_dir, img_name)

                img = cv2.imread(img_path)

                if img is None:
                    print(f"⚠️ Skipping corrupted image: {img_name}")
                    continue

                h, w, _ = img.shape
                print(f"📏 Original image: {img_name} - Size: {w}x{h}")

                split_image(img, img_name, label_dir, save_img_dir, save_label_dir, w, h)

# Run the tiling process
process_dataset(DATASET_PATH, NEW_DATASET_PATH)
print("✅ Dataset tiling complete!")