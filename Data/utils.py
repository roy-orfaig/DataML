import os
from datetime import datetime
import csv
def save_labels(bboxes_list, class_mapping, image_width, image_height, output_file="filtered_bboxes.txt"):
    """
    Filters bounding boxes based on the given class_mapping dictionary.
    Normalizes bounding box coordinates by dividing by the image size.
    Replaces the class names with corresponding indices and writes the result to a file.

    :param bboxes_list: List of tuples (label, bbox_xywh)
    :param class_mapping: Dictionary mapping class names to indices
    :param image_width: Width of the image
    :param image_height: Height of the image
    :param output_file: Name of the output file to save results
    """
    filtered_bboxes = []
    
    for label, bbox in bboxes_list:
        if label in class_mapping:
            new_label = class_mapping[label]  # Replace label with its index
            # Normalize bounding box coordinates (x, y, width, height)
            x, y, w, h = bbox
            x += w / 2
            y += h / 2
            x /= image_width
            y /= image_height
            w /= image_width
            h /= image_height


# Check for values exceeding 1 and print them
            assert 0 <= x <= 1, f"Invalid x value: {x} (should be in [0,1])"
            assert 0 <= y <= 1, f"Invalid y value: {y} (should be in [0,1])"
            assert 0 <= w <= 1, f"Invalid width value: {w} (should be in [0,1])"
            assert 0 <= h <= 1, f"Invalid height value: {h} (should be in [0,1])"

            if x > 1 or y > 1:
                print(f"Check values: x={x}, y={y}, w={w}, h={h}")

            filtered_bboxes.append((new_label, [x, y, w, h]))

    #Write to file in YOLO format
    with open(output_file, "w") as f:
        for label, bbox in filtered_bboxes:
            f.write(f"{label} {bbox[0]:.5f} {bbox[1]:.5f} {bbox[2]:.5f} {bbox[3]:.5f}\n")

    print(f"Filtered bounding boxes saved to {output_file}")

def save_labels_pixels(bboxes_list, class_mapping, output_file="filtered_bboxes.txt"):
    """
    Filters bounding boxes based on the given class_mapping dictionary.
    Normalizes bounding box coordinates by dividing by the image size.
    Replaces the class names with corresponding indices and writes the result to a file.

    :param bboxes_list: List of tuples (label, bbox_xywh)
    :param class_mapping: Dictionary mapping class names to indices
    :param image_width: Width of the image
    :param image_height: Height of the image
    :param output_file: Name of the output file to save results
    """
    filtered_bboxes = []

    for label, bbox in bboxes_list:
        if label in class_mapping:
            new_label = class_mapping[label]  # Replace label with its index
            x, y, w, h = bbox
            x += w / 2
            y += h / 2

            filtered_bboxes.append((new_label, [x, y, w, h]))

    # Write to file in YOLO format
    with open(output_file, "w") as f:
        for label, bbox in filtered_bboxes:
            f.write(f"{label} {bbox[0]:.5f} {bbox[1]:.5f} {bbox[2]:.5f} {bbox[3]:.5f}\n")

    print(f"Filtered bounding boxes saved to {output_file}")

def extract_bboxes(single_frame):
    """
    Extracts a list of bounding boxes and corresponding labels from the SingleFrame object.
    """
    bboxes = []
    for annotation in single_frame.annotations:
        label = annotation.labels[0] if annotation.labels else "unknown"
        bbox_xywh = annotation.bounding_box_xywh  # [x, y, w, h]
        bboxes.append((label, bbox_xywh))
    return bboxes


def get_image_size(single_frame):
    """
    Extracts the width and height of the image from the SingleFrame object.

    :param single_frame: The SingleFrame object containing metadata.
    :return: Tuple (width, height)
    """
    width = getattr(single_frame, "width", None)  # Extract width
    height = getattr(single_frame, "height", None)  # Extract height
    return width, height

def create_dataset_folders(base_path: str = '/home/roy.o@uveye.local/projects/clearml/Dataset'):
    """
    Creates a dataset folder with the current timestamp and subfolders for 'train', 'val', and 'test'.

    Args:
        base_path (str): The base directory where datasets should be stored.

    Returns:
        str: The path to the created dataset folder.
    """
    # Get current date and time in YYYY-MM-DD_HH-MM-SS format
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Create the main dataset folder with timestamp
    dataset_folder = os.path.join(base_path, current_datetime)
    os.makedirs(dataset_folder, exist_ok=True)

    # Define dataset types
    dataset_types = ['train', 'val', 'test']

    # Create subfolders
    folders = []
    for dataset_type in dataset_types:
        images_folder = os.path.join(dataset_folder, 'images', dataset_type)
        label_folder = os.path.join(dataset_folder, 'labels', dataset_type)
        labels_pixels_folder = os.path.join(dataset_folder, 'labels_pixels', dataset_type)

        folders.extend([images_folder, label_folder, labels_pixels_folder])

    # Ensure all folders exist
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # Print created folders
    print(f"Dataset folders created at: {dataset_folder}")
    for folder in folders:
        print(f"- {folder}")

    return dataset_folder  # Return the path to the dataset folder

# Function to write missing frames to a CSV file
def save_missing_frames_to_csv(list_of_missing_frames,filename="missing_frames.csv"):
    with open(filename, mode='w', newline='') as file:
        writer =csv.writer(file)
        # Write header
        writer.writerow(["missing frame ID", "bucket","context_id","preview_uri"])
        # Write data
        writer.writerows(list_of_missing_frames)