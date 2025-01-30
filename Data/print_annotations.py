import os
import cv2

# Define class mapping and colors
class_mapping = {"broken_part": 0, "missing_part": 1, "missing_lp": 2, "manual_fix": 3}
CLASS_COLORS = {
    0: (0, 0, 255),   # Red
    1: (255, 0, 0),   # Blue
    2: (0, 255, 0),   # Green
    3: (0, 255, 255)  # Yellow
}

# Reverse class mapping for lookup
class_names = {v: k for k, v in class_mapping.items()}

def process_dataset(base_dir, output_dir):
    for split in ['train', 'val']:
        image_folder = os.path.join(base_dir, 'images', split)
        label_folder = os.path.join(base_dir, 'labels', split)
        output_folder = os.path.join(output_dir, split)
        
        os.makedirs(output_folder, exist_ok=True)

        for image_file in os.listdir(image_folder):
            if not image_file.endswith(('.jpg', '.png', '.jpeg')):  
                continue
            
            # Load image
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # Get image dimensions (width & height)
            img_h, img_w, _ = image.shape

            # Load corresponding label file
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_folder, label_file)
            
            if not os.path.exists(label_path):
                print(f"No label file found for: {image_file}")
                continue

            # Read and parse label file
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Invalid label format in: {label_path}")
                    continue
                
                class_id, x_center, y_center, width, height = map(float, parts)
                class_id = int(class_id)

                if class_id not in CLASS_COLORS:
                    print(f"Unknown class {class_id} in {label_path}")
                    continue

                # Convert YOLO format (normalized) to pixel coordinates
                x_center, y_center, width, height = (
                    int(x_center * img_w), int(y_center * img_h),
                    int(width * img_w), int(height * img_h)
                )

                x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
                x2, y2 = int(x_center + width / 2), int(y_center + height / 2)

                # Draw bounding box
                color = CLASS_COLORS[class_id]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Draw label text (Class Name + Bounding Box Size)
                class_name = class_names.get(class_id, f"Class-{class_id}")  # Get class name
                label_text = f"{class_name} ({width}x{height})"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw filled background for better visibility
                cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                cv2.putText(image, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text

            # Save the modified image
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, image)

    print("Processing complete. Check the output directory.")

# Example usage:
data_directory = "/home/roy.o@uveye.local/projects/clearml/Dataset/2025-01-28_11-27-25"  # Change this to your dataset path
output_directory = os.path.join(data_directory, "annotations")  # Output folder where labeled images will be saved
os.makedirs(output_directory, exist_ok=True)
process_dataset(data_directory, output_directory)
print(f"Output: {output_directory}")
