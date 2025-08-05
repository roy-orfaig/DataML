import cv2
import os
import glob

# Parameters
image_folder = '/isilon/Automotive/RnD/roy.o/clustering/reid/test'  # Replace with your path
output_video = '/home/roy.o@uveye.local/projects/animations/heat_map_video.mp4'     # Output video name
frame_rate = 2                        # Frames per second

# Get list of images
images = glob.glob(os.path.join(image_folder, '*.png'))
images.sort()  # Ensure the images are in order

# Check if images are found
if not images:
    print("No PNG files found in the specified directory.")
    exit()

# Get the width and height from the first image
frame = cv2.imread(images[0])
height, width, _ = frame.shape

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Write each image to the video
for image_path in images:
    frame = cv2.imread(image_path)
    video.write(frame)

# Release the video writer
video.release()
print(f"Video saved as {output_video}")
