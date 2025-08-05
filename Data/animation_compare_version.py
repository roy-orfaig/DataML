import os
import cv2
#import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image

# Parameters
start_frame = 1
end_frame = 50
# input_pattern_1 = "/path/to/first_sequence/frame_{:04d}.png"
# input_pattern_2 = "/path/to/second_sequence/frame_{:04d}.png"
#input_pattern_1 = "/isilon/Automotive/RnD/roy.o/clustering/IM/1b84c86e-4698-42d2-8974-59700df741d2/at_front_01/20250505_145912/frame_{:04d}.png"
#input_pattern_2 = "/home/roy.o@uveye.local/projects/20250505_145912_tracker_output_mode_flow/frame_{:04d}.jpg"
#input_pattern_1 = "/isilon/Automotive/RnD/roy.o/clustering/IM/1b84c86e-4698-42d2-8974-59700df741d2/at_cam_04/20250506_145943/frame_{:04d}.png"
#input_pattern_2 = "/isilon/Automotive/RnD/roy.o/clustering/IM/1b84c86e-4698-42d2-8974-59700df741d2/at_cam_04/20250506_145943_tracker_output_mode_flow/frame_{:04d}.jpg"
# input_pattern_1 = "/isilon/Automotive/RnD/roy.o/clustering/results_scan/38e48af2-5bd1-45ad-a3c0-1219e277d9d7/compressed_at_cam_04/20250512_142352_all_flow/frame_{:06d}.jpg"
# input_pattern_2= "/isilon/Automotive/RnD/roy.o/clustering/results_scan/38e48af2-5bd1-45ad-a3c0-1219e277d9d7/compressed_at_cam_04/20250513_130341_all_flow/frame_{:06d}.jpg"

input_pattern_2 = "/isilon/Automotive/RnD/roy.o/clustering/results_scan/1b84c86e-4698-42d2-8974-59700df741d2/20250513_144518_all_flow/frame_{:06d}.jpg"
input_pattern_1= "/isilon/Automotive/RnD/roy.o/clustering/results_scan/1b84c86e-4698-42d2-8974-59700df741d2/20250513_144819_all_flow/frame_{:06d}.jpg"

input_pattern_2 = "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/tracker_area/at_cam_01_{:03d}.png"
input_pattern_1= "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/tracker_bug/at_cam_01_{:03d}.png"

input_pattern_1= "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/067806cf-35f6-447b-a30e-bc3367d20116/at_cam_01_no_reid_broken/at_cam_01_{:03d}.png"
input_pattern_2 = "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/067806cf-35f6-447b-a30e-bc3367d20116/at_cam_01_reid_broken/at_cam_01_{:03d}.png"
mp4_output = "/home/roy.o@uveye.local/projects/animations/067806cf-35f6-447b-a30e-bc3367d20116_cam01_broken.mp4"

input_pattern_1= "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/0587d4c6-7cd8-4d02-ba46-5dfe5f3417ec/at_rear_00_no_reid_broken/at_rear_00_{:03d}.png"
input_pattern_2 = "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/0587d4c6-7cd8-4d02-ba46-5dfe5f3417ec/at_rear_00_reid_broken/at_rear_00_{:03d}.png"
mp4_output = "/home/roy.o@uveye.local/projects/animations/0587d4c6-7cd8-4d02-ba46-5dfe5f3417ec_rear_00_scratch.mp4"

input_pattern_1= "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/check_tracker_067806cf-35f6-447b-a30e-bc3367d20116_broken_reid_off1/at_cam_01_{:03d}.png"
input_pattern_2 = "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/check_tracker_067806cf-35f6-447b-a30e-bc3367d20116_broken_reid_on1/at_cam_01_{:03d}.png"
mp4_output = "/home/roy.o@uveye.local/projects/animations/067806cf-35f6-447b-a30e-bc3367d20116_reid_compare_cam01_update.mp4"

input_pattern_1= "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/check_tracker_a3172da1-2438-498d-b8ae-cddb161df9df_broken_reid_off/at_cam_01_{:03d}.png"
input_pattern_2 = "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/check_tracker_a3172da1-2438-498d-b8ae-cddb161df9df_broken_reid_on/at_cam_01_{:03d}.png"
mp4_output = "/home/roy.o@uveye.local/projects/animations/a3172da1-2438-498d-b8ae-cddb161df9df_reid_compare_cam01.mp4"

input_pattern_1= "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/check_tracker_959b2fa7-7236-4846-948a-d35f568f2553_broken_new_mechanism_off/at_rear_00_{:03d}.png"
input_pattern_2 = "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/check_tracker_959b2fa7-7236-4846-948a-d35f568f2553_broken_new_mechanism_on_fix/at_rear_00_{:03d}.png"
mp4_output = "/home/roy.o@uveye.local/projects/animations/a3172da1-959b2fa7-7236-4846-948a-d35f568f2553_mechanizm_cam01.mp4"

# input_pattern_1= "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/check_tracker_0587d4c6-7cd8-4d02-ba46-5dfe5f3417ec_scratch_reid_off1/at_rear_00_{:03d}.png"
# input_pattern_2 = "/isilon/Automotive/RnD/roy.o/atlas_lite_damages/debug/check_tracker_0587d4c6-7cd8-4d02-ba46-5dfe5f3417ec_scratch_reid_on1/at_rear_00_{:03d}.png"
# mp4_output = "/home/roy.o@uveye.local/projects/animations/0587d4c6-7cd8-4d02-ba46-5dfe5f3417ec_reid_compare_rear_update.mp4"


fps = 1 # frames per second

def create_combined_frame(img1, img2, frame_id,title1='May', title2='filter by aspect ratio'):
    """Create a side-by-side figure with two titles using matplotlib and return it as a BGR image"""
    
    title1 = f'July (frame: {frame_id})'
    title2='August'
    #title2='ReID'
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].set_title(title1)
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].set_title(title2)
    axs[1].axis('off')

    # Save plot to image in memory
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=200)
    plt.close(fig)
    buf.seek(0)

    # Convert to OpenCV BGR image
    pil_img = Image.open(buf)
    img_array = np.array(pil_img)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

# Process frames
frames = []
for frame_id in range(start_frame, end_frame + 1):
    file1 = input_pattern_1.format(frame_id)
    file2 = input_pattern_2.format(frame_id)

    if os.path.exists(file1) and os.path.exists(file2):
        img1 = cv2.imread(file1)
        img2 = cv2.imread(file2)
        combined = create_combined_frame(img1, img2,frame_id, title1=f"May", title2=f"With Reid")
        frames.append(combined)
    else:
        print(f"Warning: Missing one or both frames at index {frame_id}")

# Save GIF
if frames:
    # imageio.mimsave(gif_output, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], duration=1/fps)
    # print(f"GIF saved to {gif_output}")

    # Save MP4
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_output, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"MP4 saved to {mp4_output}")
else:
    print("No frames were saved.")

# import os
# import cv2
# import imageio.v2 as imageio  # For GIF creation only

# # Parameters
# start_frame = 5
# end_frame = 30
# input_pattern_1 = "/home/roy.o@uveye.local/projects/20250505_145912_tracker_output_mode_flow/frame_{:04d}.jpg"
# input_pattern_1 = "/isilon/Automotive/RnD/roy.o/clustering/IM/1b84c86e-4698-42d2-8974-59700df741d2/at_front_01/20250505_145912/frame_{:04d}.png"
# gif_output = "may_version.gif"
# mp4_output = "may_version.mp4"
# fps = 1  # frames per second

# # Load images
# frames = []
# for i in range(start_frame, end_frame + 1):
#     filename = input_pattern.format(i)
#     if os.path.exists(filename):
#         img = cv2.imread(filename)
#         frames.append(img)
#     else:
#         print(f"Warning: {filename} not found.")

# # Save GIF using imageio
# if frames:
#     # Convert BGR (OpenCV) to RGB for GIF
#     rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
#     imageio.mimsave(gif_output, rgb_frames, duration=1 / fps)
#     print(f"GIF saved to {gif_output}")

#     # Save MP4 using OpenCV
#     height, width, _ = frames[0].shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' or 'avc1'
#     video_writer = cv2.VideoWriter(mp4_output, fourcc, fps, (width, height))

#     for frame in frames:
#         video_writer.write(frame)
#     video_writer.release()
#     print(f"MP4 saved to {mp4_output}")
# else:
#     print("No valid frames found. Nothing was saved.")
