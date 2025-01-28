import os
os.environ['CLEARML_CONFIG_FILE'] = "/isilon/Automotive/RnD/roy.o/workspace/clearml/clearml.conf"

#/isilon/Automotive/Data/Algo/clearml_global_cache/storage/s3/production-us-eks-data/0z280SJTBbxpEfSOo1Do/usa-volvo-smythe/0f0a634e-140d-4225-8ff4-0a9ebf6db680/compressed_at_cam_07
from datetime import datetime
from allegroai import DataView, DatasetVersion, Task
from uv_dtlp_clml_util.clml_video_wrapper import local_source_video_aware
import tqdm
import argparse
import cv2


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

            filtered_bboxes.append((new_label, [x, y, w, h]))

    # Write to file in YOLO format
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg_example", default="look_at_me", type=str)
    opts = parser.parse_args()

    # clearml_task = Task.init(project_name='roy_o_example_project', task_name='example_task')
    # clearml_logger = clearml_task.get_logger()

   # print('arg_example:', opts.arg_example)

    output_dataset_name = 'Dataset_BrokenPart'
    output_version_name = 'initial_train_val'
    DatasetVersion.create_new_dataset(dataset_name=output_dataset_name)
    dst_out = DatasetVersion.create_version(dataset_name=output_dataset_name, version_name=output_version_name)

    dv = DataView(name='dv_dataset_atlas_lite_damage')
    example_quary='meta.splits.generic.set:"train" OR meta.splits.generic.set:"val"'
    roi_query='label.keyword:"broken_part" OR label.keyword:"missing_part" OR label.keyword:"missing_lp" OR label.keyword:"manual_fix"'
    dv.add_query(dataset_name='atlas_lite_damages', version_name='split_20-0-0__severe_spring',frame_query= example_quary,roi_query=roi_query)
   # dv.add_query(dataset_name='atlas_lite_damages', version_name='split_20-0-0__severe_spring', frame_query='id: "0f0a634e-140d-4225-8ff4-0a9ebf6db680__at_cam_07__frame_0036"')
    #dv.add_query(dataset_name='atlas_lite_damages', version_name='split_20-0-0__severe_spring', frame_query='id: "fff3a0ad-6e4a-490c-81b8-cc2241ca050b__at_cam_04__frame_0033"')
    num_frames = dv.get_count()[0]
    print('\nnumber of frames: {}\n'.format(num_frames))
    new_frames = []
    #save_folder='/home/roy.o@uveye.local/projects/clearml/Dataset'
    counter=0
    class_mapping={"broken_part":0 ,"missing_part":1 ,"missing_lp":2 ,"manual_fix":3}


    # Base folder
    save_folder = '/home/roy.o@uveye.local/projects/clearml/Dataset'

    # Get current date in YYYY-MM-DD format
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Create new path with date as a subfolder
    save_folder = os.path.join(save_folder, current_date)

    # Ensure the directory exists
    os.makedirs(save_folder, exist_ok=True)

    # Dataset types
    dataset_types = ['train', 'val', 'test']

    # List to store all folder paths
    folders = []
    # Loop through each dataset type and create subfolders
    for dataset_type in dataset_types:
        images_folder = os.path.join(save_folder, 'images', dataset_type)
        label_folder = os.path.join(save_folder, 'labels', dataset_type)
        labels_pixels = os.path.join(save_folder, 'labels_pixels', dataset_type)
        
        # Append to folder list
        folders.extend([images_folder, label_folder, labels_pixels])

    # Ensure all required folders exist
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print(f"Folders verified and created if necessary:")
    for folder in folders:
        print(f"- {folder}")

    new_frames=[]
    for i_frame, frame in enumerate(tqdm.tqdm(dv, total=num_frames)):
        # commented out example for getting local path to frame image:
        img_path = local_source_video_aware(frame)
        #v_path = frame.get_local_source()
        if img_path == None or img_path == '':
            continue
        image = cv2.imread(img_path)
        name_for_save = frame.id;
        dataset_type=frame.metadata["splits"]["generic"]["set"]
        bboxes_list = extract_bboxes(frame)
        image_width, image_height=get_image_size(frame)
        # Define new save path
        save_img_path = os.path.join(save_folder, 'images',dataset_type,name_for_save + '.png')
        save_txt_path = os.path.join(save_folder, 'labels', dataset_type,name_for_save+'.txt')
        save_txt_path_pixels = os.path.join(save_folder, 'labels_pixels', dataset_type,name_for_save+'.txt')
        # Save the image
        cv2.imwrite(save_img_path, image)
        save_labels(bboxes_list, class_mapping, image_width, image_height,save_txt_path)
        save_labels_pixels(bboxes_list, class_mapping,save_txt_path_pixels )
        #save_in_dataset
        counter += 1
        new_frames.append(frame)

        if len(new_frames) >= 100:
            dst_out.add_frames(new_frames)
            new_frames = []

        # if len(new_frames) > 0:
        #     dst_out.add_frames(new_frames)

        # counter+=1
        #
        # if i_frame > 50:
        #     break

        # example log scalar (like loss or IoU)
        # clearml_logger.current_logger().report_scalar(
        #     "Loses", "train_loss", iteration=i_frame, value=i_frame*2
        # )

        # dataset operations examples:

        # frame.metadata['new_field_example'] = {'i_can_be_a_dictionary': 1, 'bla': 2}
        # existing_color_value = frame.metadata['color'] if 'color' in frame.metadata else None
        # frame.metadata['color'] = 'overwriting_info_example'

    #     # replacing annotations example
    #     for ann in frame.annotations:
    #         if 'antenna' in ann.labels:
    #             ann.metadata['annotation_type'] = 'updating_label_metadata'
    #             ann.metadata['new_field_example'] = existing_color_value

    #     # removing annotations & adding a new annotation
    #     if not any(['antenna' in ann.labels for ann in frame.annotations]):
    #         frame.remove_annotations()
    #         frame.annotations = []
    #         frame.add_annotation(labels='silly_rect', box2d_xywh=[frame.width/4, frame.height/4, frame.width/2, frame.height/2], confidence=0.95)

    #     new_frames.append(frame)

    #     if len(new_frames) >= 1000:
    #         dst_out.add_frames(new_frames)
    #         new_frames = []

    # if len(new_frames) > 0:
    #     dst_out.add_frames(new_frames)


if __name__ == '__main__':
    main()



# from allegroai import DataView, DatasetVersion, Dataset, Task
# import tqdm
# from uv_dtlp_clml_util.clml_video_wrapper import local_source_video_aware


# dv = DataView(name='dv_for_10K')
# dv.add_query(dataset_name='atlas_lite__2024_Q4', version_name='week_46', frame_query='meta.cam:"at_front_00" AND meta.scan_id:"ff62ede7-1ec9-42e9-8416-a5398817516c"')
# # dv.add_query(dataset_name='atlas_lite__2024_Q4', version_name='week_46', frame_query='meta.cam:"at_cam_03"')
# # dv.add_query(dataset_name='atlas_lite__2024_Q4', version_name='week_46', frame_query='meta.cam:"at_cam_05"')

# num_frames = dv.get_count()[0]
# print('\nnumber of frames: {}\n'.format(num_frames))
# for frame in tqdm.tqdm(dv, total=num_frames):
#     img_path = local_source_video_aware(frame)
#     print(frame.id)
#
# def write_annotations_to_txt(frame, classes, output_file):
#     # Open the output text file in write mode
#     with open(output_file, 'w') as f:
#         # Loop through all annotations in frame
#         for annotation in frame.annotations:
#             # Extract the class label from annotation
#             class_label = annotation.labels[0]
#
#             # Map the class label to its index using the 'classes' dictionary
#             if class_label in classes:
#                 class_index = classes[class_label]
#             else:
#                 # If class is not found in dictionary, skip this annotation or handle as needed
#                 continue
#
#             # Extract the bounding box in xywh format (x, y, width, height)
#             bounding_box = annotation.bounding_box_xywh
#
#             # Write the class index and bounding box to the text file
#             f.write(f"{class_index} {bounding_box[0]} {bounding_box[1]} {bounding_box[2]} {bounding_box[3]}\n")
#
#     # Example usage
#     frame = ...  # Assuming you have a frame object with annotations
#     classes = {
#         "broken_part": 0,
#         "missing_part": 1,
#         "missing_lp": 2,
#         "manual_fix": 3
#     }
#     output_file = "annotations.txt"
#     write_annotations_to_txt(frame, classes, output_file)