import os
os.environ['CLEARML_CONFIG_FILE'] = "/isilon/Automotive/RnD/roy.o/workspace/clearml/clearml.conf"
from allegroai import DataView, DatasetVersion, Task
from uv_dtlp_clml_util.clml_video_wrapper import local_source_video_aware
import tqdm
import argparse
import cv2
from my_clml_video_wrapper import my_local_source_video_aware
from utils import create_dataset_folders,save_labels,save_labels_pixels,extract_bboxes,get_image_size


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
    num_frames = dv.get_count()[0]
    print('\nnumber of frames: {}\n'.format(num_frames))
    new_frames = []
    save_folder='/home/roy.o@uveye.local/projects/clearml/Dataset'
    counter=0
    class_mapping={"broken_part":0 ,"missing_part":1 ,"missing_lp":2 ,"manual_fix":3}

    create_dataset_folders(save_folder)

    new_frames=[]
    for i_frame, frame in enumerate(tqdm.tqdm(dv, total=num_frames)):
        # commented out example for getting local path to frame image:
        img_path = my_local_source_video_aware(frame)
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

if __name__ == '__main__':
    main()

