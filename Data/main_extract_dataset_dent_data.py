import os
os.environ['CLEARML_CONFIG_FILE'] = "/isilon/Automotive/RnD/roy.o/workspace/clearml/clearml.conf"
from allegroai import DataView, DatasetVersion, Task
from uv_dtlp_clml_util.clml_video_wrapper import local_source_video_aware
import tqdm
import argparse
import cv2
from my_clml_video_wrapper import my_local_source_video_aware
from utils import create_dataset_folders,create_dataset_folders,save_missing_frames_to_csv,save_labels_pixels,save_labels,extract_bboxes,get_image_size


def convert_names(bboxes):
    """
    Converts the labels in the bounding boxes to the desired format.
    "dent__bubble", "dent__glasses", "dent__cluster" -> "dent_is"
    "dent__out_of_stripes" -> "dent_os"
    """
    return [(label if label not in ["dent__bubble", "dent__glasses", "dent__cluster", "dent__out_of_stripes"]
             else "dent_is" if label in ["dent__bubble", "dent__glasses", "dent__cluster"]
             else "dent_os", bbox) for label, bbox in bboxes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg_example", default="look_at_me", type=str)
    opts = parser.parse_args()

    # clearml_task = Task.init(project_name='roy_o_example_project', task_name='example_task')
    # clearml_logger = clearml_task.get_logger()

   # print('arg_example:', opts.arg_example)

    # output_dataset_name = 'Dataset_Dent'
    # output_version_name = 'classes_dent_is_oos'
    # DatasetVersion.create_new_dataset(dataset_name=output_dataset_name)
    # dst_out = DatasetVersion.create_version(dataset_name=output_dataset_name, version_name=output_version_name)
    dv = DataView(name='dv_dataset_atlas_lite_damage')

    roi_query_is_bubble='label.keyword:"dent__bubble"'
    roi_query_is_glasses='label.keyword:"dent__glasses"'
    roi_query_is_cluster='label.keyword:"dent__cluster"'
    roi_query_oos='"dent__out_of_stripes"'
    #example_query='NOT meta.cam:at_front* AND NOT meta.cam:at_rear'
    example_query_is='meta.splits.generic.set:"val" ANT NOT meta.cam:at_front* AND NOT meta.cam:at_rear*'
    example_query_oos='meta.splits.generic.set:"val"'
    dv.add_query(dataset_name='atlas_lite_damages', version_name='split_16-0-0__scratch&dents__with_attributes',frame_query=example_query_is,roi_query=roi_query_is_bubble)
    dv.add_query(dataset_name='atlas_lite_damages', version_name='split_18-0-0__scratch&dents__with_attributes',frame_query=example_query_is,roi_query=roi_query_is_glasses)
    dv.add_query(dataset_name='atlas_lite_damages', version_name='split_18-0-0__scratch&dents__with_attributes',frame_query=example_query_is,roi_query=roi_query_is_cluster)
    dv.add_query(dataset_name='atlas_lite_damages', version_name='split_18-0-0__scratch&dents__with_attributes',frame_query=example_query_oos,roi_query=roi_query_oos) 
    num_frames = dv.get_count()[0]
    print('\nnumber of frames: {}\n'.format(num_frames))
    new_frames = []
    save_root_folder='/home/uveye.local/roy.o/Dataset/dent_part'
    os.makedirs(save_root_folder, exist_ok=True)
    print(f"Directory '{save_root_folder}' is ready.")
    counter=0
    class_mapping={"dent_is":0 ,"dent_oos":1}

    save_folder=create_dataset_folders(save_root_folder)
    csv_file=save_folder+"/missing_file.csv"
    csv_file_prod=save_folder+"/production_bucket.csv"
    list_of_missing_frame=[]
    production_bucket_frame=[]
    new_frames=[]
    missing_count=0
    production_bucket_count=0
    for i_frame, frame in enumerate(tqdm.tqdm(dv, total=num_frames)):
            # commented out example for getting local path to frame image:
            img_path = my_local_source_video_aware(frame)
            frame_id=frame.id
            bucket_name = frame.context_id.split('/')[2]
            context_id=frame.context_id
            preview_uri=frame.preview_uri
            if bucket_name=="production-us-eks-data":
                 production_bucket_count+=1
                 if production_bucket_count<100:
                    production_bucket_frame.append([frame_id, bucket_name,context_id,preview_uri])
                 
            #v_path = frame.get_local_source()
            if img_path == None or img_path == '':
                missing_count+=1
                if missing_count<100:
                    list_of_missing_frame.append([frame_id, bucket_name,context_id,preview_uri])
                    print(f"missing_count: {missing_count}")
                continue
            image = cv2.imread(img_path)
            name_for_save = frame.id
            dataset_type=frame.metadata["splits"]["generic"]["set"]
            bboxes_list = extract_bboxes(frame)
            bboxes_list = convert_names(bboxes_list) 
            image_height,image_width,_=image.shape
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
            # new_frames.append(frame)
            # print(f"Missing count: {int(missing_count)}")
            # if len(new_frames) >= 100:
            #     dst_out.add_frames(new_frames)
            #     new_frames = []
    
    save_missing_frames_to_csv(list_of_missing_frame,csv_file)
    save_missing_frames_to_csv(production_bucket_frame,csv_file_prod)
    print(f"Final missing_count: {missing_count}")
if __name__ == '__main__':
    main()

