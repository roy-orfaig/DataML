import os
os.environ['CLEARML_CONFIG_FILE'] = "/isilon/Automotive/RnD/roy.o/workspace/clearml/clearml.conf"
from allegroai import DataView, DatasetVersion, Task
from uv_dtlp_clml_util.clml_video_wrapper import local_source_video_aware
import tqdm
import argparse
import cv2
from my_clml_video_wrapper import my_local_source_video_aware
from utils import create_dataset_folders,create_dataset_folders,save_missing_frames_to_csv,save_labels_pixels,save_labels,extract_bboxes,get_image_size
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg_example", default="look_at_me", type=str)
    opts = parser.parse_args()

    # clearml_task = Task.init(project_name='roy_o_example_project', task_name='example_task')
    # clearml_logger = clearml_task.get_logger()

   # print('arg_example:', opts.arg_example)

    # output_dataset_name = 'Dataset_BrokenPart'
    # output_version_name = 'Including_all_images'
    # DatasetVersion.create_new_dataset(dataset_name=output_dataset_name)
    # dst_out = DatasetVersion.create_version(dataset_name=output_dataset_name, version_name=output_version_name)
    # ds_versions = [v.version_name for v in Dataset.get(dataset_name="atlas_lite__2025_Q1").get_versions() if
    #             v.version_name != 'Current']
    dv = DataView(name='dv_dataset_atlas_lite_damage')
    example_quary='meta.splits.generic.set:"train" OR meta.splits.generic.set:"val"'
    if True:
        roi_query='label.keyword:"broken_part" OR label.keyword:"missing_part" OR label.keyword:"missing_lp" OR label.keyword:"manual_fix"'
        dv.add_query(dataset_name='atlas_lite_damages', version_name='split_20-0-0__severe_spring',roi_query=roi_query)
    else:
        roi_query='label.keyword:"scratch" OR label.keyword:"dent""'
        dv.add_query(dataset_name='atlas_lite_damages', version_name='ingest__atlas_lite_all__approved_week2__scratch&dents',frame_query= example_quary,roi_query=roi_query)
    num_frames = dv.get_count()[0]
    print('\nnumber of frames: {}\n'.format(num_frames))
    new_frames = []
    save_root_folder='/home/uveye.local/roy.o/Dataset/Broken_part'
    counter=0
    class_mapping={"broken_part":0 ,"missing_part":1 ,"missing_lp":2 ,"manual_fix":3}

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
            image_height, image_width,_=image.shape
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
            print(f"Missing count: {int(missing_count)}")
        # if len(new_frames) >= 100:
        #     dst_out.add_frames(new_frames)
        #     new_frames = []
    
    save_missing_frames_to_csv(list_of_missing_frame,csv_file)
    save_missing_frames_to_csv(production_bucket_frame,csv_file_prod)
    print(f"Final missing_count: {missing_count}")
if __name__ == '__main__':
    main()

