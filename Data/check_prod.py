import os
os.environ['CLEARML_CONFIG_FILE'] = "/isilon/Automotive/RnD/roy.o/workspace/clearml/clearml.conf"
from allegroai import DataView, DatasetVersion, Task
from uv_dtlp_clml_util.clml_video_wrapper import local_source_video_aware
import tqdm
import argparse
import cv2
from my_clml_video_wrapper import my_local_source_video_aware
from utils import create_dataset_folders,create_dataset_folders,save_missing_frames_to_csv,save_labels_pixels,save_labels,extract_bboxes,get_image_size


dv = DataView()
dv.add_query(dataset_name='atlas_lite_damages', version_name='split_20-0-0__severe_spring',roi_query='label.keyword:"broken_part" OR label.keyword:"missing_part" OR label.keyword:"missing_lp" OR label.keyword:"manual_fix"')
clml_frames = dv.to_list()
img=clml_frames[0].get_local_source()
