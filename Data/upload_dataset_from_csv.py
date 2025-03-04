from allegroai import DataView, DatasetVersion
from uv_dtlp_clml_util.clml_video_wrapper import local_source_video_aware
import random
import csv

file_path = "/isilon/Automotive/RnD/roy.o/workspace/data/datasets/borken_part/reannotations/2025_03_03_20_56_26/iou_results_2025_03_03_20_56_26_LOW.csv"  # Replace with your actual CSV file path
scan_ids = []

with open(file_path, mode='r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')  # Use '\t' if it's tab-separated
    for row in reader:
        if row:  # Ensure the row is not empty
            file_name = row[0].replace('.png', '')  # Remove .png extension
            scan_ids.append(file_name)

print(scan_ids)

#scan_ids = ['00065488-720d-4c9c-aa60-9c129b98430d__at_front_00__frame_0006']
scan_ids = list(set(scan_ids))


# def create_subset():
dataset_name = 'atlas_lite_damages'
dv = DataView()
dv.add_query(
    dataset_name=dataset_name,
    version_name='split_20-0-0__severe_spring',
)
frames = dv.to_list()

valid = []
# for f in frames:
#     if f.metadata['scan_id'] in scan_ids:
#         valid.append(f)

for f in frames:
    if f.id in scan_ids:
        valid.append(f)

new_version = DatasetVersion.create_version(
    dataset_name='atlas_lite_damages',
    parent_version_names=['Annotations'],
    version_name='split_20-0-0__severe_spring_for_reannotations'
)
new_version.add_frames(valid)