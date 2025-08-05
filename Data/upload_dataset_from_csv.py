from allegroai import DataView, DatasetVersion
from uv_dtlp_clml_util.clml_video_wrapper import local_source_video_aware
import random
import csv

file_path = "/isilon/Automotive/RnD/roy.o/workspace/data/datasets/broken_part/reannotations/2025_03_03_20_56_26/iou_results_2025_03_03_20_56_26_LOW.csv"  # Replace with your actual CSV file path
#file_path = "/isilon/Automotive/RnD/roy.o/workspace/data/datasets/broken_part/reannotations/2025_03_07_09_36_55/iou_results_2025_03_07_09_36_55_LOW.csv"
scan_ids = []

with open(file_path, mode='r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')  # Use '\t' if it's tab-separated
    for row in reader:
        if row:  # Ensure the row is not empty
            first_column = row[0].split(',')[0]
            file_name = first_column.replace('.png', '')  # Remove .png extension
            scan_ids.append(file_name)

#print(scan_ids)

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
item_id = []
counter = 0

for f in frames:
    if f.id in scan_ids:
        valid.append(f)
        try:
            item_id.append([f.metadata['dataloop_tasks'][0]['item_id']])
        except Exception as e:  # Use 'except' instead of 'catch'
            counter += 1  # Corrected the counter increment syntax
            print(f"Error at index {counter}: {e}")  # Print error messag

import pandas as pd

# Convert to DataFrame
df = pd.DataFrame(item_id, columns=['item_id'])

# Save to CSV file
csv_filename = "item_ids_broken_part_train.csv"
df.to_csv(csv_filename, index=False)

print(f"CSV file saved as {csv_filename}")

# new_version = DatasetVersion.create_version(
#     dataset_name='Dataset_BrokenPart',
#     parent_version_names='Annotations',
#     version_name='split_20-0-0__severe_spring_for_reannotations'
# )

# new_version = DatasetVersion.create_version(
#     dataset_name='Dataset_BrokenPart',
#     version_name='split_20-0-0__severe_spring_for_reannotations'
# )
# new_version.add_frames(valid)