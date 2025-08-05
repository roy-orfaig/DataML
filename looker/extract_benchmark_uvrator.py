import copy
from tqdm import tqdm
from allegroai import DataView, DatasetVersion, IterationOrder

dataset_name = ''
versions = [
    ### versions here
]
dataset_out = ''
# frame_query = f"(meta.cam:(at_cam_04 OR at_cam_05 OR at_cam_06 OR at_windshield_00))"
frame_query = f""
# roi_query_str = "(BodyDent OR BodyScratch OR BodySevereDamage) AND rois.meta.internal_review_conclusion.keyword:Approved"
roi_query_str = "rois.meta.internal_review_conclusion.keyword:Rejected"



output_frames = []
infered_frames= []
for version in tqdm(versions):
    print(f'working on version \n{version} \n')
    dv_tagging_pool = DataView(iteration_order=IterationOrder.random)
    dv_tagging_pool.add_query(
        dataset_name=dataset_name,
        version_name=version,
    )
    infered_frames.extend(dv_tagging_pool.to_list())
#                     

print(f'iterating over all infered frames, total frames: {len(infered_frames)}')
for frame in tqdm(infered_frames):
    out_anns = []
    for ann in frame.annotations:
        irc = ann.metadata['internal_review_conclusion']
        if irc == 'Rejected':
            out_anns.append(irc)
    
    new_frame = copy.deepcopy(frame)
    new_frame.annotations = out_anns
    output_frames.append(new_frame)
            


print(f'creating new dataset version')
new_version = DatasetVersion.create_version(
    dataset_name=dataset_out,
    version_name='new_version'
)
new_version.add_frames(output_frames)