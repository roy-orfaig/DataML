import os.path

from allegroai import SingleFrame
import cv2
import logging
from tqdm import tqdm

def my_local_source_video_aware(clml_frame: SingleFrame, frame_format: str = 'png', save_one_img: bool = True) -> str:
    local_prefix = "/isilon/Automotive/Data/Algo/clearml_global_cache/storage/s3"
    local_path = clml_frame.get_local_source()
    if local_path is None:
        s3_url=clml_frame.source
        s3_parts = s3_url.replace("s3://", "").split("/", 1)
        bucket_name = s3_parts[0]
        object_key = s3_parts[1]
        # Construct local path
        local_path = os.path.join(local_prefix, bucket_name, object_key)
        if os.path.splitext(local_path)[-1] in [".mp4", ".mkv", ".webm"] and os.path.isfile(local_path):
            print(f"File is not at S3 but found at local path: {local_path}")
        else:
            local_path=''

    # Only supported types (mp4, mkv, webm)
    if os.path.splitext(local_path)[-1] in [".mp4", ".mkv", ".webm"]:
        decompressed_root = local_path + "__decompressed"
        frame_idx = int(clml_frame.timestamp / 100)  # All our videos are recorded at 10 fps
        decompressed_frame_file_path = os.path.join(decompressed_root, f"frame_{frame_idx:04d}.{frame_format}")

        # Exist if file already exist
        if os.path.isfile(decompressed_frame_file_path):
            return decompressed_frame_file_path

        # Decompression is needed (generate frames from mkv)
        logger = logging.getLogger("clml_video_wrapper")
        logger.info(f"Decompressing (save_one_img: {save_one_img}){local_path} into {decompressed_root}...")
        os.makedirs(decompressed_root, exist_ok=True)
        cap = cv2.VideoCapture(local_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video file {local_path}")
        video_frame_idx = -1
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = tqdm(total=frames_count, desc=f"Decoding frames", unit="frames")
        if frame_idx >= frames_count:
            raise Exception(f"Frame timestamp ({clml_frame.timestamp} msec) is outside the video file")
        while cap.isOpened():
            video_frame_idx += 1
            ret, frame = cap.read()
            if ret:
                # Skip all not relevant frames if save_one_img
                if save_one_img and f"frame_{video_frame_idx:04d}.{frame_format}" != decompressed_frame_file_path.split('/')[-1]:
                    continue

                frame_file_path = os.path.join(decompressed_root, f"frame_{video_frame_idx:04d}.{frame_format}")

                # For quiting after relevant frame
                if save_one_img:
                    if frame_file_path == decompressed_frame_file_path:
                        # Save one image and exit
                        cv2.imwrite(frame_file_path, frame)
                        break

                # For continue after relevant frame to save all images
                else:
                    cv2.imwrite(frame_file_path, frame)

                progress.update()
                del ret
                del frame
            else:
                break
        cap.release()
        logger.info(f"Decoded {video_frame_idx} frames")
        if os.path.isfile(decompressed_frame_file_path):
            return decompressed_frame_file_path
        else:
            raise Exception(f"Frame with timestamp {clml_frame.timestamp} does not exist in video file {decompressed_frame_file_path}")
    else:
        local_path=''
        return local_path

