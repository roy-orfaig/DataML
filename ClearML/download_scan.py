import os
import cv2
import boto3


def download_directory_from_s3(bucket_name, s3_folder, local_dir):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        target_dir = os.path.dirname(target)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        bucket.download_file(obj.key, target)
        

def unpack_mkv(local_path, output_path):
    frame_format = "png"
    print("Unpacking ", local_path)
    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        raise Exception(f"Failed to open video file {local_path}")
    video_frame_idx = -1

    while cap.isOpened():
        video_frame_idx += 1
        # Skip all not relevant frames if save_one_img
        ret, frame = cap.read()
        if ret:
            frame_file_path = os.path.join(output_path, f"frame_{video_frame_idx:04d}.{frame_format}")
            cv2.imwrite(frame_file_path, frame)
        else:
            break

def unpack_whole_scan(scan_path, out_path):
    compressed_dirs = [dirname for dirname in os.listdir(scan_path) if dirname.startswith('compressed_')]
    for compressed_dir in compressed_dirs:
        camera_name = compressed_dir.split('compressed_')[1]
        local_path = os.path.join(scan_path, compressed_dir, 'frames.mkv')
        output_path = os.path.join(out_path, camera_name)
        os.makedirs(output_path, exist_ok=True)
        unpack_mkv(local_path, output_path)

def get_scan(main_dir_path, bucket_name, s3_folder):
    assert not s3_folder.endswith('/')
    print(' ~~ get_scan', bucket_name, s3_folder)
    # download
    print(' ~ download')
    scan_id = s3_folder.split('/')[-1]
    local_dir = os.path.join(main_dir_path, 'download', scan_id)
    download_directory_from_s3(bucket_name, s3_folder, local_dir)

    # unpack
    print(' ~ unpack')
    scan_path = local_dir
    out_path = os.path.join(main_dir_path, 'frames/Neverland/20241013', '2024-04-29T11-42-24.000Z_' + scan_id)
    unpack_whole_scan(scan_path, out_path)

    print(' ~ get_scan done')


def uvcamp_path_to_s3_folder(uvcamp_path):
    uvcamp_splits = uvcamp_path.split("/")
    s3_folder = "/".join((uvcamp_splits[3], uvcamp_splits[5], uvcamp_splits[6]))
    return s3_folder


def get_scan_from_uvcamp_path(main_dir_path, bucket_name, uvcamp_path):
    s3_folder = uvcamp_path_to_s3_folder(uvcamp_path)
    get_scan(main_dir_path, bucket_name, s3_folder)


def main():
    bucket_name = "production-us-eks-data"
   # s3_folder = "8z39DfzVRhzwWuf1Zyaw/O9GTKitGkWXFh1I1kmkC/2f67b7df-5860-4f88-aa77-51004fbe1793"
    #main_dir_path = "/isilon/Automotive/RnD/Amit/Projects/data/cluster_miss_2025_01_19"
    
    # bucket_name = "production-us-eks-data"
    s3_folder = "AhdNZGfMY8HkuOrS9FNA/SDtTSRpbuc3n1NqZDOMx/3b263006-ec95-4ae9-ba9b-c3cfec63f432"
    main_dir_path = "/isilon/Automotive/RnD/roy.o/workspace/data/bug_volvo_dec2024"
 #   main_dir_path="/home/roy.o@uveye.local/projects/data/delete_me"
    get_scan(main_dir_path, bucket_name, s3_folder)


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    main()