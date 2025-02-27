
import os
import pandas as pd

def count_labels(directory):
    count_0, count_1 = 0, 0
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Assuming label files are .txt
            file_path = os.path.join(directory, filename)
            
            # Read the file, expecting space-separated values
            try:
                df = pd.read_csv(file_path, sep=r'\s+', header=None)
                if not df.empty:
                    count_0 += (df[0] == 0).sum()
                    count_1 += (df[0] == 1).sum()
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    print(f"Total count of 0: {count_0}")
    print(f"Total count of 1: {count_1}")

if __name__ == "__main__":
    directory = "/home/uveye.local/roy.o/Dataset/dent_part/2025-02-24_14-49-04/labels_Copy/val"  # Change this to your actual directory
    count_labels(directory)


# import os
# import pandas as pd

# def count_labels(directory):
#     count_0, count_1 = 0, 0
    
#     # Iterate over all files in the directory
#     for filename in os.listdir(directory):
#         if filename.endswith(".txt"):  # Assuming label files are .txt
#             file_path = os.path.join(directory, filename)
            
#             # Read the file, expecting space-separated values
#             try:
#                 df = pd.read_csv(file_path, delim_whitespace=True, header=None)
#                 if not df.empty:
#                     count_0 += (df[0] == 0).sum()
#                     count_1 += (df[0] == 1).sum()
#             except Exception as e:
#                 print(f"Error reading {filename}: {e}")
    
#     print(f"Total count of 0: {count_0}")
#     print(f"Total count of 1: {count_1}")

# if __name__ == "__main__":
#     directory = "/home/uveye.local/roy.o/Dataset/dent_part/2025-02-24_14-49-04/labels_Copy/val"  # Change this to your actual directory
#     count_labels(directory)
