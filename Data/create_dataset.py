import os

# Define the folder path containing the text files
folder_path = '/home/uveye.local/roy.o/Dataset/dent_part/2025-02-24_14-49-04/labels/val'

# Get a list of all text files in the folder
text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Iterate over each file
for filename in text_files:
    file_path = os.path.join(folder_path, filename)
    
    # Open the file in read mode
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Open the file in write mode to overwrite with updated content
    with open(file_path, 'w') as file:
        for line in lines:
            # Split the line into columns
            columns = line.split()
            
            # Replace the first column with 0
            columns[0] = '0'
            
            # Write the modified line back to the file
            file.write(' '.join(columns) + '\n')

print("Finished updating all files.")
