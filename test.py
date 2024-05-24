import os

# Define the base directory and the output text file
base_dir = 'daVinci/test'
output_file = 'test.txt'
print("tushir")
# Open the output file in write mode
with open(output_file, 'w') as file:
    # Iterate through the subdirectories (image_0 and image_1)
    for sub_dir in ['image_0', 'image_1']:
        # Construct the full path to the subdirectory
        full_sub_dir = os.path.join(base_dir, sub_dir)
        
        # Check if the subdirectory exists
        if os.path.exists(full_sub_dir):
            # Walk through the subdirectory
            for root, _, files in os.walk(full_sub_dir):
                for file_name in files:
                    # Construct the full file path
                    file_path = os.path.join(root, file_name)
                    # Write the file path to the output file
                    file.write(file_path + '\n')
        else:
            print(f"Directory {full_sub_dir} does not exist.")

print(f"File paths written to {output_file}.")
# import random

# # Define the path to the existing text file and the output files for training and validation
# input_file_path = 'IID-SfmLearner/splits/hamlyn/train_files.txt'
# train_file_path = 'train_paths.txt'
# val_file_path = 'val_paths.txt'

# # Define the number of images for training and validation
# num_train = 9108
# num_val = 1567

# # Read the existing text file
# with open(input_file_path, 'r') as file:
#     lines = [line.strip() for line in file if line.strip()]

# # Ensure there are enough images
# total_images = len(lines)
# assert total_images >= num_train + num_val, "Not enough images to split."

# # Shuffle the lines to ensure random selection
# random.shuffle(lines)

# # Split the lines into training and validation sets
# train_paths = lines[:num_train]
# val_paths = lines[num_train:num_train + num_val]

# # Write the training paths to the train_paths.txt file
# with open(train_file_path, 'w') as file:
#     for path in train_paths:
#         file.write(path + '\n')

# # Write the validation paths to the val_paths.txt file
# with open(val_file_path, 'w') as file:
#     for path in val_paths:
#         file.write(path + '\n')

# # Output the number of images in each set
# print(f"Total number of images: {total_images}")
# print(f"Number of training images: {num_train}")
# print(f"Number of validation images: {num_val}")
