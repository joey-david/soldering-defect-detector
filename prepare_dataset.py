import os
import shutil
import re
import zipfile

if os.path.exists("DATASET_Sujet2"):
    os.rename("DATASET_Sujet2", "dataset")

zip_files = ['Defaut.zip', 'Sans_Defaut.zip']

# Unzip the files to the dataset path
for zip_file in zip_files:
    zip_path = os.path.join('./dataset', zip_file)  # Create full path to zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./dataset')

# Create a regex pattern to match the file naming format
pattern = re.compile(r'image_(\d+)_(.+)\.png')

# Get the list of files in the dataset directory
files = os.listdir('./dataset/Defaut')

# Loop through the files and move them to the appropriate folder
for file in files:
    if file not in zip_files:  # Skip the zip files themselves
        match = pattern.match(file)
        if match:
            folder_name = os.path.join('./dataset/Defaut', match.group(2))
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            shutil.move(os.path.join('./dataset/Defaut', file), os.path.join(folder_name, file))

print("Files extracted and organized successfully")