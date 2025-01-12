import os
import shutil
import re

# Create a regex pattern to match the file naming format
pattern = re.compile(r'image_(\d+)_(.+)\.png')

# Get the list of files in the current directory
files = os.listdir('.')

for file in files:
    match = pattern.match(file)
    if match:
        folder_name = match.group(2)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        shutil.move(file, os.path.join(folder_name, file))
        print("blah blah blah")