import os
import shutil
import re
import zipfile
import random

def extractAndOrder(binary=False):
    # Create dataset directory if it doesn't exist
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # List of zip files to extract
    zip_files = ["Defaut.zip", "Sans_Defaut.zip"]

    # Extract each zip file into the dataset directory
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
    
    # Split the contents of the Sans_Defaut subdirectory into training and testing sets
    sans_defaut_dir = os.path.join(dataset_dir, "Sans_Defaut")
    os.makedirs(os.path.join(dataset_dir, "train", "Sans_Defaut"))
    os.makedirs(os.path.join(dataset_dir, "val", "Sans_Defaut"))

    for file in os.listdir(sans_defaut_dir):
        if random.random() < 0.2:
            shutil.move(os.path.join(sans_defaut_dir, file), os.path.join(dataset_dir, "val", "Sans_Defaut"))
        else:
            shutil.move(os.path.join(sans_defaut_dir, file), os.path.join(dataset_dir, "train", "Sans_Defaut"))

    # Extract default names from the Defaut subdirectory and move to corresponding directories
    defaut_dir = os.path.join(dataset_dir, "Defaut")
    os.makedirs(os.path.join(dataset_dir, "train", "Defaut"))
    os.makedirs(os.path.join(dataset_dir, "val", "Defaut"))

    if binary:
        for file in os.listdir(defaut_dir):
            if random.random() < 0.2:
                shutil.move(os.path.join(defaut_dir, file), os.path.join(dataset_dir, "val", "Defaut"))
            else:
                shutil.move(os.path.join(defaut_dir, file), os.path.join(dataset_dir, "train", "Defaut"))
        

    else:
        pattern = re.compile(r'image_\d+_(\w+).png')
        defauts = ["SL", "ST_Inf", "ST_Sup", "STP", "ST_Sup_Pli"]
        for defaut in defauts:
            os.makedirs(os.path.join(dataset_dir, "train", "Defaut", defaut))
            os.makedirs(os.path.join(dataset_dir, "val", "Defaut", defaut))

        for file in os.listdir(defaut_dir):
            match = pattern.match(file)
            if match:
                defaut = match.group(1)
                if random.random() < 0.2:
                    shutil.move(os.path.join(defaut_dir, file), os.path.join(dataset_dir, "val", "Defaut", defaut))
                else:
                    shutil.move(os.path.join(defaut_dir, file), os.path.join(dataset_dir, "train", "Defaut", defaut))



if __name__ == "__main__":
    if os.path.exists("dataset"):
        shutil.rmtree("dataset")
    extractAndOrder(False)
    shutil.rmtree("dataset/Defaut")
    shutil.rmtree("dataset/Sans_Defaut")
    print("Dataset prepared and extracted.")