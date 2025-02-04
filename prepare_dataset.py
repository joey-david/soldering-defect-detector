import os
import shutil
import re
import zipfile

def extractAndOrder(binary=False):
    # Create dataset directory
    dataset_dir = "dataset" if binary else "dataset_multi"
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    # Extract zip files
    for zip_file in ["Defaut.zip", "Sans_Defaut.zip"]:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

    # Handle defect type organization for multiclass
    if not binary:
        pattern = re.compile(r'image_\d+_(\w+).png')
        defauts = ["SL", "ST_Inf", "ST_Sup", "STP", "ST_Sup_Pli"]
        defaut_dir = os.path.join(dataset_dir, "Defaut")

        # Create defect type subdirectories
        for defaut in defauts:
            os.makedirs(os.path.join(dataset_dir, "Defaut", defaut), exist_ok=True)

        # Move files to their respective defect type directories
        for file in os.listdir(defaut_dir):
            if file.endswith(".png"):
                match = pattern.match(file)
                if match and match.group(1) in defauts:
                    src = os.path.join(defaut_dir, file)
                    dest = os.path.join(dataset_dir, "Defaut", match.group(1), file)
                    shutil.move(src, dest)

if __name__ == "__main__":
    extractAndOrder(False)  # Multiclass
    extractAndOrder(True)   # Binary
    print("Dataset prepared successfully.")