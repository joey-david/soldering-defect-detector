# Soldering Anomaly Detection with anomalib


## About

This project aims to identify defects in soldering processes using deep learning. This repository uses [anomalib](https://github.com/openvinotoolkit/anomalib) to detect soldering issues. The directions for this project can be found under [the `docs` subdirectory](docs/Sujet_M1_2024_2.pdf). A complete report of the conception and development of this project can be found [here](docs/rapport.pdf).

## Getting Started

Clone the repository and activate the virtual environment:

```bash
git clone https://forge.univ-lyon1.fr/p2115771/ouverture_recherche_2024.git
cd ouverture_recherche_2024
python3 -m venv .venv
source .venv/bin/activate
```
# Installing requirements

run ```pip install -r requirements.txt``` prior to running any module. It contains a specific list of version requirements, as anomalib is incompatible with a lot of recent libraries.


## Extracting dataset

If needing to train the model yourself, run the dataset extraction and classification from the root of the project :

```bash
python3 src/prepare_dataset.py
```

# Training the model

Train your own version of the model (after optionally modifying its hyperparameters) with

```bash
python3 src/train.py
```

If you don't have a powerful enough graphics card for it, no worries ! 
You can download our version of the trained patchore model [here](https://drive.google.com/file/d/1JoaYigmb-G5rkovvWdrSJfxYui8qH4gC/view).
Make sure to unzip it and to place it under the models folder under the name `model_bin.pth`.