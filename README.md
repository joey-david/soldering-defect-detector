# Soldering Anomaly Detection with anomalib

This repository uses [anomalib](https://github.com/openvinotoolkit/anomalib) to detect soldering issues.

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

Run the dataset extraction and classification from the root of the project :

```bash
python3 prepare_dataset.py
```

## About

This project aims to identify defects in soldering processes by utilizing anomaly detection techniques provided by anomalib.
