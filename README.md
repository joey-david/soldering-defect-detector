# Soldering Anomaly Detection with anomalib

This repository uses [anomalib](https://github.com/openvinotoolkit/anomalib) to detect soldering issues.

## Included Virtual Environment

The `.venv` directory is included to simplify usage. This allows for immediate setup without additional configuration.
Anomalib is already installed in the venv: run 
```python
source .venv/bin/activate
```
before running the detection phase.

## Getting Started

Clone the repository and activate the virtual environment:

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
source .venv/bin/activate
```

## Usage

Run the anomaly detection script:

```bash
python3 detect_anomalies.py
```

## About

This project aims to identify defects in soldering processes by utilizing anomaly detection techniques provided by anomalib.
