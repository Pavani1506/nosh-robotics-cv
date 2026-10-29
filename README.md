# Egg Segmentation and Video Tracking — Nosh Robotics CV Assignment

## Overview
This project explores egg segmentation using a pretrained Segment Anything Model (SAM), classical image-processing methods, SAM-guided GrabCut refinement, and video tracking.

## Tasks
1. Generate SAM segmentation proposals for 110 dataset images.
2. Compare Otsu thresholding, HSV thresholding, and GrabCut.
3. Refine masks and evaluate predictions against selected SAM pseudo-reference masks.
4. Track egg-like objects in the assigned video using OpenCV.

## Project Structure
- `src/` — Python scripts for segmentation, experiments, evaluation, and tracking.
- `reports/experiment_log.csv` — experiment parameters and observations.
- `reports/dev_eval_split.csv` — provisional development/evaluation split.
- `requirements.txt` — Python dependencies.

Generated masks, videos, dataset files, and model weights are not included.

## Setup
Developed with Python 3.11 on macOS.

Create and activate a virtual environment:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
