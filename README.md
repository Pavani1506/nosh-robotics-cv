# Egg Segmentation and Video Tracking — Nosh Robotics CV Assignment

## 1. Overview

This project explores egg segmentation using a pretrained Segment Anything Model (SAM), classical image-processing methods, SAM-guided GrabCut refinement, evaluation against SAM pseudo-reference masks, and video-based object tracking.

## 2. Tasks Implemented

1. Generated SAM segmentation proposals for 110 dataset images.
2. Explored classical segmentation methods including Otsu thresholding, HSV thresholding, and GrabCut.
3. Refined candidate masks using SAM-guided GrabCut and evaluated predictions against selected SAM pseudo-reference masks.
4. Implemented an OpenCV-based object tracking baseline on the assigned video.

## 3. Project Structure

```text
nosh-robotics-cv/
├── src/
│   ├── run_sam_all.py
│   ├── classical_baseline.py
│   ├── grabcut_baseline.py
│   ├── run_classical_experiments.py
│   ├── multi_egg_grabcut.py
│   ├── run_multi_egg_grabcut_batch_v2.py
│   ├── create_data_split.py
│   ├── automated_egg_evaluation.py
│   ├── run_holdout_grabcut.py
│   ├── calculate_holdout_metrics.py
│   ├── create_evaluation_plots.py
│   └── track_eggs_video_v2.py
├── reports/
│   ├── experiment_log.csv
│   ├── dev_eval_split.csv
│   ├── nosh_robotics_cv_submission_report.pdf
│   └── figures/
├── models/
├── data/
├── outputs/
├── requirements.txt
└── README.md
```

Generated masks, videos, dataset files, and model weights are not included in the repository.

## 4. Environment and Installation

Developed using Python 3.11 on macOS.

### Clone the repository

```bash
git clone https://github.com/Pavani1506/nosh-robotics-cv.git
cd nosh-robotics-cv
```

### Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the original Segment Anything package:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## 5. Dataset and Model Setup

### Dataset

Download the egg dataset from the assignment-provided Roboflow source:

https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1

Extract the dataset into:

```text
data/egg.v1i.folder/
```

The scripts expect the dataset folders, including `train/egg`, `valid/egg`, and `test/egg`.

### SAM checkpoint

Download the pretrained SAM ViT-B checkpoint:

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Place the checkpoint at:

```text
models/sam_vit_b_01ec64.pth
```

The dataset and checkpoint must be downloaded separately.

## 6. Running the Experiments

Run commands from the repository root after activating the virtual environment and setting up the dataset and model.

### 6.1 Inspect the dataset

```bash
python src/inspect_dataset.py
```

### 6.2 Run SAM segmentation

```bash
python src/run_sam_all.py
```

SAM masks, metadata, and overlays are saved under:

```text
outputs/sam_all/
```

### 6.3 Run classical segmentation experiments

```bash
python src/classical_baseline.py
python src/grabcut_baseline.py
python src/run_classical_experiments.py
```

These experiments explore Otsu thresholding, HSV thresholding, morphological operations, and GrabCut.

The batch experiment processes the first 10 images in the training/egg folder.

Results are saved under:

```text
outputs/classical/
outputs/classical_batch/
```

### 6.4 Run SAM-guided GrabCut

```bash
python src/multi_egg_grabcut.py
python src/run_multi_egg_grabcut_batch_v2.py
```

The scripts refine SAM candidate masks using GrabCut and save masks, overlays, and batch results under:

```text
outputs/multi_egg_grabcut/
outputs/multi_egg_grabcut_batch_v2/
```

### 6.5 Create the development/evaluation split

```bash
python src/create_data_split.py
```

The split manifest is saved at:

```text
reports/dev_eval_split.csv
```

The current split contains 88 development images and 22 evaluation images.

**Note:** This split is provisional. It was created after earlier experimentation and should not be considered a fully independent benchmark.

### 6.6 Generate selected SAM pseudo-reference masks

```bash
python src/automated_egg_evaluation.py
```

The script selects candidate masks and saves evaluation outputs under:

```text
outputs/automated_evaluation/
```

### 6.7 Run holdout predictions

```bash
python src/run_holdout_grabcut.py
```

The script processes the evaluation images listed in the split manifest.

Predicted masks and overlays are saved under:

```text
outputs/holdout_grabcut/
```

### 6.8 Calculate evaluation metrics

```bash
python src/calculate_holdout_metrics.py
```

The IoU and Dice comparison CSV is saved at:

```text
outputs/automated_evaluation/holdout_metrics.csv
```

### 6.9 Generate evaluation plots

```bash
python src/create_evaluation_plots.py
```

Evaluation plots and failure-case visualizations are generated under:

```text
outputs/automated_evaluation/report_assets/
```

### 6.10 Run video tracking

Place the assigned video at:

```text
data/video/egg_video.mp4
```

Run:

```bash
python src/track_eggs_video_v2.py
```

The script generates:

```text
outputs/video_tracking/v2/egg_tracking_v2.mp4
outputs/video_tracking/v2/tracks_v2.csv
```

The tracking baseline uses HSV-based detection and centroid-based association between frames.

## 7. Results and Evaluation

- SAM successfully processed 110 images.
- A provisional development/evaluation split of 88:22 was created.
- Holdout predictions were generated for 22 evaluation images.
- Fourteen images had matching selected SAM pseudo-reference masks.
- Mean IoU: 0.9573.
- Mean Dice: 0.9767.

These metrics measure agreement with selected SAM pseudo-reference masks, not verified ground-truth accuracy.

Evaluation plots and failure-case visualizations are available in:

```text
reports/figures/
```

## 8. Video Tracking Output

The generated tracking video is available through Google Drive:

https://drive.google.com/file/d/19ZDbHlVpOppEq5EDHB9HvjZmqRPmeBTe/view?usp=sharing

## 9. Experiment Log and Report

Experiment parameters, observations, attempted methods, and limitations are documented in:

```text
reports/experiment_log.csv
```

The development/evaluation split is documented in:

```text
reports/dev_eval_split.csv
```

The project report is available at:

```text
reports/nosh_robotics_cv_submission_report.pdf
```

## 10. Limitations

- SAM proposals are automatically generated and are not manually verified ground-truth masks.
- Eight evaluation images did not have matching selected SAM pseudo-reference masks in the current comparison.
- The development/evaluation split was created after earlier experimentation and is provisional.
- Classical segmentation may miss eggs or include background objects.
- Video tracking can produce false detections, including fingernails and reflections.
- The reported IoU and Dice values should be interpreted as agreement with pseudo-reference masks rather than true segmentation accuracy.

## 11. AI Assistance

AI tools were used to assist with implementation and debugging. The implementation, parameters, results, and limitations should be reviewed and understood before use.