# Eye-Tracking Saliency Analysis

End-to-end research pipeline for running PsychoPy/EyeLink experiments, processing gaze recordings, generating human and DeepGaze saliency maps, and comparing visual attention across visible and infrared stimuli.

## Overview

This repository covers four stages:

1. eye-tracking acquisition with PsychoPy and EyeLink;
2. parsing, quality control, cleaning, and reprojection of gaze data;
3. generation of human saliency maps, center biases, and DeepGaze IIE predictions;
4. human–human and DeepGaze–human comparisons with statistical analysis.

## Repository structure

```text
DeepGaze/
├── notebooks/
│   ├── processing.ipynb
│   └── analysis.ipynb
├── eyetracking/
│   ├── main.py
│   └── README.md
├── processing/
│   ├── parser.py
│   ├── extractor.py
│   ├── profiler.py
│   ├── cleaner.py
│   └── README.md
├── saliency/
│   ├── builder.py
│   ├── centerbias.py
│   ├── deepgaze.py
│   └── README.md
├── analysis/
│   ├── comparator.py
│   ├── metrics.py
│   ├── stats.py
│   ├── analyzer.py
│   ├── saver.py
│   └── README.md
├── helpers/
│   ├── indexer.py
│   ├── loader.py
│   ├── models.py
│   └── utils.py
├── data/
│   ├── visible/
│   └── infrared/
└── README.md
```

Each main folder contains a dedicated README with detailed implementation notes and usage instructions.


## Main components

- `notebooks` : Interactive examples used to run the main stages of the pipeline. 

- `eyetracking` : Runs the free-viewing experiment and records gaze data.

- `processing` : Transforms EyeLink ASC recordings into analysis-ready tables.

- `saliency` : Builds human and model-based saliency representations.

- `analysis` : Compares saliency maps and performs statistical analyses.


## Requirements

Python 3.10 or later is recommended.

Main dependencies:

```text
numpy
pandas
scipy
scikit-learn
opencv-python
pillow
tqdm
pingouin
torch
POT
psychopy
```

Install the general Python dependencies with:

```bash
pip install numpy pandas scipy scikit-learn opencv-python pillow tqdm pingouin torch POT psychopy
```

Additional software may be required:

- SR Research EyeLink Developers Kit and PyLink for real acquisition;
- `deepgaze_pytorch` and the associated pretrained model weights.

EyeLink and DeepGaze may require separate installation procedures depending on the operating system.

