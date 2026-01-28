# Probing Visual Concepts in Vision-Language Models for Autonomous Driving

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of our work on probing visual concepts in traffic scenes using Vision-Language Models (VLMs). We investigate how different layers of VLMs encode traffic-relevant visual concepts through linear probing.

## Overview

We analyze the internal representations of state-of-the-art small VLMs to understand how they encode various traffic scene concepts, including:

- **Presence**: Whether something is present in the scene
- **Count**: How many instances of an object are present in the scene
- **Spatial Understanding**: Where something is located
in the scene in relation to something else
- **Orientation Detection**: The orientation of an object in the scene

## Supported Models

- **[Ovis2.5-2B](https://huggingface.co/AIDC-AI/Ovis2.5-2B)**
- **[InternVL3.5-2B](https://huggingface.co/OpenGVLab/InternVL3_5-2B)**
- **[VST-3B](https://huggingface.co/rayruiyang/VST-3B-RL)** (SFT and RL variants)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/probing_visual_concepts_in_traffic_scenes.git
cd probing_visual_concepts_in_traffic_scenes

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision
pip install transformers
pip install pillow
pip install matplotlib
pip install scikit-learn
pip install tqdm
```

### Additional Requirements

For InternVL3.5 and VST models, Flash Attention 2 is required:
```bash
pip install flash-attn --no-build-isolation
```

## Dataset

Download our traffic scenes dataset from: **[Dataset Link Placeholder](https://placeholder-dataset-link.com)**

The dataset consists of synthetic traffic scenes with annotations for various visual concepts. Each annotation file is a JSON containing:
- `image_path`: Path to the image file
- `label`: Ground truth label for the visual concept
- `distance`: Distance to the object (for distance-stratified analysis)
- `town`: CARLA town identifier (used for train/val/test splits)

## Usage

### 1. Feature Extraction

Extract features from different layers of the VLMs:

```bash
cd feature_extraction

# Extract Ovis2.5 features
python extract_features.py \
    --model ovis2.5 \
    --annotations_path /path/to/annotations.json \
    --category 1 \
    --llm_features visual_embs_and_last_token \
    --save_path ./extracted_features
```

**Arguments:**
- `--model`: Model to use (`ovis2.5`, `internvl3.5`, `vst`)
- `--annotations_path`: Path to the JSON annotations file
- `--category`: Unique id for data category
- `--llm_features`: Which features to extract (`visual_embs`, `last_token`, `all`, `visual_embs_and_last_token`)
- `--save_path`: Directory to save extracted features
- `--use_cls`: (InternVL3.5) Use CLS token
- `--num_tiles`: (InternVL3.5) Number of image tiles
- `--version`: (VST) Model version (`rl` or `sft`)

**Category IDs:**
| ID | Concept |
|----|---------|
| 1  | Pedestrian presence |
| 2  | Pedestrian orientation |
| 3  | Pedestrian count |
| 4  | Blinker |
| 5  | Pedestrian side |
| 6  | Traffic barrel presence |
| 7  | Traffic barrel count |
| 8  | Bicycle orientation |

### 2. Probe Training

Train linear or MLP probes on the extracted features:

```bash
cd probe_training

python train.py \
    --annotations_path /path/to/annotations.json \
    --parent_features_directory /path/to/extracted_features \
    --num_out 1 \
    --epochs 10 \
    --num_repeats 10 \
    --save_path ./results
```

**Arguments:**
- `--annotations_path`: Path to the JSON annotations file
- `--parent_features_directory`: Directory containing extracted features
- `--num_out`: Number of output classes (1 for binary classification with BCE loss)
- `--distance`: Filter annotations by distance (optional)
- `--epochs`: Number of training epochs per learning rate
- `--num_repeats`: Number of experiment repetitions for statistical significance
- `--use_mlp`: Use MLP probe instead of linear probe
- `--random_baseline`: Randomize labels to establish baseline performance
- `--save_path`: Directory to save results

### 3. Model Evaluation (Direct VQA)

Evaluate the VLMs directly on visual question answering:

```bash
cd model_eval

python eval.py \
    --model ovis2.5 \
    --annotations_path /path/to/annotations.json \
    --category 1 \
    --answers Yes No \
    --save_path ./results
```

**Arguments:**
- `--model`: Model to evaluate (`ovis2.5`, `internvl3.5`, `vst-rl`, `vst-sft`)
- `--annotations_path`: Path to the JSON annotations file
- `--category`: Visual concept category
- `--answers`: List of possible answer tokens
- `--question_only`: Compare only answer token probabilities (optional)
- `--save_path`: Directory to save results

## Results

Pre-trained probes are available in the `trained_probes/` directory.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{author2025probing,
  title={Probing Visual Concepts in Traffic Scenes},
  author={Author Names},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
