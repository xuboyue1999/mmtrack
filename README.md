# FAMTrack: Learning Frequency and Memory-aware Prompts for Multi-modal Object Tracking (Pattern Recognition 2026)


Official repository for **"Learning Frequency and Memory-aware Prompts for Multi-modal Object Tracking"**.

> **FAMTrack** is a prompt-learning-based multi-modal tracker that injects lightweight frequency-aware and memory-aware prompts into a frozen RGB foundation tracker. It is designed for **RGB-Thermal (RGB-T)**, **RGB-Depth (RGB-D)**, and **RGB-Event (RGB-E)** tracking.

---

FAMTrack is a multimodal visual tracking codebase built on top of the `ODTrack` framework.
The current public version keeps the core training, evaluation, and demo pipeline, and removes most private experimental artifacts and local-only outputs.

## Features

- Unified training and evaluation pipeline under `lib/` and `tracking/`
- Support for multiple multimodal tracking datasets
- RGB-T, RGB-D, and event-based evaluation entry points
- Config-driven experiments under `experiments/odtrack/`

## Project Structure

```text
FAMTrack/
├─ experiments/              # experiment configs
├─ lib/                      # core models, training, testing, datasets
├─ tracking/                 # command-line entry scripts
├─ test_rgbe_mgpus.py        # RGBE / VisEvent evaluation helper
└─ README.md
```

## Requirements

Recommended environment:

- Python
- PyTorch
- torchvision
- OpenCV
- NumPy
- PyYAML
- easydict
- Pillow
- jpeg4py
- lmdb

Optional tools:

- `tensorboardX`
- `thop`
- `visdom`

A typical installation may look like:

```bash
pip install torch torchvision
pip install opencv-python numpy pyyaml easydict pillow jpeg4py lmdb
pip install tensorboardX thop visdom
```
**YOU CAN USE THE requirements.txt**
## Environment Configuration

Before training or evaluation, set dataset and workspace paths in:

- `lib/train/admin/local.py`
- `lib/test/evaluation/local.py`

If you want to generate template local config files automatically, you can use:

```bash
python tracking/create_default_local_file.py --workspace_dir <workspace> --data_dir <data_root> --save_dir <save_root>
```

## Supported Datasets

The released code contains dataset loaders for:
- RGBT234
- LasHeR
- GTOT
- DepthTrack
- RGBD1K
- VOT22-RGBD
- VisEvent

Please configure the corresponding local paths before use.


## Experimental Results

FAMTrack achieves strong results across three representative multi-modal tracking settings.

### RGB-T Tracking
Evaluated on **RGBT234** and **LasHeR**.

| Dataset | PR | SR |
|---|---:|---:|
| RGBT234 | **0.919** | **0.689** |
| LasHeR | **0.726** | 0.571 |

### RGB-D Tracking
Evaluated on **DepthTrack** and **VOT22-RGBD**.

| Dataset | Metrics |
|---|---|
| DepthTrack | **Pre 0.636 / Re 0.663 / F 0.649** |
| VOT22-RGBD | **EAO 0.773 / A 0.821 / R 0.933** |

### RGB-E Tracking
Evaluated on **VisEvent**.

| Dataset | PR | SR |
|---|---:|---:|
| VisEvent | **0.803** | **0.626** |

### Efficiency

- **Parameters:** 98.9M
- **Speed:** 65 FPS
- **Extra parameters over baseline:** 7.3M
- **Extra computation:** 1 GFLOPs


### Training strategy
Training is performed in **two stages**:
**!!!!YOU MUST LOAD THE PRETRAINED MODEL (WHICH CAN BE DOWNLOWDED IN MODEL ZOO) AND THEN START TRAINING!!!!** 
**THE PRETRAINED MODEL LOAD PATH CAN BE SET IN lib/models/odtrack/odtrack.py line306**
#### Stage 1
Fine-tune:
- visual adapter
- patch embedding layer

Freeze:
- all other network components

**Please Comment Memory Adapter When Training Stage 1 (in lib/models/odtrack/vit_ce_ada.py line 402,459,460 and lib/models/layers/attn_blocks_ada.py line 108,137)**

```bash
python tracking/train.py --script odtrack --config baseline_rgbt --save_dir ./output --mode single
```
#### Stage 2
Fine-tune on top of Stage 1:
- memory adapter
- patch embedding layer
- visual adapter

Freeze:
- all remaining parts

**Please CANCEL Comment in STAGE 1**
**PLEASE CHANGE PRETRAINED MODEL LOAD PATH TO STAGE 1 CHECKPOINTS FILE(output/checkpoints/train/odtrack/<config>/ODTrack_epXXXX.pth.tar)**
Main training entry:

```bash
python tracking/train.py --script odtrack --config baseline_rgbt --save_dir ./output --mode single
```

Notes:

- Experiment configs are located in `experiments/odtrack/`
- The training wrapper eventually calls `lib/train/run_training.py`
- Checkpoints are saved under `output/checkpoints/`

## Model Zoo and Raw Results

The checkpoints can be downloaded in

https://pan.baidu.com/s/1N4nJGyCOxogh2zNAsfEhrA 

code: v6cc 

The raw results can be downloaded in 

https://pan.baidu.com/s/1r7Uj_2IyJAInv6Mcmr844A 

code:4i22 

## Evaluation

Run dataset evaluation:

```bash
python tracking/test.py odtrack baseline_rgbt --dataset_name lasher --threads 0 --num_gpus 1
```

Run a specific sequence:

```bash
python tracking/test.py odtrack baseline_rgbt --dataset_name lasher --sequence 0 --threads 0 --num_gpus 1
```

For RGBE / VisEvent style evaluation:

```bash
python test_rgbe_mgpus.py --script_name odtrack --yaml_name baseline_rgbe --dataset_name VisEvent --mode sequential
```

Evaluation checkpoints are resolved from:

```text
output/checkpoints/train/odtrack/<config>/ODTrack_epXXXX.pth.tar
```

## Video Demo

Run tracking on a local video:

```bash
python tracking/video_demo.py odtrack baseline_rgbt <path_to_video> --debug 0
```

## Configs

Available public configs in this release:

- `baseline_rgbt`
- `baseline_rgbd`
- `baseline_rgbe`

They are stored in:

```text
experiments/odtrack/
```

## Notes

- This repository is prepared for public release, but still assumes users provide their own dataset paths and pretrained checkpoints.
- Some training scripts still reflect the original research workflow. If you adapt the project to a new environment, start by checking `tracking/train.py`, `lib/train/run_training.py`, and the local path files.
- Generated outputs such as checkpoints, tensorboard logs, and tracking results are ignored by `.gitignore`.

## Quick Start Checklist

1. Install dependencies.
2. Fill in dataset paths in `lib/train/admin/local.py` and `lib/test/evaluation/local.py`.
3. Place pretrained weights and trained checkpoints in the expected output directory.
4. Choose a config from `experiments/odtrack/`.
5. Run training, testing, or video demo.


## You can cite our work:

@article{xu2026learning,
  title={Learning Frequency and Memory-Aware Prompts for Multi-Modal Object Tracking},
  author={Xu, Boyue and Hou, Ruichao and Ren, Tongwei and Zhou, Dongming and Wu, Gangshan and Cao, Jinde},
  journal={Pattern Recognition},
  pages={113532},
  year={2026},
  publisher={Elsevier}
}
