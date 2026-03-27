# FAMTrack: Learning Frequency and Memory-aware Prompts for Multi-modal Object Tracking


Official repository for **"Learning Frequency and Memory-aware Prompts for Multi-modal Object Tracking"**.

> **FAMTrack** is a prompt-learning-based multi-modal tracker that injects lightweight frequency-aware and memory-aware prompts into a frozen RGB foundation tracker. It is designed for **RGB-Thermal (RGB-T)**, **RGB-Depth (RGB-D)**, and **RGB-Event (RGB-E)** tracking.

---


## Overview

Recent prompt-learning-based multi-modal trackers are efficient, but they still face two major limitations:

1. **Insufficient frequency-aware fusion.** Existing methods mainly fuse modalities in the spatial or channel domain, while ignoring frequency-domain discrepancies across modalities. In practice, RGB data usually contains stronger high-frequency texture details, thermal/depth data tends to provide low-frequency structure cues, and event streams often contain sparse high-frequency motion edges.
2. **Weak long-range temporal modeling.** Most trackers rely on adjacent-frame propagation or short-term template updates, which can drift under occlusion, motion blur, illumination changes, and large appearance variation.

To address these issues, **FAMTrack** introduces a **dual-adapter framework** on top of a frozen RGB tracker:

- a **frequency-guided visual adapter** for cross-modal interaction in **frequency, spatial, and channel** dimensions;
- a **multi-level memory adapter** for robust temporal cue storage, update, and retrieval.

This design improves tracking robustness while preserving the efficiency advantages of prompt learning.


---

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

---

## Training

### Training datasets
FAMTrack is trained separately for different multi-modal tracking tasks:

- **RGB-T:** LasHeR training set
- **RGB-D:** DepthTrack training set
- **RGB-E:** VisEvent training set

### Training strategy
Training is performed in **two stages**:

#### Stage 1
Fine-tune:
- visual adapter
- patch embedding layer

Freeze:
- all other network components

#### Stage 2
Fine-tune on top of Stage 1:
- memory adapter
- patch embedding layer
- visual adapter

Freeze:
- all remaining parts

### Training command

```bash
# TODO: fill in the actual training command
# Example:
# python tracking/train.py --config configs/famtrack_xxx.yaml
```

---

## Evaluation / Testing

### Evaluation benchmarks and metrics

#### RGB-T
Benchmarks:
- RGBT234
- LasHeR

Metrics:
- **PR**: Precision Rate
- **SR**: Success Rate

#### RGB-D
Benchmarks:
- DepthTrack
- VOT22-RGBD

Metrics:
- **Pre**: Precision
- **Re**: Recall
- **F-score**
- **EAO**: Expected Average Overlap
- **A**: Accuracy
- **R**: Robustness

#### RGB-E
Benchmark:
- VisEvent

Metrics:
- **PR**: Precision Rate
- **SR**: Success Rate

### Testing command

```bash
# TODO: fill in the actual evaluation command
# Example:
# python tracking/test.py --dataset LasHeR --config configs/famtrack_xxx.yaml
```



## Raw Results

We keep the uploaded file **`raw_result.zip`** in this repository.

You can directly download it to:

- reproduce the reported comparisons,
- inspect the original tracking outputs,
- conduct fair result-level comparisons with your own methods.


---

## TODO

- [ ] Release training code
- [ ] Release evaluation code
- [ ] Release pretrained models
- [ ] Add environment setup instructions
- [ ] Add dataset preparation instructions
- [ ] Add visualization/demo scripts

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{FAMTrack,
  title = {Learning frequency and memory-aware prompts for multi-modal object tracking},
  journal = {Pattern Recognition},
  volume = {179},
  pages = {113532},
  year = {2026},
  doi = {https://doi.org/10.1016/j.patcog.2026.113532},
}
```

---

## Acknowledgment

This repository is built upon prior advances in RGB tracking and prompt-learning-based multi-modal tracking, especially the RGB foundation tracker used as the base model in our paper.

---


# TODO: add contact information
```
