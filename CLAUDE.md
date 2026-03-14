# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-view face conditioning for human-centric video generation using Wan2.2-TI2V-5B as the base diffusion model. The project extends the DiffSynth-Studio framework to add multi-view face identity conditioning via a custom `RefFrameSoftMask` pipeline unit.

## Common Commands

### Training
```bash
# Single-node training (8 GPUs)
cd multi_view
bash train.sh

# Multi-node distributed training (must be run manually on each node)
bash train_multi_node.sh
```

### Inference
```bash
cd multi_view
bash test.sh
```

### Installation
```bash
# Editable install (required when modifying DiffSynth-Studio-main)
pip install -e .
```

## Architecture

### Core Components

```
multi_view/
├── train.py                # Training entry point (Accelerate-based)
├── test.py                 # Inference entry point
├── datasets/
│   └── videodataset.py     # Video dataset with face crop extraction
├── DiffSynth-Studio-main/diffsynth/
│   ├── pipelines/
│   │   └── wan_video_new.py        # WanVideoPipeline with RefFrameSoftMask
│   ├── trainers/
│   │   └── utils.py                # Training utilities
│   └── models/
│       └── wan_video_dit.py        # DiT model (trainable)
├── conf/
│   ├── multi-view.yaml             # Main training/inference config
│   ├── accelerate_config_5B.yaml           # Single-node Accelerate config
│   └── accelerate_config_14B_L_multi-node.yaml  # Multi-node config
└── ckpts/                 # Checkpoint storage
```

### Pipeline Architecture

The `WanVideoPipeline` uses a modular unit system. Units process inputs sequentially and share state:

```python
self.units = [
    WanVideoUnit_ShapeChecker(),
    WanVideoUnit_NoiseInitializer(),
    WanVideoUnit_PromptEmbedder(),
    WanVideoUnit_InputVideoEmbedder(),
    WanVideoUnit_ImageEmbedderVAE(),
    WanVideoUnit_RefEmbedderFused(),
    WanVideoUnit_RefFrameSoftMask(),  # Custom multi-view conditioning
    WanVideoUnit_SpeedControl(),
    WanVideoUnit_CfgMerger(),
]
```

### Key Innovation: RefFrameSoftMask

The custom `WanVideoUnit_RefFrameSoftMask` implements per-frame soft attention between video tokens and reference frame tokens:

1. Maps video frame indices to latent frame indices (stride=4: 81 frames → 20 latents)
2. For each latent frame, finds the nearest reference frame by index
3. Applies soft mask to attention weights based on temporal distance

This enables temporal face conditioning without overfitting to specific reference frames.

### Data Flow

```
Dataset → VAE encode (video) → latents (B,C,20,H',W')
         → VAE encode (ref images) → latents (B,C,R,H',W')
         → RefFrameSoftMask → DiT forward → loss
```

### Dataset Requirements

- MP4 videos with pre-computed face crops
- Face crop filename format: `{frame_id}_crop_face.png`
- Configurable primary/fallback directories for face images
- Optional precomputed valid video list (`filterd_useable.json`) for faster startup

### Distributed Training

Uses Accelerate + DeepSpeed ZeRO-2:
- Mixed precision (BF16)
- Gradient accumulation
- CPU offload for optimizer state and params
- Multi-node: Must be launched manually on each node (no single-command multi-node launch)
- Known issues:
  - Dataset splitting may differ across nodes due to non-deterministic ordering
  - Learning rate scaling capped at 10x

### Configuration Key Parameters

`conf/multi-view.yaml`:
- `num_frames`: 81 (5 seconds @ 16fps)
- `ref_num`: 3 (number of reference frames)
- `mask_ref_ratio`: 0.6 (MAE-style masking on refs)
- `zero_face_ratio`: 0.1 (probability to zero out face refs)
- `ref_soft_mask_strength`: 1.0

### Model Checkpoints

- Text encoder, image encoder, VAE: frozen
- DiT: trainable (prefix `pipe.dit.` in checkpoints)
- Checkpoint format: safetensors (loaded via `safetensors` library)

### Training Args

Key command-line arguments:
- `--model_id_with_origin_paths`: Base model paths (DiT, T5, VAE)
- `--learning_rate`: Training learning rate
- `--num_frames`: Video length
- `--trainable_models`: `"dit"` (only DiT is trainable)
- `--extra_inputs`: `"cropped_images"` (face crop paths)
- `--dataset_repeat`: Dataset repetition factor

### Environment Variables

For multi-node training:
- `MASTER_ADDR`: Master node IP
- `MASTER_PORT`: Communication port (e.g., 29000)
- `NODE_RANK`: Current node rank (0-indexed)
- `NCCL_IB_DISABLE=1`: Disable IB/RDMA
- `NCCL_SOCKET_IFNAME`: Network interface
- `USE_FLASH_ATTN=0`: Required when attention mask is enabled

### Documentation Files

- `README.md`: Project overview and sample config
- `multi_view/README_TRAINING.md`: Comprehensive distributed training guide (Chinese)
- `multi_view/ATTN_MASK_IMPL_NOTES.md`: Technical notes on RefFrameSoftMask
- `multi_view/TRAINING_GUIDE.md`: Quick multi-node setup reference
