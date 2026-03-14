# Multi View

Multi-view face as condition for human-centric video generation.
## ✔️ TODO List

- [x] Base code for single-shot video generation
- [x] Spliting RoPE for video and reference images
- [ ] Face selecting router
- [ ] Supporting multi-shot video generation
---

## 🚀 Training

```bash
bash train.sh
```
## 🚀 Inference

```bash
bash test.sh
```
## ⚙️ Configuration
```bash
YAML:
train_args:
  max_checkpoints_to_keep: 3
  resume_from_checkpoint: True
  seed: 42
  save_steps: 150
  save_epoches: 1
  batch_size: 8

  visual_log_project_name: Wan2.2_5B-Multi_view-normal_rope_384_640-3ref
  output_path: /root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts
  local_model_path: /root/paddlejob/workspace/qizipeng/wanx_pretrainedmodels

  zero_face_ratio: 0.1
  split_rope: False
  split1: False
  split2: False
  split3: False

infer_args:
  infer_step: 1350
  epoch_id: 17

dataset_args:
  base_path: /root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/datasets/merged_wangpan_artgrid_taobao_visionchina_123rf_nasuyun_xinpianchang_disk.json

  height: 384
  width: 640

  num_frames: 81
  ref_num: 3

```
---
