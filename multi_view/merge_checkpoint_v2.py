#!/usr/bin/env python3
"""Correctly merge DeepSpeed ZeRO-2 checkpoint with only dit as trainable"""
import torch
import os
import glob
from safetensors.torch import save_file
from tqdm import tqdm

checkpoint_dir = "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts/Wan2.1_14B-T2V-Multi_view-normal_rope_wangpan_480_832pix_RoPE_split5_210_switchstep1000_mae0p6/checkpoint-step-150-epoch-1/pytorch_model"
output_dir = "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts/Wan2.1_14B-T2V-Multi_view-normal_rope_wangpan_480_832pix_RoPE_split5_210_switchstep1000_mae0p6/checkpoint-step-150-epoch-1/merged"
os.makedirs(output_dir, exist_ok=True)

# Load all model state shards
model_files = sorted(glob.glob(os.path.join(checkpoint_dir, "zero_pp_rank_*_mp_rank_00_model_states.pt")))
print(f"Found {len(model_files)} model state shards")

all_states = []
for f in tqdm(model_files, desc="Loading shards"):
    state = torch.load(f, map_location='cpu', weights_only=False)
    all_states.append(state)

# Get trainable param names from param_shapes
trainable_params = set()
for shape_dict in all_states[0]['param_shapes']:
    trainable_params.update(shape_dict.keys())
print(f"Found {len(trainable_params)} trainable parameters (dit)")

# Frozen parameters (same on all ranks) - take from rank 0
frozen_params = {}
for key, value in tqdm(all_states[0]['module'].items(), desc="Processing frozen params"):
    if key not in trainable_params and key not in all_states[0]['buffer_names']:
        frozen_params[key] = value
print(f"Collected {len(frozen_params)} frozen parameters")

# Buffers (take from rank 0)
buffers = {}
for key in all_states[0]['buffer_names']:
    if key in all_states[0]['module']:
        buffers[key] = all_states[0]['module'][key]
print(f"Collected {len(buffers)} buffers")

# For trainable params, check if they are actually sharded or duplicated
# Let's examine one dit param to understand the structure
first_dit_param = list(trainable_params)[0]
print(f"\nExamining {first_dit_param}:")
for i, state in enumerate(all_states[:3]):
    val = state['module'].get(first_dit_param)
    if val is not None:
        print(f"  Rank {i}: shape={val.shape}, dtype={val.dtype}, mean={val.float().mean().item():.6f}")

# Check if values are the same or different
vals = []
for i, state in enumerate(all_states):
    val = state['module'].get(first_dit_param)
    if val is not None:
        vals.append(val)

if len(vals) > 1:
    is_same = torch.allclose(vals[0].float(), vals[1].float())
    print(f"  Rank 0 and Rank 1 have same values: {is_same}")

# If values are same across ranks, just take rank 0
# If different, we need to merge (but this shouldn't happen with ZeRO-2)
merged_state_dict = {}
merged_state_dict.update(frozen_params)
merged_state_dict.update(buffers)

for param_name in tqdm(trainable_params, desc="Processing trainable params"):
    val = all_states[0]['module'].get(param_name)
    if val is not None:
        merged_state_dict[param_name] = val

print(f"\nTotal merged parameters: {len(merged_state_dict)}")

# Save in shards (max 5GB per shard)
max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB
current_shard = {}
current_size = 0
shard_idx = 0

for key, tensor in tqdm(merged_state_dict.items(), desc="Saving shards"):
    tensor_size = tensor.element_size() * tensor.nelement()

    if current_size + tensor_size > max_shard_size and current_shard:
        shard_path = os.path.join(output_dir, f"model-{shard_idx:05d}-of-XXXXX.safetensors")
        save_file(current_shard, shard_path)
        print(f"Saved shard {shard_idx} with {len(current_shard)} tensors ({current_size/1024/1024/1024:.2f}GB)")
        current_shard = {}
        current_size = 0
        shard_idx += 1

    current_shard[key] = tensor
    current_size += tensor_size

# Save last shard
if current_shard:
    shard_path = os.path.join(output_dir, f"model-{shard_idx:05d}-of-XXXXX.safetensors")
    save_file(current_shard, shard_path)
    print(f"Saved shard {shard_idx} with {len(current_shard)} tensors ({current_size/1024/1024/1024:.2f}GB)")

print(f"\nTotal shards: {shard_idx + 1}")
print(f"Output directory: {output_dir}")
