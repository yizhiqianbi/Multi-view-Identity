#!/usr/bin/env python3
"""Merge partial DeepSpeed checkpoint (8 shards) for testing"""
import torch
import os
import glob
from safetensors.torch import save_file
from tqdm import tqdm

checkpoint_dir = "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts/Wan2.1_14B-T2V-Multi_view-normal_rope_wangpan_480_832pix_RoPE_split5_210_switchstep1000_mae0p6/checkpoint-step-150-epoch-1/pytorch_model"
output_dir = "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts/Wan2.1_14B-T2V-Multi_view-normal_rope_wangpan_480_832pix_RoPE_split5_210_switchstep1000_mae0p6/checkpoint-step-150-epoch-1/merged_8_shards"
os.makedirs(output_dir, exist_ok=True)

# Load all model state shards
model_files = sorted(glob.glob(os.path.join(checkpoint_dir, "zero_pp_rank_*_mp_rank_00_model_states.pt")))
print(f"Found {len(model_files)} model state shards")

all_states = []
for f in tqdm(model_files, desc="Loading shards"):
    state = torch.load(f, map_location='cpu', weights_only=False)
    all_states.append(state)

# Collect all parameters from all shards
merged_state_dict = {}
for rank, state in enumerate(all_states):
    module_state = state.get('module', {})
    for key, value in tqdm(module_state.items(), desc=f"Processing rank {rank}"):
        # Skip buffers
        if key in state.get('buffer_names', []):
            continue
        # Add parameter with rank suffix if duplicate
        if key in merged_state_dict:
            print(f"Warning: duplicate key {key}, renaming with rank {rank}")
            merged_state_dict[f"{key}_rank{rank}"] = value
        else:
            merged_state_dict[key] = value

print(f"\nMerged {len(merged_state_dict)} parameters")

# Save in shards (max 5GB per shard)
max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB
current_shard = {}
current_size = 0
shard_idx = 0

for key, tensor in tqdm(merged_state_dict.items(), desc="Saving shards"):
    tensor_size = tensor.element_size() * tensor.nelement()

    if current_size + tensor_size > max_shard_size and current_shard:
        # Save current shard
        shard_path = os.path.join(output_dir, f"model-{shard_idx:05d}-of-XXXXX.safetensors")
        save_file(current_shard, shard_path)
        print(f"Saved shard {shard_idx} with {len(current_shard)} tensors")
        current_shard = {}
        current_size = 0
        shard_idx += 1

    current_shard[key] = tensor
    current_size += tensor_size

# Save last shard
if current_shard:
    shard_path = os.path.join(output_dir, f"model-{shard_idx:05d}-of-XXXXX.safetensors")
    save_file(current_shard, shard_path)
    print(f"Saved shard {shard_idx} with {len(current_shard)} tensors")

print(f"\nTotal shards: {shard_idx + 1}")
print(f"Output directory: {output_dir}")
