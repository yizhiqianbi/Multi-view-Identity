#!/usr/bin/env python3
"""Merge DeepSpeed ZeRO-2 checkpoint (8 shards) for testing resume"""
import torch
import os
import gc
from tqdm import tqdm

checkpoint_dir = "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts/Wan2.1_14B-T2V-Multi_view-normal_rope_wangpan_480_832pix_RoPE_split5_210_switchstep1000_mae0p6/checkpoint-step-150-epoch-1"
output_dir = "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts/Wan2.1_14B-T2V-Multi_view-normal_rope_wangpan_480_832pix_RoPE_split5_210_switchstep1000_mae0p6/checkpoint-step-150-epoch-1/merged_for_resume"
os.makedirs(output_dir, exist_ok=True)

# Load all optim state shards (these contain the real fp32 weights)
optim_files = sorted([
    os.path.join(checkpoint_dir, "pytorch_model", f"bf16_zero_pp_rank_{i}_mp_rank_00_optim_states.pt")
    for i in range(8)
])
print(f"Found {len(optim_files)} optim state shards")

# Load all optim states (these contain fp32 master weights)
all_optim_states = []
for f in tqdm(optim_files, desc="Loading optim shards"):
    s = torch.load(f, map_location='cpu', weights_only=False)
    all_optim_states.append(s)

# Get fp32 flat groups from each rank
fp32_groups_per_rank = []
for i, s in enumerate(all_optim_states):
    opt_dict = s['optimizer_state_dict']
    fp32_groups = opt_dict.get('SINGLE_PARTITION_OF_FP32_GROUPS', [])
    fp32_groups_per_rank.append(fp32_groups)
    print(f"Rank {i}: fp32 groups={len(fp32_groups)}")
    if fp32_groups:
        for j, g in enumerate(fp32_groups):
            print(f"  Group {j}: shape={g.shape}, dtype={g.dtype}")

# Check if we can merge
if not fp32_groups_per_rank[0]:
    print("ERROR: No fp32 groups found in optim states!")
    exit(1)

# Get param shapes from model_states
model_state_file = os.path.join(checkpoint_dir, "pytorch_model", "zero_pp_rank_0_mp_rank_00_model_states.pt")
model_state = torch.load(model_state_file, map_location='cpu', weights_only=False)
param_shapes = model_state['param_shapes'][0]
print(f"\nTotal trainable parameters: {len(param_shapes)}")

# Merge fp32 weights across ranks
num_param_groups = len(fp32_groups_per_rank[0])
merged_fp32_groups = []

for group_idx in range(num_param_groups):
    print(f"\nMerging param group {group_idx}...")
    group_partitions = [fp32_groups_per_rank[i][group_idx] for i in range(8)]
    merged = torch.cat(group_partitions, dim=0)
    merged_fp32_groups.append(merged)
    print(f"  Merged shape: {merged.shape}")

# Reconstruct state_dict from merged fp32 weights and param shapes
state_dict = {}
buffer_names = model_state['buffer_names']
module_state = model_state['module']

# Add buffers (from rank 0)
for buf_name in buffer_names:
    if buf_name in module_state:
        state_dict[buf_name] = module_state[buf_name].float()

# Reconstruct parameters from merged fp32 groups
total_params = 0
total_elements = 0
offset = 0

for shapes, merged_group in zip(param_shapes, merged_fp32_groups):
    for param_name, shape in tqdm(shapes.items(), desc="Reconstructing params"):
        numel = shape.numel() if hasattr(shape, 'numel') else int(shape[0]) * int(shape[1])
        tensor = merged_group[offset:offset+numel].view(shape)
        state_dict[param_name] = tensor
        offset += numel
        total_params += 1
        total_elements += numel

print(f"\nReconstructed {total_params} parameters, {total_elements} elements")

# Save the merged state_dict
output_file = os.path.join(output_dir, "merged_state_dict.pt")
torch.save(state_dict, output_file)
print(f"\nSaved merged state_dict to: {output_file}")

# Also save as safetensors
from safetensors.torch import save_file
save_file(state_dict, os.path.join(output_dir, "merged_model.safetensors"))
print(f"Also saved as safetensors")

print("\nDone!")
print("Note: Only 8/40 ranks were merged, so dit weights are partial.")
print("This is for testing resume functionality only.")
