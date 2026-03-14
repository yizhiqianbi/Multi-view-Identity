#!/usr/bin/env python3
import torch
import sys

checkpoint_dir = "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts/Wan2.1_14B-T2V-Multi_view-normal_rope_wangpan_480_832pix_RoPE_split5_210_switchstep1000_mae0p6/checkpoint-step-150-epoch-1/pytorch_model"
output_file = "/tmp/check_fp32_groups.txt"

with open(output_file, 'w') as f:
    # Check rank 0 optim state
    s = torch.load(f"{checkpoint_dir}/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt", map_location='cpu', weights_only=False)
    opt_dict = s['optimizer_state_dict']

    f.write(f"zero_stage: {opt_dict['zero_stage']}\n")
    f.write(f"partition_count: {opt_dict['partition_count']}\n")

    fp32_groups = opt_dict.get('fp32_flat_groups', [])
    f.write(f"fp32_flat_groups count: {len(fp32_groups)}\n")

    if fp32_groups:
        for i, g in enumerate(fp32_groups):
            f.write(f"Group {i}: shape={g.shape}, dtype={g.dtype}\n")

    # Also check optim_state_dict key
    opt_state_dict = opt_dict.get('optimizer_state_dict', {})
    f.write(f"optimizer_state_dict keys: {list(opt_state_dict.keys())[:5]}\n")

print(f"Saved to {output_file}")
