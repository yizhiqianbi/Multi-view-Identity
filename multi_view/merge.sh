source /root/paddlejob/workspace/qizipeng/anaconda3/etc/profile.d/conda.sh   # 你的 conda 安装路径可能不同
conda activate diffusion

WORK_DIR=/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts
PROJECT=Wan2.1_14B-T2V-Multi_view-normal_rope_wangpan_480_832pix_RoPE_split5_210_switchstep1000_mae0p6
STEP=4650
EPOCH=17

# Torch 2.5.1+cu124 requires nvJitLink 12.4 symbols. Ensure conda CUDA libs
# are searched before system CUDA libs (for example /usr/local/cuda/lib64).
if [ -n "${CONDA_PREFIX:-}" ]; then
    PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    NVIDIA_ROOT="$CONDA_PREFIX/lib/python${PYVER}/site-packages/nvidia"
    if [ -d "$NVIDIA_ROOT" ]; then
        CUDA_LIB_PATHS="$NVIDIA_ROOT/nvjitlink/lib:$NVIDIA_ROOT/cusparse/lib:$NVIDIA_ROOT/cublas/lib:$NVIDIA_ROOT/cuda_runtime/lib"
        export LD_LIBRARY_PATH="$CUDA_LIB_PATHS:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    fi
fi


python $WORK_DIR/$PROJECT/checkpoint-step-$STEP-epoch-$EPOCH/zero_to_fp32.py \
    $WORK_DIR/$PROJECT/checkpoint-step-$STEP-epoch-$EPOCH \
    $WORK_DIR/$PROJECT/checkpoint-step-$STEP-epoch-$EPOCH \
    --safe_serialization \
    --max_shard_size "100GB"
