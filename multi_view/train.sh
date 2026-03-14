# 初始化 conda
source /root/paddlejob/workspace/qizipeng/anaconda3/etc/profile.d/conda.sh   # 你的 conda 安装路径可能不同
# 激活环境
conda activate diffusion

export http_proxy=http://10.63.229.53:8891
export https_proxy=http://10.63.229.53:8891
export HTTP_PROXY=http://10.63.229.53:8891
export HTTPS_PROXY=http://10.63.229.53:8891
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=localhost,127.0.0.1,::1

export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PROJECT_NAME=Wan2.2_5B-Multi_view-normal_rope_wangpan_480_832pix_RoPE_split5_210_switchstep1000_mae0p6_dynamic_input ###piexels
# export WANDB_MODE=offline      # 关键：离线
# export WANDB_DIR=./save_checkpoint/${PROJECT_NAME}/wandb_log # 可选：本地保存目录
# cd /root/paddlejob/workspace/qizipeng
# bash /root/paddlejob/workspace/qizipeng/tar.sh

cd /root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view
### 下面的代码 让在多机，仅使用一台机器训练
# 明确告诉后端：主节点就是本机
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=39900       # 换成一个空闲端口也行
# 防止去走 IB/RDMA 或跨网卡
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo    # 或者写本机实际网卡名，lo 最保险只走回环
export NCCL_CROSS_NIC=0
#（可选）把 elastic/多机相关的残留变量清掉
unset WORLD_SIZE NODE_RANK

# 检查端口占用并杀死进程
if lsof -i :$MASTER_PORT > /dev/null 2>&1; then
    echo "[INFO] 端口 $MASTER_PORT 被占用，正在杀死相关进程..."
    kill -9 $(lsof -t -i:$MASTER_PORT)
    sleep 2
    # 再次检查是否成功杀死
    if lsof -i :$MASTER_PORT > /dev/null 2>&1; then
        echo "[WARNING] 端口 $MASTER_PORT 仍然被占用，可能需要手动处理"
    else
        echo "[INFO] 端口 $MASTER_PORT 已释放"
    fi
else
    echo "[INFO] 端口 $MASTER_PORT 未被占用"
fi

#-------------- 写 YAML（优先 yq，回退 sed） --------------
CONF_YAML="/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/conf/multi-view.yaml"

LOG_DIR="/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts/${PROJECT_NAME}/log"
# 检查目录是否存在，如果不存在就创建
if [ ! -d "$LOG_DIR" ]; then
    echo "[$(date)] $LOG_DIR 不存在，正在创建..."
    mkdir -p "$LOG_DIR"
else
    echo "[$(date)] $LOG_DIR 已存在"
fi
cp -rf "${CONF_YAML}" /root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts/${PROJECT_NAME}/log

set +e
accelerate launch --config_file="/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/conf/accelerate_config_14B.yaml" \
  --main_process_port $MASTER_PORT \
  train.py \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --num_frames 81 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --trainable_models "dit" \
  --train_yaml "${CONF_YAML}" \
  --extra_inputs "cropped_images" \

EXIT_CODE=$?
set -e

  # --use_gradient_checkpointing_offload \
  # --output_path "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/ckpts/${PROJECT_NAME}" \



if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[parent] training failed with code ${EXIT_CODE}"
  echo "[parent] running gpu_hog.py to investigate GPU status..."
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /root/paddlejob/workspace/qizipeng/gpu_hog.py
  exit $EXIT_CODE
else
  echo "[$(date)] 训练正常结束 ✅"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /root/paddlejob/workspace/qizipeng/gpu_hog.py
