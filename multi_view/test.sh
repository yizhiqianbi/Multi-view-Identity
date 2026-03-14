export http_proxy=http://10.63.229.53:8891
export https_proxy=http://10.63.229.53:8891
export HTTP_PROXY=http://10.63.229.53:8891
export HTTPS_PROXY=http://10.63.229.53:8891
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=localhost,127.0.0.1,::1

# GPU配置
# 单GPU模式: export CUDA_VISIBLE_DEVICES=0
# 多GPU模式: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 初始化 conda
source /root/paddlejob/workspace/qizipeng/anaconda3/etc/profile.d/conda.sh   # 你的 conda 安装路径可能不同
# 激活环境
conda activate diffusion

cd /root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view

CONF_YAML="/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/conf/multi-view.yaml"
PROJECT_NAME=Wan2.2_5B-Multi_view-normal_rope_wangpan_480_832pix_RoPE_split5_207
LOG_DIR="./output/${PROJECT_NAME}/log"

# 检查目录是否存在，如果不存在就创建
if [ ! -d "$LOG_DIR" ]; then
    echo "[$(date)] $LOG_DIR 不存在，正在创建..."
    mkdir -p "$LOG_DIR"
else
    echo "[$(date)] $LOG_DIR 已存在"
fi

cp -rf "${CONF_YAML}" ./output/${PROJECT_NAME}/log

# 多GPU测试配置
NUM_GPUS=8  # 使用的GPU数量
TEST_MODE="bench"  # 测试模式: train_set, selected10, original
NUM_SAMPLES=20  # train_set模式下测试的样本数量

echo "================================================"
echo "Test Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Test Mode: $TEST_MODE"
echo "  Num Samples: $NUM_SAMPLES"
echo "  Config: $CONF_YAML"
echo "================================================"

# 使用 torchrun 启动多GPU测试
torchrun --nproc_per_node=$NUM_GPUS test.py \
    --train_yaml "${CONF_YAML}" \
    --num_frames 81 \
    --test_mode "$TEST_MODE" \
    --num_test_samples $NUM_SAMPLES
