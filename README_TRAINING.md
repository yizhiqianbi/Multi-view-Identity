# Multi-view 视频生成模型训练文档

> 基于 Wan2.2-TI2V-5B 的多视角视频生成模型训练指南

---

## 目录

1. [项目概述](#1-项目概述)
2. [多机分布式训练原理](#2-多机分布式训练原理)
3. [代码架构分析](#3-代码架构分析)
4. [训练脚本详解](#4-训练脚本详解)
5. [当前问题与不足](#5-当前问题与不足)
6. [改进建议](#6-改进建议)

---

## 1. 项目概述


### 1.1 项目目标

本项目旨在训练一个**多视角一致性视频生成模型**，能够根据输入的多张不同角度的人脸参考图，生成保持人脸一致性的视频。

### 1.2 技术栈

| 组件 | 技术选型 |
|------|---------|
| 基础模型 | Wan2.2-TI2V-5B (Text-Image to Video) |
| 分布式框架 | DeepSpeed + Accelerate + torchrun |
| 显存优化 | ZeRO-2 + CPU Offload |
| 混合精度 | BF16 |
| 数据格式 | MP4 视频 + 人脸裁剪坐标 JSON |

### 1.3 训练规模

```
┌─────────────────────────────────────────┐
│  3 节点 × 8 GPU = 24 GPU 并行训练        │
│  Global Batch Size = 8 × 24 = 192       │
│  视频帧数 = 81 帧 (约 5 秒 @ 16fps)      │
│  分辨率 = 480 × 832                     │
└─────────────────────────────────────────┘
```

---

## 2. 多机分布式训练原理

### 2.1 分布式训练架构

```
                    ┌──────────────────────────────────────┐
                    │         MASTER NODE (rank=0)         │
                    │  IP: hostfile 第一行                  │
                    │  负责：协调、汇总梯度、保存checkpoint   │
                    └──────────────────┬───────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
            ▼                          ▼                          ▼
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
    │   Node 0      │          │   Node 1      │          │   Node 2      │
    │ GPU 0-7       │          │ GPU 0-7       │          │ GPU 0-7       │
    │ NODE_RANK=0   │◄────────►│ NODE_RANK=1   │◄────────►│ NODE_RANK=2   │
    └───────────────┘   NCCL   └───────────────┘   NCCL   └───────────────┘
                      AllReduce              AllReduce
```

### 2.2 关键概念说明

| 概念 | 含义 | 本项目配置 |
|------|------|-----------|
| **WORLD_SIZE** | 总进程数 | 24 (3×8) |
| **NODE_RANK** | 节点编号 | 0, 1, 2 |
| **LOCAL_RANK** | 节点内 GPU 编号 | 0-7 |
| **GLOBAL_RANK** | 全局进程编号 | 0-23 |
| **MASTER_ADDR** | 主节点 IP | hostfile 第一行 |
| **MASTER_PORT** | 通信端口 | 23000 |

### 2.3 通信流程 (Data Parallel)

```
Step 1: 数据分片
┌─────────────────────────────────────────────────────┐
│ Global Batch (192 samples)                          │
│  ├─ Node 0: samples 0-63   (8 GPU × 8 per GPU)     │
│  ├─ Node 1: samples 64-127                          │
│  └─ Node 2: samples 128-191                         │
└─────────────────────────────────────────────────────┘

Step 2: 前向传播 (各节点独立计算)
┌─────────────────────────────────────────────────────┐
│ Node 0: loss_0 = model(batch_0)                     │
│ Node 1: loss_1 = model(batch_1)                     │
│ Node 2: loss_2 = model(batch_2)                     │
└─────────────────────────────────────────────────────┘

Step 3: 反向传播 + AllReduce
┌─────────────────────────────────────────────────────┐
│ 各节点计算本地梯度 → NCCL AllReduce → 平均梯度       │
│ grad_avg = (grad_0 + grad_1 + grad_2) / 3           │
└─────────────────────────────────────────────────────┘

Step 4: 参数更新 (所有节点同步)
┌─────────────────────────────────────────────────────┐
│ params = params - lr × grad_avg                     │
└─────────────────────────────────────────────────────┘
```

### 2.4 ZeRO-2 优化原理

```
传统数据并行 (每张 GPU 都保存完整副本):
┌─────────────────────────────────────────────────────┐
│ GPU 0: [Model 5B] [Optimizer 20B] [Gradients 5B]   │
│ GPU 1: [Model 5B] [Optimizer 20B] [Gradients 5B]   │
│ ...    (大量冗余!)                                  │
└─────────────────────────────────────────────────────┘

ZeRO-2 (优化器状态 + 梯度分片):
┌─────────────────────────────────────────────────────┐
│ GPU 0: [Model 5B] [Opt_shard_0 0.8B] [Grad_0 0.2B] │
│ GPU 1: [Model 5B] [Opt_shard_1 0.8B] [Grad_1 0.2B] │
│ ...                                                 │
│ 显存节省 ≈ 8x (优化器) + Nx (梯度)                  │
└─────────────────────────────────────────────────────┘

本项目配置 (ZeRO-2 + CPU Offload):
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"},  ← 优化器卸载到 CPU
    "offload_param": {"device": "cpu"}       ← 参数也可卸载
  }
}
```

### 2.5 NCCL 通信配置

```bash
export NCCL_IB_DISABLE=0          # 启用 InfiniBand (高速网络)
export NCCL_SOCKET_IFNAME=xgbe0   # 指定网卡接口
export NCCL_IB_GID_INDEX=3        # IB GID 索引
export NCCL_IB_TIMEOUT=22         # 超时设置
```

| 通信方式 | 带宽 | 延迟 | 适用场景 |
|---------|------|------|---------|
| **InfiniBand** | 100-400 Gbps | 极低 | 高端集群 ✓ |
| RoCE | 25-100 Gbps | 低 | 以太网 IB 模拟 |
| TCP/IP | 10-25 Gbps | 中等 | 普通以太网 |

---

## 3. 代码架构分析

### 3.1 项目结构

```
Multi-view/
├── multi_view/
│   ├── train.py                 # 训练入口
│   ├── train_deepspeed_multinode.sh  # 多机启动脚本
│   ├── conf/
│   │   ├── multi-view.yaml      # 训练配置
│   │   └── ds_config_multinode.json  # DeepSpeed 配置
│   ├── datasets/
│   │   ├── videodataset.py      # 数据集类
│   │   └── videos_duration_5_7.json  # 数据标注
│   └── DiffSynth-Studio-main/
│       └── diffsynth/
│           ├── trainers/utils.py    # 训练工具
│           ├── pipelines/wan_video_new.py  # Pipeline
│           └── models/wan_video_dit.py     # DiT 模型
└── README_TRAINING.md           # 本文档
```

### 3.2 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  videos_duration_5_7.json                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ {                                                        │   │
│  │   "disk_path": "/path/to/video.mp4",                    │   │
│  │   "text": "A woman talking...",                         │   │
│  │   "facedetect_v1": [                                    │   │
│  │     [{"detect": {top,left,w,h}, "angle": {yaw,pitch}}], │   │
│  │     ...                                                  │   │
│  │   ],                                                     │   │
│  │   "facedetect_v1_frame_index": [0, 30, 60, ...]        │   │
│  │ }                                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          MulltiShot_MultiView_Dataset                   │   │
│  │  1. 解析 JSON                                           │   │
│  │  2. 筛选有效人脸 (prob > 0.99, size > 80px)            │   │
│  │  3. 按角度差异分组 (yaw/pitch/roll > 50°)              │   │
│  │  4. 视频采样 (5秒 → 81帧 @ 16fps)                      │   │
│  │  5. 裁剪人脸参考图 (3张)                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Output:                                                  │   │
│  │  - video: List[PIL.Image]  # 81 帧                      │   │
│  │  - ref_images: List[PIL.Image]  # 3 张人脸参考图        │   │
│  │  - single_caption: str  # 文本描述                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 模型结构

```
WanVideoPipeline
├── text_encoder (T5-XXL, frozen)     # 文本编码
├── image_encoder (frozen)            # 图像编码  
├── vae (frozen)                      # 视频 VAE
└── dit (DiT, trainable)              # 扩散 Transformer ← 训练目标
```

### 3.4 训练循环

```python
# train.py 核心流程
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 预处理
        inputs = model.forward_preprocess(batch)
        
        # 2. 计算 loss
        loss = model.forward(batch, args, inputs)
        
        # 3. 反向传播 (Accelerator 自动处理 ZeRO 梯度同步)
        accelerator.backward(loss)
        
        # 4. 优化器更新
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # 5. 保存 checkpoint
        checkpoint_manager.save_checkpoint(model, global_step, epoch)
```

---

## 4. 训练脚本详解

### 4.1 `train_deepspeed_multinode.sh` 逐段解析

#### 环境初始化
```bash
source /root/.../anaconda3/etc/profile.d/conda.sh
conda activate diffusion
```
激活 Python 环境，确保所有依赖可用。

#### 路径配置
```bash
WORKDIR=/root/paddlejob/workspace/qizipeng
PROJECT_DIR=${WORKDIR}/baidu/personal-code/Multi-view/multi_view
HOSTFILE=${WORKDIR}/hostfile
```
- `HOSTFILE` 包含所有节点的 IP 列表

#### 节点自动识别
```bash
MY_IP=$(hostname -I | awk '{print $1}')
NODE_RANK=$(grep -n -w "$MY_IP" "$HOSTFILE" | cut -d: -f1)
NODE_RANK=$((NODE_RANK - 1))
```
通过比对本机 IP 与 hostfile，自动计算当前节点编号。

#### torchrun 启动
```bash
torchrun \
    --nnodes=3 \                  # 3 个节点
    --nproc_per_node=8 \          # 每节点 8 进程
    --node_rank=${NODE_RANK} \    # 当前节点编号
    --master_addr=${MASTER_ADDR} \
    --master_port=23000 \
    train.py \
    --trainable_models "dit" \    # 只训练 DiT
    --learning_rate 1e-5 \
    --num_epochs 100 \
    --num_frames 81
```

### 4.2 `multi-view.yaml` 配置说明

```yaml
train_args:
  batch_size: 8              # 每 GPU batch size
  learning_rate: 1e-5        # 会被线性缩放
  num_epochs: 100
  save_steps: 150            # 每 150 步保存
  zero_face_ratio: 0.1       # 10% 概率不使用人脸参考

dataset_args:
  num_frames: 81             # 5秒 @ 16fps + 1
  height: 480
  width: 832
  ref_num: 3                 # 参考图数量
```

---

## 5. 当前问题与不足

### 5.1 🔴 严重问题

#### 问题 1: 脚本需要在每台机器手动运行

**现状:**
```bash
# 需要在 3 台机器分别执行
ssh node0 && bash train_deepspeed_multinode.sh
ssh node1 && bash train_deepspeed_multinode.sh
ssh node2 && bash train_deepspeed_multinode.sh
```

**问题:** 脚本注释说 "只需在主节点运行一次"，但实际使用 `torchrun` 而非 `deepspeed launcher`，**每台机器都需要手动启动**。

**建议修复:**
```bash
# 方案 A: 使用 pdsh 一键启动
pdsh -w node0,node1,node2 "cd $PROJECT_DIR && bash train.sh"

# 方案 B: 改用 deepspeed launcher (真正的单节点启动)
deepspeed --hostfile=$HOSTFILE train.py ...
```

---

#### 问题 2: 学习率缩放逻辑有 Bug

**现状 (train.py L165-169):**
```python
global_batch_size = args.batch_size * num_gpus  # 8 * 24 = 192
if global_batch_size > 1:
    args.learning_rate = min(
        args.learning_rate * global_batch_size,  # 1e-5 * 192 = 1.92e-3
        args.learning_rate * 10                  # 1e-5 * 10 = 1e-4 ← 实际使用
    )
```

**问题:**
1. 线性缩放上限设为 10x 过于保守
2. 标准做法是 `lr_scaled = lr_base * sqrt(global_batch_size)` 或带 warmup 的线性缩放
3. 日志打印的 `base_lr` 计算逻辑有误

**建议:**
```python
# 使用 sqrt 缩放 + warmup
base_lr = args.learning_rate
scaled_lr = base_lr * math.sqrt(global_batch_size / 8)  # 假设 8 是基准 batch
args.learning_rate = min(scaled_lr, base_lr * 32)  # 上限 32x
```

---

#### 问题 3: 数据集划分在每个进程独立进行

**现状 (videodataset.py L116-125):**
```python
random.seed(42)
test_indices = set(random.sample(range(total), test_count))
self.data_test = [self.data[i] for i in range(total) if i in test_indices]
self.data_train = [self.data[i] for i in range(total) if i not in test_indices]
```

**问题:**
- 虽然设置了 `seed=42`，但如果不同节点的数据加载顺序不同，`self.data` 的顺序可能不一致
- **可能导致 train/test 划分在不同节点不一致**

**建议:**
```python
# 在划分前对数据排序，确保顺序一致
self.data.sort(key=lambda x: x['video_path'])
random.seed(42)
# ... 然后划分
```

---

### 5.2 ?? 中等问题

#### 问题 4: DeepSpeed 配置文件每次运行都重写

**现状:**
```bash
cat > ${DS_CONFIG} << 'DSEOF'
{
  "zero_optimization": {...}
}
DSEOF
```

**问题:**
- 配置直接写死在 shell 脚本中，不便于调试和版本控制
- 每次运行都会覆盖已有配置

**建议:** 将 `ds_config_multinode.json` 作为独立文件维护，shell 脚本只检查其存在性。

---

#### 问题 5: 数据加载缺乏 prefetch 和 cache

**现状 (videodataset.py):**
```python
reader = imageio.get_reader(video_path)
for frame_id in sample_ids:
    frame = reader.get_data(int(frame_id))  # 逐帧读取
```

**问题:**
- 视频解码是 CPU 密集型操作
- 没有使用 `decord` 的 GPU 解码能力
- 没有数据预加载 (prefetch)

**建议:**
```python
# 使用 decord 批量读取
vr = VideoReader(video_path, ctx=cpu(0))
frames = vr.get_batch(sample_ids).asnumpy()

# 在 DataLoader 中启用 prefetch
DataLoader(..., num_workers=4, prefetch_factor=2)
```

---

#### 问题 6: 参考图角度筛选策略过于简单

**现状 (videodataset.py L168-172):**
```python
if (
    abs(angle_i["yaw"] - angle_j["yaw"]) > 50 or
    abs(angle_i["pitch"] - angle_j["pitch"]) > 50 or
    abs(angle_i["roll"] - angle_j["roll"]) > 50
):
```

**问题:**
- 只要任意一个角度差 > 50° 就选中，可能选到 pitch/roll 差异但 yaw 相似的图
- 没有保证"正面、侧面、另一侧"的多样性

**建议:**
```python
# 更完善的角度覆盖策略
def select_diverse_refs(angles, num_refs=3):
    # 使用 K-means 或贪心算法选择角度最分散的组合
    ...
```

---

### 5.3 🟢 轻微问题 / 建议

| 问题 | 描述 | 建议 |
|------|------|------|
| 硬编码路径 | `/root/paddlejob/...` 写死 | 使用环境变量或相对路径 |
| 缺少训练监控 | 没有 TensorBoard/WandB 的 image 日志 | 定期生成样本视频 |
| 没有梯度裁剪日志 | `gradient_clipping: 1.0` 但不知是否触发 | 添加梯度 norm 监控 |
| checkpoint 命名 | `checkpoint-step-X` 不含时间戳 | 加上 `datetime` |
| 无 EMA | 没有 Exponential Moving Average | 对扩散模型很重要 |

---

## 6. 改进建议

### 6.1 推荐的脚本改进

```bash
#!/bin/bash
# train_deepspeed_multinode_v2.sh

set -e

# ===== 配置 =====
WORKDIR=${WORKDIR:-/root/paddlejob/workspace/qizipeng}
PROJECT_DIR=${WORKDIR}/baidu/personal-code/Multi-view/multi_view
HOSTFILE=${WORKDIR}/hostfile
DS_CONFIG=${PROJECT_DIR}/conf/ds_config_multinode.json

# ===== 验证 =====
[[ -f "$HOSTFILE" ]] || { echo "❌ hostfile not found"; exit 1; }
[[ -f "$DS_CONFIG" ]] || { echo "❌ DeepSpeed config not found"; exit 1; }

# ===== 环境 =====
source ${WORKDIR}/anaconda3/etc/profile.d/conda.sh
conda activate diffusion
export PYTHONPATH="${PROJECT_DIR}/DiffSynth-Studio-main:${PYTHONPATH}"

# ===== 分布式配置 =====
export MASTER_ADDR=$(head -n1 "$HOSTFILE" | awk '{print $1}')
export MASTER_PORT=${MASTER_PORT:-23000}

NUM_MACHINES=$(wc -l < "$HOSTFILE")  # 自动检测节点数
GPUS_PER_MACHINE=${GPUS_PER_MACHINE:-8}

# ===== 节点识别 =====
MY_IP=$(hostname -I | awk '{print $1}')
NODE_RANK=$(grep -n "^${MY_IP}" "$HOSTFILE" | head -1 | cut -d: -f1)
NODE_RANK=$((NODE_RANK - 1))

echo "========================================
Multi-Node Training Configuration:
  MASTER: ${MASTER_ADDR}:${MASTER_PORT}
  NODES:  ${NUM_MACHINES} × ${GPUS_PER_MACHINE} GPU
  THIS:   ${MY_IP} (rank=${NODE_RANK})
========================================"

# ===== 启动 =====
cd ${PROJECT_DIR}

torchrun \
    --nnodes=${NUM_MACHINES} \
    --nproc_per_node=${GPUS_PER_MACHINE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --rdzv_backend=c10d \
    train.py \
    --train_yaml "${PROJECT_DIR}/conf/multi-view.yaml" \
    --model_id_with_origin_paths "Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
    --trainable_models "dit" \
    --learning_rate 1e-5 \
    --num_epochs 100 \
    "$@"  # 允许命令行覆盖参数
```

### 6.2 一键多节点启动脚本

```bash
#!/bin/bash
# launch_all_nodes.sh - 在主节点运行，自动 SSH 启动所有节点

HOSTFILE=${1:-hostfile}
SCRIPT="train_deepspeed_multinode.sh"

while read -r ip _; do
    echo "🚀 Launching on ${ip}..."
    ssh -o StrictHostKeyChecking=no "${ip}" \
        "cd ${PROJECT_DIR} && nohup bash ${SCRIPT} > train_${ip}.log 2>&1 &" &
done < "$HOSTFILE"

wait
echo "✅ All nodes launched. Check logs: train_*.log"
```

### 6.3 训练监控建议

```python
# 在 train.py 中添加
if accelerator.is_main_process and global_step % 500 == 0:
    # 生成样本视频
    with torch.no_grad():
        sample = model.pipe.generate(
            prompt="test prompt",
            ref_images=test_refs,
            num_frames=17,  # 短视频用于快速预览
        )
    wandb.log({"samples": wandb.Video(sample, fps=8)})
```

---

## 附录

### A. hostfile 格式示例

```
10.54.129.25 slots=8
10.54.130.142 slots=8
10.54.130.149 slots=8
```

### B. 常见错误排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `NCCL timeout` | 网络不通 | 检查防火墙、MASTER_PORT |
| `RuntimeError: CUDA OOM` | 显存不足 | 降低 batch_size，启用 gradient checkpointing |
| `FileNotFoundError: video.mp4` | 数据路径错误 | 检查 `disk_path` 挂载 |
| `NODE_RANK=-1` | IP 不在 hostfile | 检查 hostname -I 输出 |

### C. 参考资料

- [DeepSpeed ZeRO 文档](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Accelerate 多机训练](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)

---

*文档版本: v1.0 | 更新日期: 2025-01-23*