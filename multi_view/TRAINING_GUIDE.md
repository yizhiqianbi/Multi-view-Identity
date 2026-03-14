# Multi-View 分布式训练执行指南

## 📋 项目概述

- **模型**: Wan2.2-TI2V-5B (只训练 DiT 部分)
- **框架**: Accelerate + DDP
- **规模**: 3 节点 × 8 GPU = 24 GPU
- **任务**: 多视角人脸一致性视频生成

---

## 🖥️ 集群配置

### 节点信息 (hostfile)

```
10.54.129.150 slots=8    # Node 0 (Master)
10.54.130.157 slots=8    # Node 1
10.54.130.142 slots=8    # Node 2
```

### 训练参数 (conf/multi-view.yaml)

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 8 | 每卡 batch |
| global_batch | 192 | 8 × 24 GPU |
| learning_rate | 1e-5 → 1e-4 | 自动缩放 |
| num_frames | 81 | 约 5-7 秒视频 |
| resolution | 480×832 | |
| ref_num | 3 | 参考人脸数 |
| save_steps | 150 | 保存间隔 |

---

## 🚀 启动方式

### 方式一：每台机器分别执行（推荐）

在 **所有 3 台机器** 上分别执行：

```bash
cd /root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view
bash train_deepspeed_multinode.sh
```

> 脚本会自动识别当前机器的 IP 并计算 NODE_RANK

### 方式二：SSH 批量启动

在主节点 (10.54.129.150) 执行：

```bash
# 方法 A: 后台启动所有节点
for host in 10.54.129.150 10.54.130.157 10.54.130.142; do
    ssh $host "cd /root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view && \
               nohup bash train_deepspeed_multinode.sh > /tmp/train_${host}.log 2>&1 &" &
done
wait
echo "✅ 所有节点已启动"

# 方法 B: 使用 pdsh (如果可用)
pdsh -w 10.54.129.150,10.54.130.157,10.54.130.142 \
    "cd /root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view && bash train_deepspeed_multinode.sh"
```

### 方式三：单机调试 (8 卡)

```bash
cd /root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view

# 1. 激活环境
source /root/paddlejob/workspace/qizipeng/anaconda3/etc/profile.d/conda.sh
conda activate diffusion

# 2. 设置环境变量
export PYTHONPATH="${PWD}/DiffSynth-Studio-main:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 3. 启动训练
torchrun --nproc_per_node=8 train.py \
    --model_id_with_origin_paths "Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
    --learning_rate 1e-5 \
    --num_epochs 100 \
    --num_frames 81 \
    --trainable_models "dit" \
    --train_yaml "${PWD}/conf/multi-view.yaml"
```

### 方式四：少卡调试 (2 卡)

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 train.py \
    --model_id_with_origin_paths "Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --num_frames 81 \
    --trainable_models "dit" \
    --train_yaml "${PWD}/conf/multi-view.yaml"
```

---

## ✅ 启动前检查

### 1. SSH 免密登录
```bash
ssh 10.54.130.157 "hostname"  # 应返回主机名
ssh 10.54.130.142 "hostname"
```

### 2. Conda 环境
```bash
conda activate diffusion
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA GPUs: {torch.cuda.device_count()}')"
```

### 3. 网络连通性
```bash
# 测试 NCCL 端口
nc -zv 10.54.129.150 23000

# 检查网卡
ip addr | grep xgbe0
```

### 4. GPU 状态
```bash
nvidia-smi
```

---

## 📊 训练监控

### 查看日志
```bash
# 实时查看主进程日志
tail -f ckpts/Wan2.2_5B-Multi_view-*/log/rank_0.log

# 查看特定节点
tail -f /tmp/train_10.54.130.157.log
```

### 查看 GPU 使用
```bash
watch -n 1 nvidia-smi
```

### 查看 Loss 曲线
```bash
# Loss 图自动保存在
ls training_loss_plots/*/loss_mean.png
```

---

## 💾 检查点管理

### 检查点位置
```
ckpts/{project_name}/
├── checkpoint-step-150-epoch-1/
│   ├── model.safetensors        # 完整模型
│   ├── optimizer.bin            # 优化器状态
│   ├── scheduler.bin            # 学习率调度器
│   ├── trainer_state.json       # {"global_step": 150}
│   └── weights.safetensors      # 可训练权重 (只有 DiT)
├── checkpoint-step-300-epoch-1/
└── ...
```

### 断点续训

修改 `conf/multi-view.yaml`:
```yaml
train_args:
  resume_from_checkpoint: True   # 自动从最新检查点恢复
  # 或指定路径
  # resume_from_checkpoint: "/path/to/checkpoint-step-1500"
```

---

## 🔧 常见问题

### Q1: NCCL 超时
```bash
# 增加超时
export NCCL_IB_TIMEOUT=22

# 检查 InfiniBand
ibstat
```

### Q2: 显存不足 (OOM)
```yaml
# 1. 减小 batch_size
train_args:
  batch_size: 4

# 2. 启用梯度检查点 (train.py 参数)
--use_gradient_checkpointing_offload True
```

### Q3: 进程卡住或僵死
```bash
# 杀掉所有训练进程
pkill -f "torchrun.*train.py"

# 清理缓存
rm -rf /tmp/torch_distributed_*
```

### Q4: 只想在某些 GPU 上运行
```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 train.py ...
```

---

## 📁 相关文件

```
multi_view/
├── train.py                      # 训练入口
├── train_deepspeed_multinode.sh  # 多机启动脚本
├── conf/
│   └── multi-view.yaml           # 训练配置
├── datasets/
│   └── videodataset.py           # 数据集
└── DiffSynth-Studio-main/
    └── diffsynth/
        ├── pipelines/wan_video_new.py  # 模型 Pipeline
        └── trainers/utils.py           # 训练工具
```

---

## 🔄 完整启动流程 Checklist

- [ ] 确认所有节点 SSH 免密登录
- [ ] 确认 Conda 环境可用
- [ ] 确认 hostfile 配置正确
- [ ] 确认 multi-view.yaml 配置正确
- [ ] 确认数据路径可访问
- [ ] 确认模型权重路径可访问
- [ ] 在所有节点执行启动脚本
- [ ] 监控日志确认正常运行

---

*Happy Training! 🚀*

# 在主节点执行，一次性终止所有 3 台机器
for host in 10.54.129.150 10.54.130.157 10.54.130.142; do
    ssh $host "pkill -f 'torchrun.*train.py'" &
done
wait
echo "✅ 所有节点进程已终止"