# 关键改动与逻辑说明（全量 Self‑Attention Mask + 每帧软屏蔽最近 Ref）

本文档说明**当前实现的核心改动与逻辑**：
- index 从哪里引入
- 最近 ref 如何计算
- 全量 `S×S` mask 如何构造
- mask 如何注入到 Self‑Attention

---

## 0) 总览（你现在的逻辑）

- **数据集**返回 `video_frame_indices` 与 `ref_frame_indices`（绝对帧号，不做归一化）。
- **训练输入**把 index 和 `ref_soft_mask_strength` 传到 pipeline。
- **pipeline**构造全量 `self_attn_mask`：
  - 对每个 latent 帧 t，找到最近 ref r
  - 把对应的 video→ref block 加上 `log(strength)`（软 mask）
- **Self‑Attention**接收 `attn_mask` 并传入 `scaled_dot_product_attention`。

---

## 1) index 从哪里来

### `datasets/videodataset.py`
- `sample_ids = np.linspace(...)` 得到采样帧的**绝对帧号**
- `video_frame_indices = sample_ids`
- `ref_frame_indices` 从文件名解析：`199_crop_face.png → 199`

### `datasets/videodataset_movie.py`
- 正常情况 `sample_ids = np.linspace(...)`
- 如果帧数不足，用顺序帧 + 尾帧填充，同时补齐 `sample_ids`
- `video_frame_indices = sample_ids`
- `ref_frame_indices` 从文件名中提取数字

数据返回时附带：
```
"ref_frame_indices": [...],
"video_frame_indices": [...],
```

**注意：**这里使用的是**绝对帧号**，没有做 0‑based offset。

---

## 2) 训练输入如何传递

### `train.py`
- 从 YAML 读取：
```
train_args:
  ref_soft_mask_strength: 0.3
```
- 传入 pipeline：
```
"ref_frame_indices": [d.get("ref_frame_indices") for d in data],
"video_frame_indices": [d.get("video_frame_indices") for d in data],
"ref_soft_mask_strength": args.ref_soft_mask_strength,
```

---

## 3) 最近 ref 的计算逻辑（包含首帧与窗口末帧）

位置：`diffsynth/pipelines/wan_video_new.py` → `WanVideoUnit_RefFrameSoftMask`

### 步骤
1) **视频帧 → latent 帧对齐**
```
# Wan time_division_remainder=1：首帧单独保留
# latent0 -> frame0, latent1 -> frame1, latent2 -> frame5, ...
if remainder == 1:
    latent_vid_idx = [video_frame_indices[0]] + video_frame_indices[1::stride]
else:
    latent_vid_idx = video_frame_indices[::stride]
latent_vid_idx = pad_or_trim(latent_vid_idx, F)
```

2) **最近 ref**（每个 latent 帧，考虑窗口末帧）
```
vid_idx_t = latent_vid_idx[t]
vid_idx_t_end = vid_idx_t + 3  # 当前 latent 对应的末帧（窗口长度=4）
dist_start = |vid_idx_t - ref[r]|
dist_end   = |vid_idx_t_end - ref[r]|
nearest = argmin_r (dist_start + dist_end)
```

---

## 4) 全量 `S×S` Mask 的构造

### 基本量
- `F` = latent 视频帧数
- `R` = ref 数量
- `tokens_per_frame = (H'/patch_h) * (W'/patch_w)`
- `S = (F + R) * tokens_per_frame`

### 初始化
```
attn_mask = zeros((B, S, S))
mask_value = log(strength)   # strength=0.3
```

### 对每个 latent 帧 t
- 找最近 ref = r
- 计算 block 位置：
  - 行：video frame t 对应的 tokens
  - 列：ref r 对应的 tokens
- 写入软 mask：
```
attn_mask[row_start:row_end, col_start:col_end] = mask_value
```

**效果：**仅对“视频帧 t → 最近 ref r”这块 attention 权重做软衰减。

---

## 5) `attn_mask` 在 PyTorch SDPA 里的语义（关键问题）

PyTorch `scaled_dot_product_attention` 支持 **bool mask** 和 **float mask**：  

- **bool mask**：`True` 表示“允许注意力”，`False` 表示“mask（不允许）”  
  实现上等价于把 `False` 位置加 `-inf`。  

- **float mask**（本项目使用的）：直接**加到 attention logits** 上。  
  - **0** = 不影响（正常注意力）  
  - **负数** = 降权（soft mask）  
  - **-inf** = 完全屏蔽（hard mask）

**所以：不相关区域应该是 `0`（float mask）**，这样不改变注意力。  
我们对最近 ref 的对应 block 写入 `log(strength)`：  
```
strength=0.3 → log(0.3)≈-1.204
```
若 `strength=0`，则写入 `-inf`。

---

## 5) Mask 注入 Self‑Attention

### `model_fn_wan_video`
- 接收 `self_attn_mask`
- 传递给 DiT block

### `wan_video_dit.py`
- `SelfAttention.forward(...)` 新增 `attn_mask`
- 最终调用：
```
F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
```

---

## 6) 日志（mask 尺寸打印）

首次构造 mask 时打印：
```
[RefMask] attn_mask shape=(B,S,S) dtype=... size~X.XXGB
```

---

## 7) 相关文件列表

- `datasets/videodataset.py`
- `datasets/videodataset_movie.py`
- `train.py`
- `conf/multi-view.yaml`
- `diffsynth/pipelines/wan_video_new.py`
- `diffsynth/models/wan_video_dit.py`
- `train_multi_node.sh`
