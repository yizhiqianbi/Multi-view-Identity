# 评测指标说明（IC2 / Pose）

本文描述当前实现的评测流程与公式，按你的最新要求：**只记录每帧 pose，不做 MD 评估**。

## 1. 身份一致性（IC2）

### 1.1 每帧相似度（按参考图取最大）
- 对生成视频的每一帧提取人脸特征向量 `f_i`
- 对 3 张参考图分别提取特征向量 `r_j`
- 计算余弦相似度并取最大：

```
score_i = max_j cosine(f_i, r_j)
```

> 这里 `cosine` 是 L2 归一化后的向量点积。

### 1.2 视频级身份一致性（IC2）
- 对所有有效帧取平均：

```
IC2 = (1 / N) * Σ_i score_i
```

> 其中 N 是有效帧数（默认“检测不到脸的帧”跳过）。


## 2. 每帧人脸姿态角（Pose 记录）

- 默认使用 MediaPipe FaceMesh 得到关键点
- 若 FaceMesh 未检测到人脸，则回退到 InsightFace 的 5 点关键点
- 通过 `solvePnP` + `decomposeProjectionMatrix` 估计头部姿态角
- 每帧得到 3 个角度：`pitch / yaw / roll`（单位为度）

每帧输出结构示例：

```
{
  "idx": 0,
  "pose": [pitch, yaw, roll],   # 如果该帧检测不到人脸，则为 null
  "missing": false
}
```


## 3. 结果输出字段（实现侧）

```
identity_consistency:
  ic2
  total_frames
  used_frames
  missing_frames

pose_per_frame:
  idx
  pose
  missing
```

## 4. 简要结论（你当前的理解是否正确）

✅ 当前只做两件事：
1) 每帧计算人脸角度（pose 记录）
2) 每帧对参考图算相似度并取最大值，最后取平均 = IC2

如需把这个指标说明写入别的文档或加上示例，我可以继续补充。
