好的，按 PDF 精炼成“**有哪些指标 + 怎么算**”两部分：

---

## 1) Identity Consistency（身份一致性）

**特征**：用人脸识别模型 **Face-cur（CurricularFace）** 提取特征，计算与参考图的 **cosine similarity**。

设：

* 视频帧特征：(f_i)（(i=1..numf)）
* 参考图特征：(r_j)（(j=1..numr)）
* 相似度：(\cos(f_i, r_j))

### Method ：每帧对参考取 max，再对帧平均 

[
IC_2=\frac{1}{numf}\sum_{i=1}^{numf}\max_{j}\cos(f_i,r_j)
]

---

## 2) Motion Dynamic（运动动态/跟随度）

**做法**：先估计视频每帧人脸姿态角（angle），再算生成角度与条件角度的距离。

设：

* 生成角：(a^{gen}_i)
* 条件角：(a^{cond}_i)

一个常用实现（逐帧 L2 平均）：
[
MD=\frac{1}{N}\sum_{i}\left|a^{gen}_i-a^{cond}_i\right|_2
]

（(MD) 越小越好，表示更贴条件运动。）

---

## 3) Human Evaluation（人评：Win-rate）

**维度**（PDF 给的四项）：画质、身份一致性、prompt 遵循、运动动态。

**统计**：A/B 对比，按维度计 win-rate（谁被选得更多谁 win-rate 更高）。

---

如果你想我再进一步“更工程化但仍精炼”，我可以给你一行版的输出字段（IC1/IC2/MD/winrates）和推荐默认实现细节（比如无脸帧怎么处理）。

