# 融合检索以及 Celeb-reID 基准测试

## 数据集

- 数据来源：Kaggle 下载的 Celeb-reID；官方数据说明参考 https://github.com/Huang-3/Celeb-reID
- 本地路径：`/Users/jsuikak/Desktop/course_person_retrieve/data/Celeb-reID`
- 数据集大小：257.34 MB (269845205 bytes)
- 内容：裁剪后的名人/人物行人图像，用于长期换装 ReID。
- 官方 split：train 632 ID / 20208 图，query 420 ID / 2972 图，gallery 420 ID / 11006 图，total 1052 ID / 34186 图。

本次扫描结果：

| split | 路径 | 图像数 | ID 数 | 大小 |
|---|---|---:|---:|---:|
| train | `/Users/jsuikak/Desktop/course_person_retrieve/data/Celeb-reID/train` | 20208 | 632 | 151.99 MB |
| query | `/Users/jsuikak/Desktop/course_person_retrieve/data/Celeb-reID/query` | 2972 | 420 | 22.50 MB |
| gallery | `/Users/jsuikak/Desktop/course_person_retrieve/data/Celeb-reID/gallery` | 11006 | 420 | 82.86 MB |

## Benchmark 选择理由

选择 Celeb-reID 作为当前融合算法的 benchmark，主要有以下原因：

1. 规模与当前算力资源匹配。数据集本地大小为 257.34 MB，本次主评测只使用 query 和 gallery，共 13978 张图像；完整 CPU 特征提取可以在可接受时间内完成，也便于重复实验和缓存复用。
2. 数据格式契合当前“双分支融合”算法。Celeb-reID 已经提供裁剪后的行人图像和官方 query/gallery split，每张图像可以同时作为人脸分支和整体行人分支的输入，不需要额外构造 pair、track 或视频标注。
3. 它不是单独的人脸验证集，也不是单独的行人特征验证集。图像中既包含整体行人外观，也可能包含可检测人脸，能直接检验 ArcFace 与 OSNet 这两类预训练特征在同一检索任务中的互补性。
4. 数据中存在人脸不可见、姿态变化、外观变化等情况，MTCNN 在这些样本上可能检测不到明确人脸。因此它能暴露 face 分支覆盖率不足的问题，并用于比较 ArcFace-only 的无效 query、OSNet-only 的全覆盖，以及 Fusion 在 face 子特征缺失时的表现差异。

目录预览：

```text
Celeb-reID/
|-- gallery/
|   |-- 100_25_1.jpg
|   |-- 100_44_0.jpg
|   |-- 100_59_0.jpg
|   |-- 100_6_1.jpg
|   |-- 101_2_1.jpg
|   |-- 101_36_0.jpg
|   |-- 101_3_0.jpg
|   |-- 101_40_1.jpg
|   |-- 101_45_0.jpg
|   |-- 101_46_0.jpg
|   `-- ... (10996 more)
|-- query/
|   |-- 100_13_0.jpg
|   |-- 101_13_0.jpg
|   |-- 101_15_0.jpg
|   |-- 101_18_0.jpg
|   |-- 102_13_0.jpg
|   |-- 102_15_0.jpg
|   |-- 102_18_0.jpg
|   |-- 102_19_0.jpg
|   |-- 103_10_1.jpg
|   |-- 103_11_0.jpg
|   `-- ... (2962 more)
|-- train/
|   |-- 100_10_1.jpg
|   |-- 100_12_0.jpg
|   |-- 100_13_0.jpg
|   |-- 100_1_0.jpg
|   |-- 100_25_0.jpg
|   |-- 100_2_0.jpg
|   |-- 100_36_3.jpg
|   |-- 100_3_0.jpg
|   |-- 100_43_0.jpg
|   |-- 100_44_0.jpg
|   `-- ... (20198 more)
`-- readme.txt
```

## 算法与向量检索方案

本节更新原“当前向量检索方案与 L2 归一化说明”的口径：原文主要描述 face/person 两路单独建库和单独检索；本实验保留 ArcFace-only、OSNet-only 作为对比，同时新增一个融合检索分支。Fusion 评测中，query 和 gallery 索引库都使用同一套融合特征。

本实验不训练新模型，只比较预训练特征的检索效果。

1. ArcFace-only：对裁剪行人图像先做人脸检测，取最大人脸输入 ArcFace；query 未检测到人脸时，该 query 在 face-only 指标中按失败计。
2. OSNet-only：不再跑 YOLO，直接把 Celeb-reID 的裁剪行人图像输入 OSNet，得到整体行人特征。
3. ArcFace+OSNet fusion：使用固定权重加权拼接，默认 `face_weight=0.35`，`person_weight=0.65`。

### 融合特征定义

给定任意一张图像 $x_i$，其中 $i$ 是样本编号；当 $i=q$ 时表示某一张 query 图像，当 $i=g$ 时表示 gallery 中某一张图像。对 $x_i$ 提取两路特征：

$$\mathbf{f}_i, \mathbf{p}_i \in \mathbb{R}^{512}$$

其中 $\mathbf{f}_i$ 是 ArcFace 人脸特征，$\mathbf{p}_i$ 是 OSNet 整体行人特征。若 MTCNN 检测到人脸，则 $\mathbf{f}_i$ 为 L2 归一化后的 ArcFace 特征，并记 $m_i^f=1$；若 MTCNN 未检测到人脸，则 $\mathbf{f}_i=\mathbf{0}$，并记 $m_i^f=0$。OSNet 分支直接使用整张裁剪行人图像，本实验中 person 特征全部提取成功，且 $\mathbf{p}_i$ 也做 L2 归一化。

固定权重为：

$$w_f=0.35, \quad w_p=0.65$$

先构造未归一化的拼接向量：

$$\mathbf{u}_i = [\sqrt{w_f}\mathbf{f}_i ; \sqrt{w_p}\mathbf{p}_i] \in \mathbb{R}^{1024}$$

再做一次 L2 归一化，得到最终融合特征：

$$\mathbf{z}_i = \frac{\mathbf{u}_i}{\max(\|\mathbf{u}_i\|_2, \epsilon)}$$

这里的 $\mathbf{z}_i$ 就是某一张图像的最终融合特征；在输出文件中，query 图像的 $\mathbf{z}_i$ 存在 `features/query_fused.npy`，gallery 图像的 $\mathbf{z}_i$ 存在 `features/gallery_fused.npy`。

### Query-gallery 匹配

Fusion 评测中，query 和 gallery 都使用同一个融合公式。对于某一张 query 图像 $x_q$ 和某一张 gallery 图像 $x_g$，分别得到融合特征：

$$\mathbf{z}_q=\mathrm{Fuse}(x_q), \quad \mathbf{z}_g=\mathrm{Fuse}(x_g)$$

其中 $\mathbf{z}_q$ 表示 query 图像 $x_q$ 的融合特征，$\mathbf{z}_g$ 表示 gallery 图像 $x_g$ 的融合特征。两者的检索相似度为点积：

$$S_{q,g}=\mathbf{z}_q^\top \mathbf{z}_g$$

因为 $\mathbf{z}_q$ 和 $\mathbf{z}_g$ 都经过 L2 归一化，这个点积等价于 cosine similarity。对每个 query，脚本会按照 $S_{q,g}$ 从大到小排序所有 gallery 图像，得到检索结果。

令 $d_i=\|\mathbf{u}_i\|_2$，则相似度可以展开为：

$$S_{q,g}=\frac{w_f\mathbf{f}_q^\top\mathbf{f}_g + w_p\mathbf{p}_q^\top\mathbf{p}_g}{d_q d_g}$$

若 query 和 gallery 两边都检测到人脸，且当前 $w_f+w_p=1$，则 $d_q=d_g=1$，公式简化为：

$$S_{q,g}=0.35\mathbf{f}_q^\top\mathbf{f}_g + 0.65\mathbf{p}_q^\top\mathbf{p}_g$$

因此，在两路特征都有效时，融合后的相似度等价于“人脸相似度”和“整体行人相似度”的加权求和。这里使用 $\sqrt{w_f}$ 和 $\sqrt{w_p}$ 乘到特征上，是因为点积展开时 $\sqrt{w}\cdot\sqrt{w}=w$，从而让相似度层面的权重正好是 $w_f$ 和 $w_p$。

当 MTCNN 未检测到人脸时，该图像的 face 子特征标记为无效：在 ArcFace-only 评测里，query 的 face 特征无效会计入“无效 Query”；在 Fusion 中，为了保持每张图都有融合向量，face 子特征用零向量占位，person 子特征仍正常参与。这个处理不是动态切换到 OSNet-only，也不是说人脸缺失样本变成了有效的人脸检索样本；它只是让 Fusion 方法在 face 子特征缺失时仍能计算一个固定格式的融合向量。进一步看，若 query 和 gallery 两边都缺失 face 子特征，则 $S_{q,g}=\mathbf{p}_q^\top\mathbf{p}_g$；若只有一侧缺失 face 子特征，则 person 相似度项会受到 L2 归一化后的系数影响，因此不严格等价于 OSNet-only。

### L2 归一化与 ArcFace 的关系

L2 归一化把非零向量投影到单位超球面上。对任意非零向量 $\mathbf{x}$，归一化后的向量是：

$$\hat{\mathbf{x}}=\frac{\mathbf{x}}{\|\mathbf{x}\|_2}, \quad \|\hat{\mathbf{x}}\|_2=1$$

归一化后，两个向量的点积只反映夹角：

$$\hat{\mathbf{x}}^\top\hat{\mathbf{y}}=\cos\theta$$

这也是当前匹配阶段做归一化的核心原因：检索排序希望比较“方向是否接近”，而不是让向量模长影响分数。在单位向量前提下，最大化点积和最小化欧氏距离的排序是一致的：

$$\|\hat{\mathbf{x}}-\hat{\mathbf{y}}\|_2^2 = 2 - 2\hat{\mathbf{x}}^\top\hat{\mathbf{y}}$$

ArcFace 本身就是角度判别思路。训练分类头时，ArcFace 会把特征方向和类别权重方向归一化，在单位超球面上比较角度。记 $\hat{\mathbf{x}}$ 为归一化特征，$\hat{\mathbf{W}}_j$ 为第 $j$ 个类别的归一化分类权重，则：

$$\cos\theta_j=\hat{\mathbf{W}}_j^\top\hat{\mathbf{x}}$$

对真实类别 $y$，ArcFace 在角度上加入 margin：

$$l_y=s\cos(\theta_y+m), \quad l_j=s\cos\theta_j,\quad j\ne y$$

其中 $m$ 是角度间隔，$s$ 是缩放系数。这个目标会鼓励同一身份的特征方向更接近、不同身份的特征方向在超球面上有更大的角度间隔。因此做人脸检索时，比较 L2 归一化后 embedding 的 cosine similarity 与 ArcFace 的训练目标是匹配的。

关于“ArcFace 输出是不是单位向量”：从模型 backbone 直接输出的原始 embedding 不一定天然是单位向量；ArcFace 的训练和分类头会显式使用归一化后的特征方向。当前项目代码中，`models/arcface.py` 的 `extract_feature()` 会调用 `F.normalize(feat, dim=1)`，`src/tools/feature_extractor.py` 的 `_face_forward()` 也会对模型输出再做一次 L2 normalize，flip test 平均后还会再次 normalize。因此，本 benchmark 中保存和参与匹配的 ArcFace 特征是单位向量；严格地说，它们位于 512 维单位超球面的表面上。MTCNN 检测不到人脸时保存的是零向量占位，这个零向量不是 ArcFace 的有效单位特征，只用于 Fusion 的固定格式拼接。

## 实验设置

- 主评测：官方 query 对 gallery，不使用 train。
- 设备：`cpu`
- person 模型：`osnet`
- ArcFace 权重：`/Users/jsuikak/Desktop/course_person_retrieve/models/weights/arcface.pt`
- 输出目录：`/Users/jsuikak/Desktop/course_person_retrieve/outputs/benchmarks/celebreid_fusion`
- 特征缓存目录：`/Users/jsuikak/Desktop/course_person_retrieve/outputs/benchmarks/celebreid_fusion/features`
- metadata 目录：`/Users/jsuikak/Desktop/course_person_retrieve/outputs/benchmarks/celebreid_fusion/metadata`
- 指标文件：`/Users/jsuikak/Desktop/course_person_retrieve/outputs/benchmarks/celebreid_fusion/metrics.json`、`/Users/jsuikak/Desktop/course_person_retrieve/outputs/benchmarks/celebreid_fusion/metrics.csv`
- 运行命令：`scripts/benchmark_celebreid_light_fusion.py`
- 总耗时：726.42 秒

特征提取统计：

| split | 总图像 | face 成功 | face 成功率 | person 成功 | person 成功率 | fused 维度 |
|---|---:|---:|---:|---:|---:|---:|
| query | 2972 | 1275 | 42.90% | 2972 | 100.00% | 1024 |
| gallery | 11006 | 4964 | 45.10% | 11006 | 100.00% | 1024 |

特征缓存大小：

- `gallery_face.npy`: 21.50 MB
- `gallery_face_valid.npy`: 10.87 KB
- `gallery_fused.npy`: 42.99 MB
- `gallery_fused_valid.npy`: 10.87 KB
- `gallery_person.npy`: 21.50 MB
- `gallery_person_valid.npy`: 10.87 KB
- `query_face.npy`: 5.80 MB
- `query_face_valid.npy`: 3.03 KB
- `query_fused.npy`: 11.61 MB
- `query_fused_valid.npy`: 3.03 KB
- `query_person.npy`: 5.80 MB
- `query_person_valid.npy`: 3.03 KB

特征文件说明：

- `features/query_face.npy`、`features/gallery_face.npy`：ArcFace 人脸特征，512 维。
- `features/query_person.npy`、`features/gallery_person.npy`：OSNet 行人特征，512 维。
- `features/query_fused.npy`、`features/gallery_fused.npy`：融合特征，1024 维。
- `features/*_valid.npy`：对应特征是否有效的布尔 mask；face 分支检测不到人脸时为 `False`。
- `metadata/query.csv`、`metadata/gallery.csv`：记录图像路径、identity、split 等信息；CSV 行号与同 split 的 `.npy` 第一维顺序一一对应。

## 结果

| 方法 | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) | mAP (%) | 有效 Query | 无效 Query |
|---|---:|---:|---:|---:|---:|---:|
| ArcFace-only | 26.75 | 32.17 | 33.68 | 7.35 | 1275 | 1697 |
| OSNet-only | 31.19 | 39.87 | 44.95 | 3.18 | 2972 | 0 |
| ArcFace+OSNet fusion | 35.70 | 45.69 | 51.55 | 4.51 | 2972 | 0 |

指标口径：Rank-K 表示 Top-K 内是否至少命中同身份；mAP 是所有 query 的 AP 平均。数值单位均为百分比。

“无效 Query”表示某个方法在该 query 上没有可用的查询特征。本实验中主要发生在 ArcFace-only：如果 MTCNN 在裁剪行人图像中未检测到人脸，就无法提取 ArcFace 特征，该 query 在 ArcFace-only 的 Rank-K 和 AP 中按 0 计，并且仍保留在总 query 分母内。OSNet-only 直接使用整张裁剪行人图像，因此没有无效 query。Fusion 的无效 query 为 0，是因为该方法定义为 `face_or_zero + person` 的固定格式融合；其中部分 query 的 face 子特征仍然是无效的，只是 fusion 向量可以通过零向量占位和 OSNet 特征继续构造。

## 效果分析

当前固定权重融合是一个有效 baseline，但还不是最优融合方案。相比 OSNet-only，Fusion 的 Rank-1 提升 +4.51 个百分点，Rank-5 提升 +5.82 个百分点，Rank-10 提升 +6.59 个百分点，mAP 提升 +1.32 个百分点。这说明人脸分支给整体行人特征补充了身份判别信息，尤其对 Top-K 早排位有帮助。

相比 ArcFace-only，Fusion 的 Rank-1 提升 +8.95 个百分点，但 mAP 变化为 -2.85 个百分点。原因是 ArcFace-only 只在检测到人脸时可用，query 人脸成功率为 42.90%（1275/2972），gallery 人脸成功率为 45.10%；检测到人脸的样本中 ArcFace 判别力较强，但覆盖率不足。Fusion 保留 OSNet 的全量覆盖能力，在有人脸时叠加 ArcFace，因此 Top-K 命中率更稳定。

mAP 没有超过 ArcFace-only，说明当前固定权重拼接主要改善了靠前位置的命中，但没有充分优化同身份所有 gallery 样本的整体排序。下一步可以尝试基于人脸检测置信度、脸部面积、图像质量或两路相似度分布做动态权重，也可以做 score-level fusion 和权重网格搜索。
