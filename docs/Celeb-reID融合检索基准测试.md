# Celeb-reID 融合检索基准测试

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

## 算法

本实验不训练新模型，只比较预训练特征的检索效果。

1. ArcFace-only：对裁剪行人图像先做人脸检测，取最大人脸输入 ArcFace；query 未检测到人脸时，该 query 在 face-only 指标中按失败计。
2. OSNet-only：不再跑 YOLO，直接把 Celeb-reID 的裁剪行人图像输入 OSNet，得到整体行人特征。
3. ArcFace+OSNet fusion：使用固定权重加权拼接，默认 `face_weight=0.35`，`person_weight=0.65`。

融合公式：

$$
\mathbf{z}_{\mathrm{fuse}}
= \operatorname{L2Norm}\left(
\left[
\sqrt{w_f}\,\mathbf{z}_{\mathrm{face}},
\sqrt{w_p}\,\mathbf{z}_{\mathrm{person}}
\right]
\right)
$$

其中 $w_f=0.35$，$w_p=0.65$。若人脸检测失败，则 $\mathbf{z}_{\mathrm{face}}=\mathbf{0}$。

当 MTCNN 未检测到人脸时，该图像的 face 子特征标记为无效：在 ArcFace-only 评测里，query 的 face 特征无效会计入“无效 Query”；在 Fusion 中，为了保持每张图都有融合向量，face 子特征用零向量占位，person 子特征仍正常参与。这个处理不是动态切换到 OSNet-only，也不是说人脸缺失样本变成了有效的人脸检索样本；它只是让 Fusion 方法在 face 子特征缺失时仍能计算一个固定格式的融合向量。

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
