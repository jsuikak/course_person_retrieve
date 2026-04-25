# Celeb-reID-light 融合检索基准测试

## 数据集

- 数据来源：Kaggle 下载 Celeb-reID-light；官方数据说明参考 https://github.com/Huang-3/Celeb-reID
- 默认本地路径：`data/Celeb-reID-light/`
- 内容：裁剪后的名人/人物行人图像，用于长期换装 Person ReID。
- 官方 split：train 490 ID / 9021 图，query 100 ID / 887 图，gallery 100 ID / 934 图。

脚本固定使用以下目录格式：

```text
data/Celeb-reID-light/
|-- train/
|-- query/
`-- gallery/
```

## 算法

本实验不训练新模型，只比较预训练特征的离线检索效果。

1. ArcFace-only：对裁剪行人图像先做人脸检测，取最大人脸输入 ArcFace；query 未检测到人脸时，该 query 在 face-only 指标中按失败计。
2. OSNet-only：不再跑 YOLO，直接把 Celeb-reID-light 的裁剪行人图像输入 OSNet，得到整体行人特征。
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

当人脸检测失败时，face 分支用零向量，融合结果自然退化为 OSNet 主导。

## 运行命令

先将 Kaggle 下载的数据解压到 `data/Celeb-reID-light/`，再运行 smoke test：

```bash
uv run python scripts/benchmark_celebreid_light_fusion.py \
  --limit-query 20 \
  --limit-gallery 100
```

完整评测：

```bash
uv run python scripts/benchmark_celebreid_light_fusion.py
```

脚本会输出：

- `outputs/benchmarks/celebreid_light_fusion/dataset_summary.json`
- `outputs/benchmarks/celebreid_light_fusion/config.json`
- `outputs/benchmarks/celebreid_light_fusion/features/*.npy`
- `outputs/benchmarks/celebreid_light_fusion/metadata/*.csv`
- `outputs/benchmarks/celebreid_light_fusion/metrics.json`
- `outputs/benchmarks/celebreid_light_fusion/metrics.csv`

脚本完整运行后会用实际数据集大小、目录结构、耗时和指标覆盖更新本文档。

## 结果

当前工作区还没有 `data/Celeb-reID-light/`，因此尚未生成实际 benchmark 指标。

| 方法 | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) | mAP (%) |
|---|---:|---:|---:|---:|
| ArcFace-only | 待运行 | 待运行 | 待运行 | 待运行 |
| OSNet-only | 待运行 | 待运行 | 待运行 | 待运行 |
| ArcFace+OSNet fusion | 待运行 | 待运行 | 待运行 | 待运行 |

指标口径：Rank-K 表示 Top-K 内是否至少命中同身份；mAP 是所有 query 的 AP 平均。数值单位均为百分比。
