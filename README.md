# ArcFace 人脸检索

详细实现文档：`docs/IMPLEMENTATION.md`

本项目实现了统一的人脸检索框架，支持：

1. 图像库检索（image -> image）
2. 视频库检索（image -> video）

核心策略：

- ArcFace 512 维特征
- 余弦相似度 + NumPy 精确检索
- 可选质量过滤（模糊度阈值、最小人脸尺寸）
- 可选 flip test（原图 + 水平翻转）
- 视频端采用 `1 fps 采样 + IoU 轨迹关联 + mean(top3) 轨迹打分 + max 视频聚合`

## 环境安装

```sh
uv sync
```
cpu就可以运行

## 目录结构

- `retrieval/feature_extractor.py`：ArcFace 特征提取层
- `retrieval/image_index.py`：图像索引构建与检索
- `retrieval/video_index.py`：视频索引构建与检索
- `retrieval/video_annotator.py`：视频逐帧检索可视化（框+姓名+置信度）
- `retrieval/evaluation.py`：Recall@K / mAP 评估与消融实验
- `retrieval_cli.py`：统一 CLI 入口

## Manifest 格式

### 图像 manifest（gallery 或 query）

CSV 至少包含：`image_path`

可选列：

- `image_id`
- `identity`（评估 mAP/Recall 时建议提供）
- `x,y,w,h`（已知人脸框，可加速并提升稳定性）

示例：

```csv
image_path,image_id,identity,x,y,w,h
/path/to/a.jpg,a,p001,34,42,98,98
/path/to/b.jpg,b,p002,,,,
```

### 视频 manifest

CSV 至少包含：`video_path`

可选列：

- `video_id`
- `identity`（用于视频检索评估）

示例：

```csv
video_path,video_id,identity
/path/to/v1.mp4,v1,p001
/path/to/v2.mp4,v2,p002
```

## CLI 用法

默认权重路径：`./models/weights/arcface.pt`

### 1) 构建图像索引

```bash
python retrieval_cli.py build-image-index \
  --manifest ./data/gallery_manifest.csv \
  --index-dir ./outputs/image_index \
  --weights ./models/weights/arcface.pt \
  --device cpu \
  --flip-test \
  --blur-threshold 100 \
  --min-face-size 32
```

输出：`features.npy`、`meta.csv`、`index_info.json`

### 2) 图像检索

```bash
python retrieval_cli.py search-image \
  --query ./data/query.jpg \
  --index-dir ./outputs/image_index \
  --weights ./models/weights/arcface.pt \
  --topk 5 \
  --threshold 0.3
```

### 3) 构建视频索引

```bash
python retrieval_cli.py build-video-index \
  --manifest ./data/video_manifest.csv \
  --index-dir ./outputs/video_index \
  --weights ./models/weights/arcface.pt \
  --detect-face \
  --sample-fps 1.0 \
  --iou-threshold 0.3
```

### 4) 视频检索

```bash
python retrieval_cli.py search-video \
  --query ./data/query.jpg \
  --index-dir ./outputs/video_index \
  --weights ./models/weights/arcface.pt \
  --detect-face \
  --topk 5 \
  --threshold 0.3
```

### 5) 视频标注输出（MTCNN 检测 + 检索姓名与置信度）

```bash
python retrieval_cli.py annotate-video \
  --video ./data/yoasobi_TFT/short.mp4 \
  --output-video ./outputs/yoasobi_tft/results/short_annotated.mp4 \
  --index-dir ./outputs/yoasobi_tft/indexes/image_index \
  --weights ./models/weights/arcface.pt \
  --device cpu \
  --detect-face \
  --threshold 0.3
```

### 6) 四组固定消融实验

```bash
python retrieval_cli.py evaluate \
  --gallery-manifest ./data/gallery_manifest.csv \
  --query-manifest ./data/query_manifest.csv \
  --video-manifest ./data/video_manifest.csv \
  --output-dir ./outputs/eval \
  --weights ./models/weights/arcface.pt \
  --device cpu \
  --detect-face
```

输出：`./outputs/eval/ablation_results.csv`

实验固定为：

1. Baseline（单特征）
2. +Quality Filter
3. +Flip Test
4. +Video Track Aggregation
