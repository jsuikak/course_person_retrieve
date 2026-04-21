# 人脸检索系统实现文档（ArcFace + MTCNN）

## 1. 项目目标

本项目实现一个统一的人脸检索算法框架，满足课程要求：

1. 基于图像检索（图库人脸搜索）
2. 基于图像检索视频（视频库人脸搜索）

并支持工程化运行：索引构建、检索查询、评估实验、视频可视化标注。

---

## 2. 总体方案

### 2.1 技术路线

- 人脸特征模型：ArcFace（512 维 embedding）
- 相似度度量：余弦相似度（L2 归一化后矩阵乘）
- 图像检索：NumPy 精确检索（Top-K 排序 + 阈值匹配）
- 视频检索：`采样帧 -> 人脸检测 -> 轨迹聚合 -> 视频级打分`
- 视频输入增强：先用 MTCNN 检测并裁剪人脸，再送 ArcFace
- 结果可视化：在视频帧上绘制人脸框、姓名（身份）与置信度

### 2.2 核心打分策略

- 单帧得分：`cos(query_feat, frame_feat)`
- 轨迹得分：`mean(top3(frame_scores))`
- 视频得分：`max(track_scores)`

---

## 3. 代码结构

主要模块如下：

- `retrieval/feature_extractor.py`
  - ArcFace 提特征
  - `extract(image_path)` / `extract_batch(paths)`
  - 支持 `flip test`、质量过滤、人脸检测后端切换

- `retrieval/mtcnn_detector.py`
  - MTCNN 检测器封装（一次初始化，多帧复用）
  - 输出 `BBox(x, y, w, h)` 列表

- `retrieval/image_index.py`
  - 图像索引构建：输出 `features.npy + meta.csv + index_info.json`
  - 图像检索：Top-K 结果、分数、是否匹配阈值

- `retrieval/video_index.py`
  - 视频索引构建（采样 + IoU 轨迹关联）
  - 视频检索（轨迹聚合 + 视频级最大分）

- `retrieval/video_annotator.py`
  - 对输入视频逐帧检测与检索
  - 绘制框、姓名（identity/image_id）、置信度
  - 输出标注视频

- `retrieval/evaluation.py` + `retrieval/metrics.py`
  - Recall@1 / Recall@5 / mAP
  - 固定四组消融实验

- `retrieval_cli.py`
  - 统一命令行入口：
    - `build-image-index`
    - `search-image`
    - `build-video-index`
    - `search-video`
    - `annotate-video`
    - `evaluate`

---

## 4. 数据接口设计

### 4.1 图像 Manifest（CSV）

必需字段：

- `image_path`

可选字段：

- `image_id`
- `identity`（用于评估和可视化姓名）
- `x,y,w,h`（已有人脸框时可直接使用）

### 4.2 视频 Manifest（CSV）

必需字段：

- `video_path`

可选字段：

- `video_id`
- `identity`

---

## 5. 关键流程说明

### 5.1 图像检索流程

1. 从图库 manifest 读取图像列表
2. 对每张图提取 ArcFace 特征（可选 flip/质量过滤）,建立索引
3. 保存索引文件（`features.npy`, `meta.csv`）
4. 查询图提特征，与图库特征做余弦相似度
5. 返回 Top-K 排序结果与匹配标记

### 5.2 视频检索流程

1. 从视频 manifest 读取视频
2. 按 `sample_fps` 采样帧
3. 每帧使用 MTCNN 做人脸检测并裁剪
4. 对每个人脸提特征
5. 用 IoU 将跨帧人脸关联为轨迹
6. 轨迹分数 `mean(top3)`，视频分数取 `max`
7. 输出视频检索排名

### 5.3 视频可视化流程

1. 输入视频逐帧读取
2. 使用 MTCNN 检测人脸框
3. 每个框提特征，在图库索引中检索 Top-1
4. 绘制框与文本标签：`name score`
5. 写入输出视频文件

---

## 6. CLI 复现命令（YOASOBI 数据）

数据集路径：`data/yoasobi_TFT`

### 6.1 构建图像索引

```bash
uv run python retrieval_cli.py build-image-index \
  --manifest outputs/yoasobi_tft/manifests/gallery_images.csv \
  --index-dir outputs/yoasobi_tft/indexes/image_index \
  --weights ./models/weights/arcface.pt \
  --device cpu \
  --flip-test --no-detect-face
```

### 6.2 图像检索

```bash
uv run python retrieval_cli.py search-image \
  --query data/yoasobi_TFT/ayase/0001.jpg \
  --index-dir outputs/yoasobi_tft/indexes/image_index \
  --weights ./models/weights/arcface.pt --device cpu
```

### 6.3 构建视频索引（MTCNN）

```bash
uv run python retrieval_cli.py build-video-index \
  --manifest outputs/yoasobi_tft/manifests/videos.csv \
  --index-dir outputs/yoasobi_tft/indexes/video_index_short_mtcnn \
  --weights ./models/weights/arcface.pt \
  --device cpu \
  --flip-test --detect-face \
  --sample-fps 1.0 --iou-threshold 0.3
```

### 6.4 视频检索（MTCNN 索引）

```bash
uv run python retrieval_cli.py search-video \
  --query data/yoasobi_TFT/ikura/0001.jpg \
  --index-dir outputs/yoasobi_tft/indexes/video_index_short_mtcnn \
  --weights ./models/weights/arcface.pt \
  --device cpu
```

### 6.5 视频标注输出（框 + 姓名 + 置信度）

```bash
uv run python retrieval_cli.py annotate-video \
  --video data/yoasobi_TFT/short.mp4 \
  --output-video outputs/yoasobi_tft/results/short_annotated.mp4 \
  --index-dir outputs/yoasobi_tft/indexes/image_index \
  --weights ./models/weights/arcface.pt \
  --device cpu \
  --detect-face \
  --threshold 0.3
```

---

## 7. 当前测试结果（已完成）

### 7.1 数据规模

- 图库图像：364（`ayase + ikura`）
- 查询图像：20（每类各 10）
- 视频：1（`short.mp4`）

### 7.2 索引构建结果

- 图像索引：`total_records=364, indexed_records=364, skipped_records=0`
- 视频索引（MTCNN）：`total_videos=1, indexed_videos=1, indexed_frame_features=6`

### 7.3 视频标注结果

- `total_frames=301`
- `faces_detected=132`
- `matched_faces=131`
- 输出文件：`outputs/yoasobi_tft/results/short_annotated.mp4`

### 7.4 查询输出文件

- 图像检索：
  - `outputs/yoasobi_tft/results/search_image_ayase.json`
  - `outputs/yoasobi_tft/results/search_image_ikura.json`
- 视频检索：
  - `outputs/yoasobi_tft/results/search_video_ayase_short_mtcnn.json`
  - `outputs/yoasobi_tft/results/search_video_ikura_short_mtcnn.json`

---

## 8. 课程答辩可讲的设计点

1. 将视频检索统一到图像特征检索框架，但保留时序聚合（轨迹 top3 均值）。
2. 使用 MTCNN 解决大尺寸视频的人脸定位问题，再进入 ArcFace 表征空间。
3. 工程拆分明确：检测、特征、索引、检索、评估、可视化分层解耦。
4. 指标可量化（Recall@K、mAP），结果可解释（可视化视频）。

---

## 9. 已知限制与后续可优化

- 当前图库仅两类身份，视频库只有一个视频，评估场景较小。
- 当前视频轨迹使用 IoU 关联，复杂遮挡场景下稳定性有限。
- 可进一步加入：
  - 多视频库与更大身份规模测试
  - 阈值标定（FAR/FRR）
  - 更强跟踪器（如 ByteTrack）做轨迹关联

---

## 10. 版本说明

本实现对应当前仓库中的模块化版本，已包含 MTCNN 检测与视频标注能力。
