# 人脸检索（当前项目结构）

本项目当前采用 `src/` 下的函数式接口，支持：

1. 图像库检索（query image -> image gallery）
2. 视频库检索（query image -> video gallery）
3. 应用级流程：建索引（存在则跳过）+ 检索一体化执行

核心实现：

- ArcFace 512 维特征
- MTCNN 人脸检测
- NumPy 余弦相似度 TopK 匹配
- 检索输出包含：命中裁剪图 + 原图叠框标注图

## 环境安装

```bash
uv sync
```

## 当前推荐入口

1. 应用级流程入口（推荐）
   - `python3 -m src.app_retrieval ...`
   - 代码：`src/app_retrieval.py`
2. 检索函数入口
   - `search_query_in_index(...)`
   - 代码：`src/retrieval.py`

## 运行目录规范

先创建运行目录：

```bash
python3 scripts/create_runtime_folders.py --project-root .
```

规范详见：`docs/运行时文件夹结构说明.md`

最小目录结构：

```text
data_runtime/
  query/
  gallery/
    images/
    videos/
indexes/
  image_index/
  video_index/
outputs/
  retrieval/
  annotate/
```

## 快速运行

默认权重：`./models/weights/arcface.pt`

### 1) 图像库检索（自动建索引或跳过）

```bash
python3 -m src.app_retrieval \
  --query data_runtime/query/ikura.jpg \
  --gallery data_runtime/gallery/images/yoasobi_TFT \
  --topk 5 \
  --weights ./models/weights/arcface.pt \
  --device cpu
```

### 2) 视频库检索（自动写入 `indexes/video_index`）

```bash
python3 -m src.app_retrieval \
  --query data_runtime/query/Tim1.png \
  --gallery data_runtime/gallery/videos/egodeath \
  --topk 5 \
  --weights ./models/weights/arcface.pt \
  --device cpu
```

## 输出说明

每次检索输出目录：

`outputs/retrieval/<query文件名>-<索引名>/`

包含：

1. `results.json`：TopK 结果、分数、来源路径、bbox、输出路径
2. `crops/`：命中框裁剪图
3. `annotated/`：原尺寸图叠框，框上标注 query 名称

索引文件位于：

1. 图像库：`indexes/image_index/<index_name>_{features,meta,info}.*`
2. 视频库：`indexes/video_index/<index_name>_{features,meta,info}.*`

其中 `meta.csv` 包含 `x,y,w,h`、`source_name`、`frame_index`、`face_index`，可用于回溯命中位置。

## 文档索引

1. `docs/检索流程接口说明.md`
2. `docs/最简检索接口说明.md`
3. `docs/运行时文件夹结构说明.md`
4. `docs/人脸特征提取流程接口说明.md`
5. `docs/统一特征提取接口说明.md`
6. `docs/统一特征匹配接口说明.md`
7. `docs/MTCNN检测接口说明.md`
