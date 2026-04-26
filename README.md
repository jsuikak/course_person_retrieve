# 人物检索（face + person）

本项目采用 `src/` 下的函数式接口，支持：

1. 图像库检索（query image -> image gallery）
2. 视频库检索（query image -> video gallery）
3. 应用级流程：建索引（存在则跳过）+ 检索一体化执行
4. `feature_mode=face/person` 两条链路：
   - `face`：MTCNN + ArcFace
   - `person`：YOLO 人体检测 + ResNet 特征
5. 离线 benchmark 测试：在 Celeb-reID 上评测 ArcFace-only、OSNet-only 与 ArcFace+OSNet 融合检索

核心实现：

- ArcFace 512 维特征
- MTCNN 人脸检测
- YOLO 人体检测（Ultralytics）
- NumPy 余弦相似度 TopK 匹配
- 检索输出包含：命中裁剪图 + 原图叠框标注图

说明：Celeb-reID benchmark 是离线评测流程，用于验证融合检索算法和记录指标，尚未接入 Web 前端。详细协议、公式、数据集结构和结果见：[融合检索以及 Celeb-reID 基准测试](docs/融合检索以及Celeb-reID基准测试.md)。

## 快速开始（推荐顺序）

### 0) 安装Git LFS

仓库里包含了一个大权重文件，拉取仓库前建议确保安装了Git LFS

包含两个步骤
1. 安装Git LFS本体
2. 至少运行一次 `git lfs install` 进行初始化。这一步会安装必要的钩子（hooks）到你的 Git 配置中。

#### 若想先跳过大文件下载

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/jsuikak/course_person_retrieve.git

此时大文件只是一个个文本指针。

后续可以手动拉取
`git lfs pull`

### 1) 安装依赖

```bash
uv sync
```

### 2) 初始化运行目录

```bash
uv run python scripts/create_runtime_folders.py --project-root .
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

### 3) CLI 运行检索(可选)

默认 ArcFace 权重：`./models/weights/arcface.pt`

1. 图像库检索（自动建索引或跳过）

```bash
uv run python -m src.app_retrieval \
  --query data_runtime/query/ikura.jpg \
  --gallery data_runtime/gallery/images/yoasobi_TFT \
  --topk 5 \
  --feature-mode face \
  --weights ./models/weights/arcface.pt \
  --device cpu
```

2. 视频库检索（自动写入 `indexes/video_index`）

```bash
uv run python -m src.app_retrieval \
  --query data_runtime/query/Tim1.png \
  --gallery data_runtime/gallery/videos/egodeath \
  --topk 5 \
  --feature-mode face \
  --sample-fps 1.0 \
  --weights ./models/weights/arcface.pt \
  --device cpu
```

3. Person-only 检索（YOLO 检测 + ResNet 特征）

```bash
uv run python -m src.app_retrieval \
  --query data_runtime/query/ikura.jpg \
  --gallery data_runtime/gallery/images/yoasobi_TFT \
  --topk 5 \
  --feature-mode person \
  --yolo-weights ./models/weights/yolo11n.pt \
  --yolo-conf 0.25 \
  --yolo-iou 0.7 \
  --yolo-max-det 100 \
  --weights ./models/weights/arcface.pt \
  --device cpu
```

### 4) 启动 Web 页面

```bash
uv run python -m uvicorn src.app.backend.app:app --reload --host 127.0.0.1 --port 8000
```

浏览器访问：`http://127.0.0.1:8000`

不使用 `uv` 时可执行：

```bash
python3 -m uvicorn src.app.backend.app:app --reload --host 127.0.0.1 --port 8000
```

说明：优先使用 `uv run python -m uvicorn ...`，可避免 `uv run uvicorn ...` 误调用系统 Python 导致依赖不一致。

停止服务：终端按 `Ctrl + C`

## WebUI 使用说明

Web 页面包含核心操作：`刷新文件列表`、`构建索引`、`检索任务`、`检索结果`。

### 1) 文件放置与列表刷新

**WebUI 不再上传文件。使用前先按运行时目录规范放置素材：**

1. 查询图片放入 `data_runtime/query/`。
2. 图像库放入 `data_runtime/gallery/images/<库名>/`。(下面存放图片文件, 如1.jpg, 2.jpg..., 200.jpg)
3. 视频库放入 `data_runtime/gallery/videos/<库名>/`。(下面存放视频文件, 如example.mp4)
4. 打开页面或点击 `刷新文件列表` 后，前端会从后端扫描这些目录并填充下拉框。

说明：视频下拉框列出含视频文件的库目录，不列出单个视频文件。

### 2) 构建索引

用途：提前构建并持久化索引，避免每次检索都临时建索引。

操作：

1. 在图库或视频库标签页选择库路径。
2. 选择 `特征模式`：
   - `face`：人脸检索链路（MTCNN + ArcFace）。
   - `person`：人体检索链路（YOLO + ResNet）。
3. `索引名（可选）` 留空时，默认使用图库目录名或文件名。
4. 当库是视频时，可调整 `采样 FPS`（值越大，抽帧越密，构建越慢）。
5. 点击 `构建索引`，成功后会提示 `total_items`。

结果落盘：

1. 图像库索引写入 `indexes/image_index/`。
2. 视频库索引写入 `indexes/video_index/`。
3. 文件命名为 `<index_name>_<feature_mode>_{features,meta,info}.*`。

### 3) 检索任务

该板块有两个标签页：

1. `图片检索图库`
2. `图片检索视频库`

#### 图片检索图库

1. 在 `查询图片` 下拉框选择 `data_runtime/query/` 下的图片。
2. 在 `图库路径` 下拉框选择 `data_runtime/gallery/images/` 下的图库目录。
3. 选择 `特征模式` 与（可选）`索引名`。
4. 设置 `Top-K`（返回前 K 条结果）。
5. 点击 `开始检索`。

说明：若对应索引不存在，后端会先自动构建索引，再执行检索。

#### 图片检索视频库

1. 在 `查询图片` 下拉框选择 `data_runtime/query/` 下的图片。
2. 在 `视频库路径` 下拉框选择 `data_runtime/gallery/videos/` 下的视频库目录。
3. 选择 `特征模式`、`Top-K`、`采样 FPS`，并可指定 `索引名`。
4. 点击 `开始视频检索`。

说明：若对应索引不存在，后端会先自动构建索引，再执行检索。

### 4) 检索结果

检索完成后，结果板块会展示：

1. `result summary`：`mode / library / index / results`。
2. `Query` 预览图。
3. 结果卡片列表（每条包含）：
   - `Rank` 和 `Score`
   - `Source`（命中来源文件）
   - `BBox`（命中框坐标）
   - `Frame`（视频命中帧号，图片一般为 `-1`）
   - `annotated`（原图叠框）
   - `crop`（命中区域裁剪图）

清理：

1. 点击 `清理检索输出` 可清空 `outputs/web/retrieval`。
2. 该操作不会删除 `indexes/` 下已构建的索引文件。

## 当前推荐入口

1. 应用级流程入口
   - `uv run python -m src.app_retrieval ...`
   - 代码：`src/app_retrieval.py`
2. 最简检索函数入口
   - `search_query_in_index(...)`
   - 代码：`src/retrieval.py`
3. Web API 入口
   - `src.app.backend.app:app`
   - 代码：`src/app/backend/app.py`
4. 离线 benchmark 入口
   - `uv run python scripts/benchmark_celebreid_light_fusion.py`
   - 代码：`scripts/benchmark_celebreid_light_fusion.py`
   - 文档：[融合检索以及 Celeb-reID 基准测试](docs/融合检索以及Celeb-reID基准测试.md)
   - 说明：该 benchmark 未接入前端，运行结果写入 `outputs/benchmarks/celebreid_fusion/`

## 输出说明

每次检索输出目录：

`outputs/retrieval/<query文件名>-<索引名>-<feature_mode>/`

包含：

1. `results.json`：TopK 结果、分数、来源路径、bbox、输出路径
2. `crops/`：命中框裁剪图
3. `annotated/`：原尺寸图叠框，框上标注 query 名称

索引文件位于：

1. 图像库：`indexes/image_index/<index_name>_{features,meta,info}.*`
2. 视频库：`indexes/video_index/<index_name>_{features,meta,info}.*`

索引命名已统一为（仅支持新命名）：

1. `face`：`<index_name>_face_{features,meta,info}.*`
2. `person`：`<index_name>_person_{features,meta,info}.*`

其中 `meta.csv` 包含 `x,y,w,h`、`source_name`、`frame_index`，可用于回溯命中位置。

## Score 定义

`results.json` 中每条结果的 `score` 定义一致，`face` / `person` 模式都使用余弦相似度：

```text
score = cos(g, q) = (g / ||g||2) · (q / ||q||2)
```

其中：

1. `g` 是 gallery 中某一条特征向量（人脸或人体）。
2. `q` 是 query 特征向量。
3. 先做 L2 归一化，再做点积。

解释：

1. 取值范围理论上为 `[-1, 1]`。
2. 分数越大越相似，`1` 表示方向完全一致，`0` 表示近似无关，负值表示方向相反。
3. `Score` 不是检测置信度（不是 YOLO/MTCNN 的检测分数），而是特征相似度分数。

当前实现细节（`src/retrieval.py`）：

1. `person` 模式：每个 gallery 人体特征与 query 人体特征直接计算余弦相似度。
2. `face` 模式：代码按“gallery 每一行对 query 特征集合取最大相似度”实现；当前 query 实际只提取 1 个特征，因此等价于单个余弦相似度。
3. TopK 结果按 `score` 从高到低排序返回。

## 文档索引

1. `docs/检索流程接口说明.md`
2. `docs/最简检索接口说明.md`
3. `docs/运行时文件夹结构说明.md`
4. `docs/人脸特征提取流程接口说明.md`
5. `docs/统一特征提取接口说明.md`
6. `docs/统一特征匹配接口说明.md`
7. `docs/MTCNN检测接口说明.md`
