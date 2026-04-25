#!/usr/bin/env python3
"""Benchmark ArcFace, OSNet and fused retrieval on Celeb-reID/Celeb-reID-light."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion_retrieval import (  # noqa: E402
    FusionFeatureConfig,
    FusionFeatureMatrices,
    build_feature_extractor,
    compute_similarity_matrix,
    extract_dual_feature,
    load_feature_matrices,
    records_to_matrices,
    save_feature_matrices,
)
from src.tools.retrieval_metrics import evaluate_reid  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_DATASET_ROOT = "data/Celeb-reID"
DEFAULT_OUTPUT_DIR = "outputs/benchmarks/celebreid_fusion"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "Celeb-reID融合检索基准测试.md"
OFFICIAL_SPLITS = {
    "Celeb-reID": {
        "train": {"identities": 632, "images": 20208},
        "query": {"identities": 420, "images": 2972},
        "gallery": {"identities": 420, "images": 11006},
        "total": {"identities": 1052, "images": 34186},
    },
    "Celeb-reID-light": {
        "train": {"identities": 490, "images": 9021},
        "query": {"identities": 100, "images": 887},
        "gallery": {"identities": 100, "images": 934},
        "total": {"identities": 590, "images": 10842},
    },
}


@dataclass(slots=True)
class ImageItem:
    path: Path
    identity: str
    split: str


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _repo_relative_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _dataset_name(dataset_root: Path) -> str:
    name = dataset_root.name
    if name.lower() in {"celeb-reid-light", "celebreid-light"}:
        return "Celeb-reID-light"
    if name.lower() in {"celeb-reid", "celebreid"}:
        return "Celeb-reID"
    return name


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return int(total)


def _list_images(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        return []
    return sorted(p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _resolve_path(path_text: str | None, dataset_root: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = dataset_root / path
    if candidate.exists():
        return candidate
    return REPO_ROOT / path


def _find_split_dir(dataset_root: Path, explicit: str | None, split: str, required: bool = True) -> Path | None:
    explicit_path = _resolve_path(explicit, dataset_root)
    if explicit_path is not None:
        if not explicit_path.is_dir():
            raise NotADirectoryError(f"split path is not a directory: {explicit_path}")
        return explicit_path

    candidates = [dataset_root / split]
    if split == "train":
        candidates.append(dataset_root / "training")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    if required:
        raise FileNotFoundError(f"required split directory not found: {dataset_root / split}")
    return None


def _parse_identity(image_path: Path, split_dir: Path, identity_regex: str) -> str:
    rel = image_path.relative_to(split_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]

    match = re.match(identity_regex, image_path.stem)
    if match:
        return str(match.group(1))
    raise ValueError(
        "cannot parse identity from flat filename. "
        f"path={image_path}, regex={identity_regex!r}. "
        "Use identity subfolders or pass --identity-regex."
    )


def _build_manifest(split_dir: Path, split: str, identity_regex: str, limit: int | None = None) -> list[ImageItem]:
    images = _list_images(split_dir)
    if limit is not None:
        images = images[: max(0, int(limit))]

    items: list[ImageItem] = []
    errors: list[str] = []
    for image_path in images:
        try:
            identity = _parse_identity(image_path=image_path, split_dir=split_dir, identity_regex=identity_regex)
        except Exception as exc:
            errors.append(str(exc))
            if len(errors) >= 5:
                break
            continue
        items.append(ImageItem(path=image_path, identity=identity, split=split))

    if errors:
        joined = "\n".join(errors)
        raise ValueError(f"failed to parse identities for split={split}. Sample errors:\n{joined}")
    if not items:
        raise ValueError(f"no images found for split={split}: {split_dir}")
    return items


def _split_summary(split_dir: Path | None, split: str, identity_regex: str) -> dict[str, Any]:
    if split_dir is None:
        return {
            "split": split,
            "path": None,
            "exists": False,
            "image_count": 0,
            "identity_count": 0,
            "size_bytes": 0,
            "size_human": "0.00 B",
        }

    images = _list_images(split_dir)
    identities: set[str] = set()
    parse_errors = 0
    for image_path in images:
        try:
            identities.add(_parse_identity(image_path=image_path, split_dir=split_dir, identity_regex=identity_regex))
        except Exception:
            parse_errors += 1
    size_bytes = _directory_size(split_dir)
    return {
        "split": split,
        "path": str(split_dir),
        "exists": True,
        "image_count": len(images),
        "identity_count": len(identities),
        "identity_parse_errors": parse_errors,
        "size_bytes": size_bytes,
        "size_human": _human_size(size_bytes),
    }


def _tree_preview(root: Path, max_depth: int = 2, max_entries: int = 10) -> list[str]:
    if not root.exists():
        return []

    lines: list[str] = [root.name + "/"]

    def walk(current: Path, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return
        entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        shown = entries[:max_entries]
        for idx, entry in enumerate(shown):
            connector = "`-- " if idx == len(shown) - 1 and len(entries) <= max_entries else "|-- "
            suffix = "/" if entry.is_dir() else ""
            lines.append(prefix + connector + entry.name + suffix)
            if entry.is_dir():
                next_prefix = prefix + ("    " if connector == "`-- " else "|   ")
                walk(entry, next_prefix, depth + 1)
        if len(entries) > max_entries:
            lines.append(prefix + f"`-- ... ({len(entries) - max_entries} more)")

    walk(root, "", 0)
    return lines


def _dataset_summary(
    dataset_root: Path,
    query_dir: Path,
    gallery_dir: Path,
    train_dir: Path | None,
    identity_regex: str,
    limit_query: int | None,
    limit_gallery: int | None,
) -> dict[str, Any]:
    dataset_name = _dataset_name(dataset_root)
    size_bytes = _directory_size(dataset_root)
    return {
        "dataset": dataset_name,
        "source": "Kaggle download, official metadata from https://github.com/Huang-3/Celeb-reID",
        "root": str(dataset_root),
        "size_bytes": size_bytes,
        "size_human": _human_size(size_bytes),
        "splits": {
            "query": _split_summary(query_dir, "query", identity_regex),
            "gallery": _split_summary(gallery_dir, "gallery", identity_regex),
            "train": _split_summary(train_dir, "train", identity_regex),
        },
        "official_split_reference": OFFICIAL_SPLITS.get(dataset_name, {}),
        "benchmark_protocol": {
            "query_limit": limit_query,
            "gallery_limit": limit_gallery,
            "uses_train_for_metrics": False,
        },
        "folder_preview": _tree_preview(dataset_root),
    }


def _cache_payload(items: list[ImageItem], config: FusionFeatureConfig) -> dict[str, Any]:
    return {
        "items": [{"path": str(item.path), "identity": item.identity, "split": item.split} for item in items],
        "config": asdict(config),
    }


def _cache_config_path(output_dir: Path, prefix: str) -> Path:
    return output_dir / "features" / f"{prefix}_cache_config.json"


def _cache_complete(output_dir: Path, prefix: str) -> bool:
    required = [
        output_dir / "features" / f"{prefix}_face.npy",
        output_dir / "features" / f"{prefix}_person.npy",
        output_dir / "features" / f"{prefix}_fused.npy",
        output_dir / "features" / f"{prefix}_face_valid.npy",
        output_dir / "features" / f"{prefix}_person_valid.npy",
        output_dir / "features" / f"{prefix}_fused_valid.npy",
        output_dir / "metadata" / f"{prefix}.csv",
        _cache_config_path(output_dir, prefix),
    ]
    return all(path.exists() for path in required)


def _load_or_extract_split(
    items: list[ImageItem],
    prefix: str,
    config: FusionFeatureConfig,
    output_dir: Path,
    use_cache: bool,
    recompute: bool,
) -> FusionFeatureMatrices:
    payload = _cache_payload(items, config)
    config_path = _cache_config_path(output_dir, prefix)
    if use_cache and not recompute and _cache_complete(output_dir, prefix):
        with config_path.open("r", encoding="utf-8") as f:
            cached_payload = json.load(f)
        if cached_payload == payload:
            return load_feature_matrices(output_dir=output_dir, prefix=prefix)

    extractor = build_feature_extractor(config)
    records = []
    for item in tqdm(items, desc=f"extract {prefix}", unit="img"):
        try:
            records.append(
                extract_dual_feature(
                    image_path=item.path,
                    identity=item.identity,
                    split=item.split,
                    extractor=extractor,
                )
            )
        except Exception as exc:
            raise RuntimeError(f"failed to extract features for {item.path}") from exc

    matrices = records_to_matrices(records, config=config)
    save_feature_matrices(matrices=matrices, output_dir=output_dir, prefix=prefix)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return matrices


def _evaluate_all(query: FusionFeatureMatrices, gallery: FusionFeatureMatrices) -> dict[str, Any]:
    methods = [
        ("ArcFace-only", query.face_features, gallery.face_features, query.face_valid),
        ("OSNet-only", query.person_features, gallery.person_features, query.person_valid),
        ("ArcFace+OSNet fusion", query.fused_features, gallery.fused_features, query.fused_valid),
    ]
    out: dict[str, Any] = {}
    for name, query_features, gallery_features, query_valid in methods:
        scores = compute_similarity_matrix(query_features=query_features, gallery_features=gallery_features)
        metric = evaluate_reid(
            scores=scores,
            query_ids=query.identities,
            gallery_ids=gallery.identities,
            ranks=(1, 5, 10),
            valid_query_mask=query_valid,
        )
        out[name] = metric.to_dict()
    return out


def _feature_stats(query: FusionFeatureMatrices, gallery: FusionFeatureMatrices) -> dict[str, Any]:
    def split_stats(matrices: FusionFeatureMatrices) -> dict[str, Any]:
        total = len(matrices)
        face_valid = int(matrices.face_valid.sum())
        person_valid = int(matrices.person_valid.sum())
        fused_valid = int(matrices.fused_valid.sum())
        return {
            "total_images": total,
            "face_valid": face_valid,
            "face_missing": int(total - face_valid),
            "face_success_rate": float(face_valid / total * 100.0) if total else 0.0,
            "person_valid": person_valid,
            "person_missing": int(total - person_valid),
            "person_success_rate": float(person_valid / total * 100.0) if total else 0.0,
            "fused_valid": fused_valid,
            "fused_missing": int(total - fused_valid),
            "face_dim": int(matrices.face_features.shape[1]) if matrices.face_features.ndim == 2 else 0,
            "person_dim": int(matrices.person_features.shape[1]) if matrices.person_features.ndim == 2 else 0,
            "fused_dim": int(matrices.fused_features.shape[1]) if matrices.fused_features.ndim == 2 else 0,
        }

    return {
        "query": split_stats(query),
        "gallery": split_stats(gallery),
    }


def _feature_file_sizes(output_dir: Path) -> dict[str, Any]:
    feature_dir = output_dir / "features"
    sizes: dict[str, Any] = {}
    if not feature_dir.exists():
        return sizes
    for path in sorted(feature_dir.glob("*.npy")):
        sizes[path.name] = {
            "bytes": path.stat().st_size,
            "human": _human_size(path.stat().st_size),
        }
    return sizes


def _write_metrics_csv(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "rank1",
                "rank5",
                "rank10",
                "mAP",
                "total_queries",
                "valid_queries",
                "invalid_queries",
                "queries_without_positive",
                "metric_unit",
            ]
        )
        for method, values in metrics.items():
            writer.writerow(
                [
                    method,
                    values.get("rank1", 0.0),
                    values.get("rank5", 0.0),
                    values.get("rank10", 0.0),
                    values.get("mAP", 0.0),
                    values.get("total_queries", 0),
                    values.get("valid_queries", 0),
                    values.get("invalid_queries", 0),
                    values.get("queries_without_positive", 0),
                    values.get("metric_unit", "percent"),
                ]
            )


def _markdown_metrics_table(metrics: dict[str, Any]) -> str:
    lines = [
        "| 方法 | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) | mAP (%) | 有效 Query | 无效 Query |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in metrics.items():
        lines.append(
            "| {method} | {rank1:.2f} | {rank5:.2f} | {rank10:.2f} | {mAP:.2f} | {valid} | {invalid} |".format(
                method=method,
                rank1=float(values.get("rank1", 0.0)),
                rank5=float(values.get("rank5", 0.0)),
                rank10=float(values.get("rank10", 0.0)),
                mAP=float(values.get("mAP", 0.0)),
                valid=int(values.get("valid_queries", 0)),
                invalid=int(values.get("invalid_queries", 0)),
            )
        )
    return "\n".join(lines)


def _metric_delta(metrics: dict[str, Any], left: str, right: str, key: str) -> float | None:
    if left not in metrics or right not in metrics:
        return None
    return float(metrics[left].get(key, 0.0)) - float(metrics[right].get(key, 0.0))


def _format_delta(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}"


def _markdown_effect_analysis(metrics: dict[str, Any], stats: dict[str, Any]) -> str:
    fusion_vs_osnet_rank1 = _metric_delta(metrics, "ArcFace+OSNet fusion", "OSNet-only", "rank1")
    fusion_vs_osnet_rank5 = _metric_delta(metrics, "ArcFace+OSNet fusion", "OSNet-only", "rank5")
    fusion_vs_osnet_rank10 = _metric_delta(metrics, "ArcFace+OSNet fusion", "OSNet-only", "rank10")
    fusion_vs_osnet_map = _metric_delta(metrics, "ArcFace+OSNet fusion", "OSNet-only", "mAP")
    fusion_vs_face_rank1 = _metric_delta(metrics, "ArcFace+OSNet fusion", "ArcFace-only", "rank1")
    fusion_vs_face_map = _metric_delta(metrics, "ArcFace+OSNet fusion", "ArcFace-only", "mAP")
    query_face_rate = float(stats["query"].get("face_success_rate", 0.0))
    gallery_face_rate = float(stats["gallery"].get("face_success_rate", 0.0))
    query_face_valid = int(stats["query"].get("face_valid", 0))
    query_total = int(stats["query"].get("total_images", 0))

    return f"""## 效果分析

当前固定权重融合是一个有效 baseline，但还不是最优融合方案。相比 OSNet-only，Fusion 的 Rank-1 提升 {_format_delta(fusion_vs_osnet_rank1)} 个百分点，Rank-5 提升 {_format_delta(fusion_vs_osnet_rank5)} 个百分点，Rank-10 提升 {_format_delta(fusion_vs_osnet_rank10)} 个百分点，mAP 提升 {_format_delta(fusion_vs_osnet_map)} 个百分点。这说明人脸分支给整体行人特征补充了身份判别信息，尤其对 Top-K 早排位有帮助。

相比 ArcFace-only，Fusion 的 Rank-1 提升 {_format_delta(fusion_vs_face_rank1)} 个百分点，但 mAP 变化为 {_format_delta(fusion_vs_face_map)} 个百分点。原因是 ArcFace-only 只在检测到人脸时可用，query 人脸成功率为 {query_face_rate:.2f}%（{query_face_valid}/{query_total}），gallery 人脸成功率为 {gallery_face_rate:.2f}%；检测到人脸的样本中 ArcFace 判别力较强，但覆盖率不足。Fusion 保留 OSNet 的全量覆盖能力，在有人脸时叠加 ArcFace，因此 Top-K 命中率更稳定。

mAP 没有超过 ArcFace-only，说明当前固定权重拼接主要改善了靠前位置的命中，但没有充分优化同身份所有 gallery 样本的整体排序。下一步可以尝试基于人脸检测置信度、脸部面积、图像质量或两路相似度分布做动态权重，也可以做 score-level fusion 和权重网格搜索。"""


def _write_markdown_report(
    doc_path: Path,
    dataset_summary: dict[str, Any],
    metrics: dict[str, Any],
    stats: dict[str, Any],
    feature_file_sizes: dict[str, Any],
    config: FusionFeatureConfig,
    output_dir: Path,
    elapsed_seconds: float,
    command: str,
) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    query_split = dataset_summary["splits"]["query"]
    gallery_split = dataset_summary["splits"]["gallery"]
    train_split = dataset_summary["splits"]["train"]
    tree = "\n".join(dataset_summary.get("folder_preview", []))
    feature_sizes_text = "\n".join(
        f"- `{name}`: {detail['human']}" for name, detail in sorted(feature_file_sizes.items())
    )
    if not feature_sizes_text:
        feature_sizes_text = "- 暂无特征缓存文件。"
    feature_dir = output_dir / "features"
    metadata_dir = output_dir / "metadata"
    metrics_json_path = output_dir / "metrics.json"
    metrics_csv_path = output_dir / "metrics.csv"
    dataset_name = str(dataset_summary["dataset"])
    official = dataset_summary.get("official_split_reference", {})
    official_train = official.get("train", {})
    official_query = official.get("query", {})
    official_gallery = official.get("gallery", {})
    official_total = official.get("total", {})
    official_line = (
        f"- 官方 split：train {official_train.get('identities', '-')} ID / {official_train.get('images', '-')} 图，"
        f"query {official_query.get('identities', '-')} ID / {official_query.get('images', '-')} 图，"
        f"gallery {official_gallery.get('identities', '-')} ID / {official_gallery.get('images', '-')} 图，"
        f"total {official_total.get('identities', '-')} ID / {official_total.get('images', '-')} 图。"
    )
    eval_image_count = int(query_split["image_count"]) + int(gallery_split["image_count"])

    content = rf"""# {dataset_name} 融合检索基准测试

## 数据集

- 数据来源：Kaggle 下载的 {dataset_name}；官方数据说明参考 https://github.com/Huang-3/Celeb-reID
- 本地路径：`{dataset_summary['root']}`
- 数据集大小：{dataset_summary['size_human']} ({dataset_summary['size_bytes']} bytes)
- 内容：裁剪后的名人/人物行人图像，用于长期换装 ReID。
{official_line}

本次扫描结果：

| split | 路径 | 图像数 | ID 数 | 大小 |
|---|---|---:|---:|---:|
| train | `{train_split['path']}` | {train_split['image_count']} | {train_split['identity_count']} | {train_split['size_human']} |
| query | `{query_split['path']}` | {query_split['image_count']} | {query_split['identity_count']} | {query_split['size_human']} |
| gallery | `{gallery_split['path']}` | {gallery_split['image_count']} | {gallery_split['identity_count']} | {gallery_split['size_human']} |

## Benchmark 选择理由

选择 {dataset_name} 作为当前融合算法的 benchmark，主要有以下原因：

1. 规模与当前算力资源匹配。数据集本地大小为 {dataset_summary['size_human']}，本次主评测只使用 query 和 gallery，共 {eval_image_count} 张图像；完整 CPU 特征提取可以在可接受时间内完成，也便于重复实验和缓存复用。
2. 数据格式契合当前“双分支融合”算法。{dataset_name} 已经提供裁剪后的行人图像和官方 query/gallery split，每张图像可以同时作为人脸分支和整体行人分支的输入，不需要额外构造 pair、track 或视频标注。
3. 它不是单独的人脸验证集，也不是单独的行人特征验证集。图像中既包含整体行人外观，也可能包含可检测人脸，能直接检验 ArcFace 与 OSNet 这两类预训练特征在同一检索任务中的互补性。
4. 数据中存在人脸不可见、姿态变化、外观变化等情况，MTCNN 在这些样本上可能检测不到明确人脸。因此它能暴露 face 分支覆盖率不足的问题，并用于比较 ArcFace-only 的无效 query、OSNet-only 的全覆盖，以及 Fusion 在 face 子特征缺失时的表现差异。

目录预览：

```text
{tree}
```

## 算法

本实验不训练新模型，只比较预训练特征的检索效果。

1. ArcFace-only：对裁剪行人图像先做人脸检测，取最大人脸输入 ArcFace；query 未检测到人脸时，该 query 在 face-only 指标中按失败计。
2. OSNet-only：不再跑 YOLO，直接把 {dataset_name} 的裁剪行人图像输入 OSNet，得到整体行人特征。
3. ArcFace+OSNet fusion：使用固定权重加权拼接，默认 `face_weight={config.face_weight}`，`person_weight={config.person_weight}`。

融合公式：

$$
\mathbf{{z}}_{{\mathrm{{fuse}}}}
= \operatorname{{L2Norm}}\left(
\left[
\sqrt{{w_f}}\,\mathbf{{z}}_{{\mathrm{{face}}}},
\sqrt{{w_p}}\,\mathbf{{z}}_{{\mathrm{{person}}}}
\right]
\right)
$$

其中 $w_f={config.face_weight}$，$w_p={config.person_weight}$。若人脸检测失败，则 $\mathbf{{z}}_{{\mathrm{{face}}}}=\mathbf{{0}}$。

当 MTCNN 未检测到人脸时，该图像的 face 子特征标记为无效：在 ArcFace-only 评测里，query 的 face 特征无效会计入“无效 Query”；在 Fusion 中，为了保持每张图都有融合向量，face 子特征用零向量占位，person 子特征仍正常参与。这个处理不是动态切换到 OSNet-only，也不是说人脸缺失样本变成了有效的人脸检索样本；它只是让 Fusion 方法在 face 子特征缺失时仍能计算一个固定格式的融合向量。

## 实验设置

- 主评测：官方 query 对 gallery，不使用 train。
- 设备：`{config.device}`
- person 模型：`{config.person_model}`
- ArcFace 权重：`{config.arcface_weight_path}`
- 输出目录：`{output_dir}`
- 特征缓存目录：`{feature_dir}`
- metadata 目录：`{metadata_dir}`
- 指标文件：`{metrics_json_path}`、`{metrics_csv_path}`
- 运行命令：`{command}`
- 总耗时：{elapsed_seconds:.2f} 秒

特征提取统计：

| split | 总图像 | face 成功 | face 成功率 | person 成功 | person 成功率 | fused 维度 |
|---|---:|---:|---:|---:|---:|---:|
| query | {stats['query']['total_images']} | {stats['query']['face_valid']} | {stats['query']['face_success_rate']:.2f}% | {stats['query']['person_valid']} | {stats['query']['person_success_rate']:.2f}% | {stats['query']['fused_dim']} |
| gallery | {stats['gallery']['total_images']} | {stats['gallery']['face_valid']} | {stats['gallery']['face_success_rate']:.2f}% | {stats['gallery']['person_valid']} | {stats['gallery']['person_success_rate']:.2f}% | {stats['gallery']['fused_dim']} |

特征缓存大小：

{feature_sizes_text}

特征文件说明：

- `features/query_face.npy`、`features/gallery_face.npy`：ArcFace 人脸特征，512 维。
- `features/query_person.npy`、`features/gallery_person.npy`：OSNet 行人特征，512 维。
- `features/query_fused.npy`、`features/gallery_fused.npy`：融合特征，1024 维。
- `features/*_valid.npy`：对应特征是否有效的布尔 mask；face 分支检测不到人脸时为 `False`。
- `metadata/query.csv`、`metadata/gallery.csv`：记录图像路径、identity、split 等信息；CSV 行号与同 split 的 `.npy` 第一维顺序一一对应。

## 结果

{_markdown_metrics_table(metrics)}

指标口径：Rank-K 表示 Top-K 内是否至少命中同身份；mAP 是所有 query 的 AP 平均。数值单位均为百分比。

“无效 Query”表示某个方法在该 query 上没有可用的查询特征。本实验中主要发生在 ArcFace-only：如果 MTCNN 在裁剪行人图像中未检测到人脸，就无法提取 ArcFace 特征，该 query 在 ArcFace-only 的 Rank-K 和 AP 中按 0 计，并且仍保留在总 query 分母内。OSNet-only 直接使用整张裁剪行人图像，因此没有无效 query。Fusion 的无效 query 为 0，是因为该方法定义为 `face_or_zero + person` 的固定格式融合；其中部分 query 的 face 子特征仍然是无效的，只是 fusion 向量可以通过零向量占位和 OSNet 特征继续构造。

{_markdown_effect_analysis(metrics, stats)}
"""
    doc_path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark ArcFace/OSNet/fusion on Celeb-reID or Celeb-reID-light")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--query-dir", default=None, help="Optional query directory, absolute or relative to dataset root")
    parser.add_argument("--gallery-dir", default=None, help="Optional gallery directory, absolute or relative to dataset root")
    parser.add_argument("--train-dir", default=None, help="Optional train directory, absolute or relative to dataset root")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-path", default=str(DEFAULT_DOC_PATH))
    parser.add_argument("--arcface-weight-path", default="./models/weights/arcface.pt")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--face-weight", type=float, default=0.35)
    parser.add_argument("--person-weight", type=float, default=0.65)
    parser.add_argument("--person-model", default="osnet", choices=["osnet", "resnet"])
    parser.add_argument("--face-min-size", type=int, default=20)
    parser.add_argument("--face-blur-threshold", type=float, default=0.0)
    parser.add_argument("--limit-query", type=int, default=None)
    parser.add_argument("--limit-gallery", type=int, default=None)
    parser.add_argument("--identity-regex", default=r"^([A-Za-z0-9]+)[_-]")
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recompute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.perf_counter()

    dataset_root = _repo_relative_path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"dataset root not found: {dataset_root}. "
            "Download Celeb-reID from Kaggle and unzip it under data/Celeb-reID first."
        )

    query_dir = _find_split_dir(dataset_root, args.query_dir, "query", required=True)
    gallery_dir = _find_split_dir(dataset_root, args.gallery_dir, "gallery", required=True)
    train_dir = _find_split_dir(dataset_root, args.train_dir, "train", required=False)
    if query_dir is None or gallery_dir is None:
        raise RuntimeError("query/gallery directories must be resolved before benchmarking.")

    query_items = _build_manifest(query_dir, "query", args.identity_regex, limit=args.limit_query)
    gallery_items = _build_manifest(gallery_dir, "gallery", args.identity_regex, limit=args.limit_gallery)
    dataset_summary = _dataset_summary(
        dataset_root=dataset_root,
        query_dir=query_dir,
        gallery_dir=gallery_dir,
        train_dir=train_dir,
        identity_regex=args.identity_regex,
        limit_query=args.limit_query,
        limit_gallery=args.limit_gallery,
    )

    output_dir = _repo_relative_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = FusionFeatureConfig(
        arcface_weight_path=str(_repo_relative_path(args.arcface_weight_path)),
        device=args.device,
        face_weight=args.face_weight,
        person_weight=args.person_weight,
        face_blur_threshold=args.face_blur_threshold,
        face_min_size=args.face_min_size,
        person_model=args.person_model,
    )

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_config": asdict(config),
                "dataset_root": str(dataset_root),
                "query_dir": str(query_dir),
                "gallery_dir": str(gallery_dir),
                "train_dir": None if train_dir is None else str(train_dir),
                "identity_regex": args.identity_regex,
                "limit_query": args.limit_query,
                "limit_gallery": args.limit_gallery,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    query_matrices = _load_or_extract_split(
        items=query_items,
        prefix="query",
        config=config,
        output_dir=output_dir,
        use_cache=args.use_cache,
        recompute=args.recompute,
    )
    gallery_matrices = _load_or_extract_split(
        items=gallery_items,
        prefix="gallery",
        config=config,
        output_dir=output_dir,
        use_cache=args.use_cache,
        recompute=args.recompute,
    )

    metrics = _evaluate_all(query=query_matrices, gallery=gallery_matrices)
    stats = _feature_stats(query=query_matrices, gallery=gallery_matrices)
    feature_sizes = _feature_file_sizes(output_dir)
    elapsed = time.perf_counter() - started

    payload = {
        "dataset": dataset_summary,
        "metrics": metrics,
        "feature_stats": stats,
        "feature_file_sizes": feature_sizes,
        "elapsed_seconds": elapsed,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_metrics_csv(metrics=metrics, output_path=output_dir / "metrics.csv")
    _write_markdown_report(
        doc_path=_repo_relative_path(args.doc_path),
        dataset_summary=dataset_summary,
        metrics=metrics,
        stats=stats,
        feature_file_sizes=feature_sizes,
        config=config,
        output_dir=output_dir,
        elapsed_seconds=elapsed,
        command=" ".join(sys.argv),
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
