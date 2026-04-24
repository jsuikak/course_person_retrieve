"""最简检索接口：query + 索引名 -> TopK 结果落盘（face/person 共用）。"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .tools.feature_extractor import FeatureExtractor, FeatureExtractorConfig, FeatureMode


def _resolve_feature_mode(feature_mode: FeatureMode | str) -> FeatureMode:
    if isinstance(feature_mode, FeatureMode):
        return feature_mode
    try:
        return FeatureMode(str(feature_mode).strip().lower())
    except Exception as exc:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}") from exc


def _l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if mat.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if mat.shape[0] == 0:
        return mat.astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1, keepdims=True).astype(np.float32)
    norms = np.maximum(norms, eps)
    return (mat / norms).astype(np.float32, copy=False)


def _to_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _find_index_files(index_name: str, indexes_root: str, feature_mode: FeatureMode) -> tuple[Path, Path, str]:
    root = Path(indexes_root)
    if not root.exists():
        raise FileNotFoundError(f"indexes root not found: {indexes_root}")

    key = Path(index_name).stem
    pattern = f"{key}_{feature_mode.value}_features.npy"
    candidates = sorted(root.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"index not found for name={index_name}, mode={feature_mode.value}. "
            f"expected: **/{pattern} under {indexes_root}"
        )
    if len(candidates) > 1:
        joined = ", ".join(str(p) for p in candidates)
        raise ValueError(f"multiple index files match name={index_name}, mode={feature_mode.value}: {joined}")

    features_path = candidates[0]
    meta_path = features_path.with_name(features_path.name.replace("_features.npy", "_meta.csv"))
    if not meta_path.exists():
        raise FileNotFoundError(f"meta file not found: {meta_path}")
    return features_path, meta_path, key


def _resolve_source_path(source_name: str, project_root: Path) -> Path:
    raw = Path(source_name)
    if raw.exists():
        return raw

    rel = project_root / raw
    if rel.exists():
        return rel

    for base in (
        project_root / "data_runtime" / "gallery" / "images",
        project_root / "data_runtime" / "gallery" / "videos",
        project_root / "data",
    ):
        p = base / raw.name
        if p.exists():
            return p

    raise FileNotFoundError(f"source file not found: {source_name}")


def _read_source_image(source_type: str, source_path: Path, frame_index: int) -> np.ndarray:
    if source_type == "video":
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"failed to open video: {source_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(f"failed to read frame={frame_index} from video: {source_path}")
        return frame

    image = cv2.imread(str(source_path))
    if image is None or image.size == 0:
        raise FileNotFoundError(f"failed to read image: {source_path}")
    return image


def _crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    h_img, w_img = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=image.dtype)
    return image[y1:y2, x1:x2]


def _draw_bbox_with_label(image: np.ndarray, x: int, y: int, w: int, h: int, label: str) -> np.ndarray:
    out = image.copy()
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text_y = y1 - 8 if y1 > 16 else y1 + 18
    cv2.putText(out, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def _read_query_image(query_file: Path) -> np.ndarray:
    query_image = cv2.imread(str(query_file))
    if query_image is None or query_image.size == 0:
        raise FileNotFoundError(f"failed to read query image: {query_file}")
    return query_image


def _extract_face_query_feature(
    query_file: Path,
    arcface_weight_path: str,
    device: str,
) -> tuple[np.ndarray, dict[str, int]]:
    query_image = _read_query_image(query_file)

    extractor = FeatureExtractor(
        FeatureExtractorConfig(
            arcface_weight_path=arcface_weight_path,
            device=device,
            detect_face=False,
            face_flip_test=True,
        )
    )
    feat = extractor.extract(FeatureMode.FACE, query_image)
    if feat is None:
        raise ValueError(f"failed to extract face feature from query image: {query_file}")

    h, w = query_image.shape[:2]
    return (
        feat.reshape(1, -1).astype(np.float32, copy=False),
        {"x": 0, "y": 0, "w": int(w), "h": int(h)},
    )


def _extract_person_query_feature(
    query_file: Path,
    arcface_weight_path: str,
    device: str,
    person_model: str,
    resnet_backbone: str,
    resnet_pretrained: bool,
    resnet_weight_path: str | None,
    person_input_size: int,
) -> np.ndarray:
    query_image = _read_query_image(query_file)

    extractor = FeatureExtractor(
        FeatureExtractorConfig(
            arcface_weight_path=arcface_weight_path,
            device=device,
            detect_face=False,
            face_flip_test=False,
            person_model=person_model,
            resnet_backbone=resnet_backbone,
            resnet_pretrained=resnet_pretrained,
            resnet_weight_path=resnet_weight_path,
            person_input_size=person_input_size,
        )
    )
    feat = extractor.extract(FeatureMode.PERSON, query_image)
    if feat is None:
        raise ValueError(f"failed to extract person feature from query image: {query_file}")
    return feat.reshape(1, -1).astype(np.float32, copy=False)


def search_query_in_index(
    query_path: str,
    index_name: str,
    topk: int = 5,
    arcface_weight_path: str = "./models/weights/arcface.pt",
    device: str = "cpu",
    indexes_root: str = "indexes",
    retrieval_output_root: str = "outputs/retrieval",
    feature_mode: FeatureMode | str = FeatureMode.FACE,
    yolo_weights: str = "./models/weights/yolo11n.pt",
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.7,
    yolo_max_det: int = 100,
    person_model: str = "resnet",
    resnet_backbone: str = "resnet18",
    resnet_pretrained: bool = False,
    resnet_weight_path: str | None = None,
    person_input_size: int = 224,
) -> dict[str, Any]:
    """执行最简检索并将结果写入文件夹。"""
    del yolo_weights, yolo_conf, yolo_iou, yolo_max_det  # 当前检索阶段 query 不做人脸/人体检测，保留参数用于接口统一。

    resolved_mode = _resolve_feature_mode(feature_mode)
    project_root = Path.cwd()
    query_file = Path(query_path)
    if not query_file.exists():
        query_file = project_root / query_file
    if not query_file.exists():
        raise FileNotFoundError(f"query file not found: {query_path}")

    features_path, meta_path, library_name = _find_index_files(
        index_name=index_name,
        indexes_root=indexes_root,
        feature_mode=resolved_mode,
    )
    gallery_features = np.load(features_path).astype(np.float32, copy=False)
    if gallery_features.ndim != 2:
        raise ValueError(f"invalid gallery feature matrix shape: {gallery_features.shape}")
    if gallery_features.shape[0] == 0 or gallery_features.shape[1] == 0:
        raise ValueError(
            f"index contains no valid features: {features_path}. "
            "Please rebuild the index with valid detections."
        )

    with meta_path.open("r", newline="", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))
    if len(meta_rows) != gallery_features.shape[0]:
        raise ValueError(
            f"meta rows ({len(meta_rows)}) do not match feature rows ({gallery_features.shape[0]}): {meta_path}"
        )

    query_face_bbox: dict[str, int] | None = None
    if resolved_mode == FeatureMode.FACE:
        query_features, query_face_bbox = _extract_face_query_feature(
            query_file=query_file,
            arcface_weight_path=arcface_weight_path,
            device=device,
        )
    else:
        query_features = _extract_person_query_feature(
            query_file=query_file,
            arcface_weight_path=arcface_weight_path,
            device=device,
            person_model=person_model,
            resnet_backbone=resnet_backbone,
            resnet_pretrained=resnet_pretrained,
            resnet_weight_path=resnet_weight_path,
            person_input_size=person_input_size,
        )

    if query_features.ndim != 2 or query_features.shape[1] != gallery_features.shape[1]:
        detail = (
            f"feature dimension mismatch for mode={resolved_mode.value}: "
            f"index dim={gallery_features.shape[1]}, query dim={query_features.shape[1]}"
        )
        if resolved_mode == FeatureMode.PERSON:
            detail += (
                f". Rebuild/select an index for person_model={person_model}, "
                f"resnet_backbone={resnet_backbone}."
            )
        raise ValueError(detail)

    gallery_norm = _l2_normalize_rows(gallery_features)
    query_norm = _l2_normalize_rows(query_features)
    if resolved_mode == FeatureMode.FACE:
        scores_matrix = gallery_norm @ query_norm.T
        best_query_idx = np.argmax(scores_matrix, axis=1)
        best_scores = scores_matrix[np.arange(scores_matrix.shape[0]), best_query_idx]
    else:
        best_scores = gallery_norm @ query_norm[0]
        best_query_idx = np.zeros((gallery_norm.shape[0],), dtype=np.int64)

    k = max(1, min(int(topk), gallery_features.shape[0]))
    top_indices = np.argsort(-best_scores)[:k]

    output_dir = Path(retrieval_output_root) / f"{query_file.stem}-{library_name}-{resolved_mode.value}"
    crops_dir = output_dir / "crops"
    annotated_dir = output_dir / "annotated"
    if output_dir.exists():
        # 避免历史运行残留文件与本次结果混杂（例如 topk 变化或切换 image/video 索引）。
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    query_label = query_file.stem
    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        row = meta_rows[int(idx)]
        source_type = row.get("source_type", "image")
        source_name = row.get("source_name", "")
        frame_index = _to_int(row.get("frame_index"), -1)
        face_index = _to_int(row.get("face_index"), -1)
        person_index = _to_int(row.get("person_index"), -1)
        x = _to_int(row.get("x"), 0)
        y = _to_int(row.get("y"), 0)
        w = _to_int(row.get("w"), 0)
        h = _to_int(row.get("h"), 0)

        source_path = _resolve_source_path(source_name, project_root=project_root)
        source_image = _read_source_image(source_type=source_type, source_path=source_path, frame_index=frame_index)

        crop = _crop(source_image, x=x, y=y, w=w, h=h)
        if crop.size == 0:
            crop = source_image

        if w > 0 and h > 0:
            annotated = _draw_bbox_with_label(source_image, x=x, y=y, w=w, h=h, label=query_label)
        else:
            annotated = source_image.copy()

        stem = f"rank{rank:02d}_row{int(idx):06d}"
        crop_path = crops_dir / f"{stem}.jpg"
        annotated_path = annotated_dir / f"{stem}.jpg"
        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(f"failed to write crop: {crop_path}")
        if not cv2.imwrite(str(annotated_path), annotated):
            raise RuntimeError(f"failed to write annotated image: {annotated_path}")

        item: dict[str, Any] = {
            "rank": rank,
            "row_index": int(idx),
            "score": float(best_scores[int(idx)]),
            "source_type": source_type,
            "source_name": source_name,
            "frame_index": frame_index,
            "face_index": face_index,
            "person_index": person_index,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "crop_path": str(crop_path),
            "annotated_path": str(annotated_path),
        }

        q_i = int(best_query_idx[int(idx)])
        if resolved_mode == FeatureMode.FACE:
            item["matched_query_face_index"] = q_i
            item["matched_query_source"] = "whole_image"
            if query_face_bbox is not None:
                item["matched_query_bbox"] = dict(query_face_bbox)
        else:
            item["matched_query_person_index"] = 0
            item["matched_query_source"] = "whole_image"

        results.append(item)

    result_json = output_dir / "results.json"
    payload = {
        "query_path": str(query_file),
        "index_name": library_name,
        "feature_mode": resolved_mode.value,
        "features_path": str(features_path),
        "meta_path": str(meta_path),
        "topk": k,
        "query_item_count": int(query_features.shape[0]),
        "results": results,
    }
    if resolved_mode == FeatureMode.FACE:
        payload["query_face_count"] = int(query_features.shape[0])
    else:
        payload["query_person_count"] = int(query_features.shape[0])
        payload["person_model"] = str(person_model or "resnet").strip().lower()

    with result_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    out = {
        "output_dir": str(output_dir),
        "result_json": str(result_json),
        "feature_mode": resolved_mode.value,
        "topk": k,
        "query_item_count": int(query_features.shape[0]),
    }
    if resolved_mode == FeatureMode.FACE:
        out["query_face_count"] = int(query_features.shape[0])
    else:
        out["query_person_count"] = int(query_features.shape[0])
        out["person_model"] = str(person_model or "resnet").strip().lower()
    return out


__all__ = ["search_query_in_index"]
