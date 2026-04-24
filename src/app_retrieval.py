"""应用级检索流程：建索引（存在则跳过）+ 执行检索。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.face_index_builder import build_feature_index
from src.retrieval import search_query_in_index
from src.tools.feature_extractor import FeatureMode


def resolve_person_model(person_model: str) -> str:
    model_name = str(person_model or "resnet").strip().lower()
    if model_name == "resnet":
        return "resnet"
    if model_name in {"osnet", "osnet_x1_0"}:
        return "osnet"
    raise ValueError(f"unsupported person_model: {person_model}")


def person_model_index_key(person_model: str, resnet_backbone: str) -> str:
    resolved_model = resolve_person_model(person_model)
    if resolved_model == "osnet":
        return "osnet_x1_0"

    backbone = str(resnet_backbone or "resnet18").strip().lower()
    if backbone not in {"resnet18", "resnet34", "resnet50"}:
        raise ValueError(f"unsupported resnet_backbone: {resnet_backbone}")
    return backbone


def resolve_effective_index_name(
    index_name: str,
    feature_mode: FeatureMode | str,
    person_model: str,
    resnet_backbone: str,
) -> str:
    resolved_mode = _resolve_feature_mode(feature_mode)
    base_name = str(index_name or "").strip()
    if not base_name:
        raise ValueError("index_name is empty after resolving.")
    if resolved_mode != FeatureMode.PERSON:
        return base_name

    key = person_model_index_key(person_model=person_model, resnet_backbone=resnet_backbone)
    if base_name == key or base_name.endswith(f"_{key}"):
        return base_name
    return f"{base_name}_{key}"


def _default_index_name(gallery_path: str) -> str:
    p = Path(gallery_path)
    if p.is_dir():
        return p.name
    return p.stem


def _resolve_gallery_type(gallery_path: str) -> str:
    p = Path(gallery_path)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if p.is_file():
        ext = p.suffix.lower()
        if ext in image_exts:
            return "image"
        if ext in video_exts:
            return "video"
        raise ValueError(f"unsupported gallery file type: {gallery_path}")

    if p.is_dir():
        has_image = any(f.suffix.lower() in image_exts for f in p.rglob("*") if f.is_file())
        has_video = any(f.suffix.lower() in video_exts for f in p.rglob("*") if f.is_file())
        if has_image and not has_video:
            return "image"
        if has_video and not has_image:
            return "video"
        if has_image and has_video:
            raise ValueError("gallery contains both image and video files; please split galleries by type")
        raise ValueError(f"no supported media found in gallery: {gallery_path}")

    raise FileNotFoundError(f"gallery path not found: {gallery_path}")


def _resolve_feature_mode(feature_mode: FeatureMode | str) -> FeatureMode:
    if isinstance(feature_mode, FeatureMode):
        return feature_mode
    try:
        return FeatureMode(str(feature_mode).strip().lower())
    except Exception as exc:
        raise ValueError(f"unsupported feature_mode: {feature_mode}") from exc


def _index_paths(index_name: str, index_dir: str, feature_mode: FeatureMode) -> dict[str, str]:
    suffix = f"{index_name}_{feature_mode.value}"
    d = Path(index_dir)
    return {
        "features_path": str(d / f"{suffix}_features.npy"),
        "meta_path": str(d / f"{suffix}_meta.csv"),
        "info_path": str(d / f"{suffix}_info.json"),
    }


def _index_exists(index_name: str, index_dir: str, feature_mode: FeatureMode) -> bool:
    p = _index_paths(index_name=index_name, index_dir=index_dir, feature_mode=feature_mode)
    return all(Path(v).exists() for v in p.values())


def run_app_retrieval_flow(
    query_path: str,
    gallery_path: str,
    index_name: str | None = None,
    topk: int = 5,
    arcface_weight_path: str = "./models/weights/arcface.pt",
    device: str = "cpu",
    indexes_root: str = "indexes",
    retrieval_output_root: str = "outputs/retrieval",
    feature_mode: FeatureMode | str = FeatureMode.FACE,
    sample_fps: float = 1.0,
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
    """应用级流程：

    1. 根据 gallery_path 决定 index_name（可显式传入）。
    2. 检查索引是否已存在，存在则跳过构建。
    3. 执行检索并写出结果到 outputs/retrieval。
    """
    base_index_name = (index_name or _default_index_name(gallery_path)).strip()
    if not base_index_name:
        raise ValueError("index_name is empty after resolving.")
    library_type = _resolve_gallery_type(gallery_path)
    resolved_mode = _resolve_feature_mode(feature_mode)
    resolved_person_model = resolve_person_model(person_model) if resolved_mode == FeatureMode.PERSON else "resnet"
    resolved_index_name = resolve_effective_index_name(
        index_name=base_index_name,
        feature_mode=resolved_mode,
        person_model=resolved_person_model,
        resnet_backbone=resnet_backbone,
    )

    index_subdir = "image_index" if library_type == "image" else "video_index"
    index_dir = str(Path(indexes_root) / index_subdir)
    index_file_paths = _index_paths(index_name=resolved_index_name, index_dir=index_dir, feature_mode=resolved_mode)

    build_summary: dict[str, Any]
    if _index_exists(index_name=resolved_index_name, index_dir=index_dir, feature_mode=resolved_mode):
        build_summary = {
            "status": "skipped",
            "reason": "index already exists",
            "index_name": resolved_index_name,
            "feature_mode": resolved_mode.value,
            "person_model": resolved_person_model,
            "index_dir": index_dir,
            "index_paths": index_file_paths,
        }
    else:
        result = build_feature_index(
            library_path=gallery_path,
            output_dir=index_dir,
            arcface_weight_path=arcface_weight_path,
            feature_mode=resolved_mode,
            library_type=library_type,
            prefix=resolved_index_name,
            device=device,
            sample_fps=sample_fps,
            yolo_weights=yolo_weights,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            yolo_max_det=yolo_max_det,
            person_model=resolved_person_model,
            resnet_backbone=resnet_backbone,
            resnet_pretrained=resnet_pretrained,
            resnet_weight_path=resnet_weight_path,
            person_input_size=person_input_size,
        )
        build_summary = {
            "status": "built",
            "index_name": resolved_index_name,
            "feature_mode": resolved_mode.value,
            "person_model": resolved_person_model,
            "index_dir": index_dir,
            "index_paths": index_file_paths,
            "total_items": result.total_items,
            "feature_dim": result.feature_dim,
        }

    retrieval_summary = search_query_in_index(
        query_path=query_path,
        index_name=resolved_index_name,
        topk=topk,
        arcface_weight_path=arcface_weight_path,
        device=device,
        indexes_root=index_dir,
        retrieval_output_root=retrieval_output_root,
        feature_mode=resolved_mode,
        yolo_weights=yolo_weights,
        yolo_conf=yolo_conf,
        yolo_iou=yolo_iou,
        yolo_max_det=yolo_max_det,
        person_model=resolved_person_model,
        resnet_backbone=resnet_backbone,
        resnet_pretrained=resnet_pretrained,
        resnet_weight_path=resnet_weight_path,
        person_input_size=person_input_size,
    )

    return {
        "query_path": query_path,
        "gallery_path": gallery_path,
        "library_type": library_type,
        "feature_mode": resolved_mode.value,
        "person_model": resolved_person_model,
        "index_name": resolved_index_name,
        "build": build_summary,
        "retrieval": retrieval_summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="App-level retrieval flow")
    p.add_argument("--query", required=True, help="Query image path")
    p.add_argument("--gallery", required=True, help="Gallery image directory or file path")
    p.add_argument("--index-name", default=None, help="Index name; default uses gallery file/dir name")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--weights", default="./models/weights/arcface.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--indexes-root", default="indexes")
    p.add_argument("--retrieval-output-root", default="outputs/retrieval")
    p.add_argument("--feature-mode", choices=["face", "person"], default="face")
    p.add_argument("--sample-fps", type=float, default=1.0)
    p.add_argument("--yolo-weights", default="./models/weights/yolo11n.pt")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-iou", type=float, default=0.7)
    p.add_argument("--yolo-max-det", type=int, default=100)
    p.add_argument("--person-model", default="resnet", choices=["resnet", "osnet"])
    p.add_argument("--resnet-backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--resnet-pretrained", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resnet-weight-path", default=None)
    p.add_argument("--person-input-size", type=int, default=224)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    summary = run_app_retrieval_flow(
        query_path=args.query,
        gallery_path=args.gallery,
        index_name=args.index_name,
        topk=args.topk,
        arcface_weight_path=args.weights,
        device=args.device,
        indexes_root=args.indexes_root,
        retrieval_output_root=args.retrieval_output_root,
        feature_mode=args.feature_mode,
        sample_fps=args.sample_fps,
        yolo_weights=args.yolo_weights,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_max_det=args.yolo_max_det,
        person_model=args.person_model,
        resnet_backbone=args.resnet_backbone,
        resnet_pretrained=args.resnet_pretrained,
        resnet_weight_path=args.resnet_weight_path,
        person_input_size=args.person_input_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
