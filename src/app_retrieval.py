"""应用级检索流程：建索引（存在则跳过）+ 执行检索。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.face_index_builder import build_face_feature_index
from src.retrieval import search_query_in_index


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


def _index_paths(index_name: str, index_dir: str) -> dict[str, str]:
    d = Path(index_dir)
    return {
        "features_path": str(d / f"{index_name}_features.npy"),
        "meta_path": str(d / f"{index_name}_meta.csv"),
        "info_path": str(d / f"{index_name}_info.json"),
    }


def _index_exists(index_name: str, index_dir: str) -> bool:
    p = _index_paths(index_name=index_name, index_dir=index_dir)
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
) -> dict[str, Any]:
    """应用级流程：

    1. 根据 gallery_path 决定 index_name（可显式传入）。
    2. 检查索引是否已存在，存在则跳过构建。
    3. 执行检索并写出结果到 outputs/retrieval。
    """
    resolved_index_name = (index_name or _default_index_name(gallery_path)).strip()
    if not resolved_index_name:
        raise ValueError("index_name is empty after resolving.")
    library_type = _resolve_gallery_type(gallery_path)

    index_subdir = "image_index" if library_type == "image" else "video_index"
    index_dir = str(Path(indexes_root) / index_subdir)
    index_file_paths = _index_paths(index_name=resolved_index_name, index_dir=index_dir)

    build_summary: dict[str, Any]
    if _index_exists(index_name=resolved_index_name, index_dir=index_dir):
        build_summary = {
            "status": "skipped",
            "reason": "index already exists",
            "index_name": resolved_index_name,
            "index_dir": index_dir,
            "index_paths": index_file_paths,
        }
    else:
        result = build_face_feature_index(
            library_path=gallery_path,
            output_dir=index_dir,
            arcface_weight_path=arcface_weight_path,
            library_type=library_type,
            prefix=resolved_index_name,
            device=device,
        )
        build_summary = {
            "status": "built",
            "index_name": resolved_index_name,
            "index_dir": index_dir,
            "index_paths": index_file_paths,
            "total_faces": result.total_faces,
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
    )

    return {
        "query_path": query_path,
        "gallery_path": gallery_path,
        "library_type": library_type,
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
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
