from __future__ import annotations

import json
import importlib.util
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.app_retrieval import resolve_effective_index_name, resolve_person_model, run_app_retrieval_flow
from src.face_index_builder import build_feature_index

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_DIR = PROJECT_ROOT / "src" / "app"
DATA_RUNTIME_DIR = PROJECT_ROOT / "data_runtime"
INDEXES_ROOT = PROJECT_ROOT / "indexes"
IMAGE_INDEX_DIR = INDEXES_ROOT / "image_index"
VIDEO_INDEX_DIR = INDEXES_ROOT / "video_index"
WEB_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "web"
RETRIEVAL_OUTPUT_DIR = WEB_OUTPUT_DIR / "retrieval"
ARCFACE_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "arcface.pt"
YOLO_WEIGHT_CANDIDATES = [
    PROJECT_ROOT / "models" / "weights" / "yolo11n.pt",
    PROJECT_ROOT / "yolo11n.pt",
]

DEFAULT_QUERY_DIR = DATA_RUNTIME_DIR / "query"
DEFAULT_GALLERY_IMAGES = PROJECT_ROOT / "data_runtime" / "gallery" / "images"
DEFAULT_GALLERY_VIDEOS = PROJECT_ROOT / "data_runtime" / "gallery" / "videos"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FEATURE_MODES = {"face", "person"}


def ensure_runtime_dirs() -> None:
    for directory in (
        DEFAULT_QUERY_DIR,
        DEFAULT_GALLERY_IMAGES,
        DEFAULT_GALLERY_VIDEOS,
        INDEXES_ROOT,
        IMAGE_INDEX_DIR,
        VIDEO_INDEX_DIR,
        WEB_OUTPUT_DIR,
        RETRIEVAL_OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


ensure_runtime_dirs()


@dataclass(slots=True)
class SearchOptions:
    feature_mode: str = "face"
    index_name: str | None = None
    topk: int = 5
    device: str | None = None
    sample_fps: float = 1.0
    arcface_weight_path: str | None = None
    yolo_weights: str | None = None
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    yolo_max_det: int = 100
    person_model: str = "resnet"
    resnet_backbone: str = "resnet18"
    resnet_pretrained: bool = False
    resnet_weight_path: str | None = None
    person_input_size: int = 224


@dataclass(slots=True)
class RebuildIndexOptions:
    gallery_path: str
    feature_mode: str = "face"
    index_name: str | None = None
    device: str | None = None
    sample_fps: float = 1.0
    arcface_weight_path: str | None = None
    yolo_weights: str | None = None
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    yolo_max_det: int = 100
    person_model: str = "resnet"
    resnet_backbone: str = "resnet18"
    resnet_pretrained: bool = False
    resnet_weight_path: str | None = None
    person_input_size: int = 224


@dataclass(slots=True)
class IndexStatusOptions:
    feature_mode: str = "face"
    index_name: str | None = None
    gallery_path: str = ""
    person_model: str = "resnet"
    resnet_backbone: str = "resnet18"


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_yolo_weights() -> Path:
    for path in YOLO_WEIGHT_CANDIDATES:
        if path.exists():
            return path
    return YOLO_WEIGHT_CANDIDATES[0]


def _resolve_feature_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode: {value}")
    return mode


def _normalize_index_name(index_name: str | None, fallback: str) -> str:
    candidate = (index_name or fallback).strip()
    candidate = re.sub(r"[^a-zA-Z0-9_-]+", "_", candidate)
    candidate = candidate.strip("_")
    if not candidate:
        raise ValueError("index_name is empty after normalization")
    return candidate


def _resolve_input_path(path_value: str, *, must_exist: bool = True) -> Path:
    raw = (path_value or "").strip()
    if not raw:
        raise ValueError("path is required")

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()

    if must_exist and not path.exists():
        raise FileNotFoundError(f"path not found: {path}")
    return path


def _relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _has_supported_file(root: Path, extensions: set[str]) -> bool:
    return root.exists() and any(p.is_file() and p.suffix.lower() in extensions for p in root.rglob("*"))


def _runtime_option(path: Path) -> dict[str, str]:
    value = _relative_to_project(path)
    return {
        "label": value,
        "name": path.name,
        "value": value,
    }


def _list_query_images(root: Path = DEFAULT_QUERY_DIR) -> list[dict[str, str]]:
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=lambda p: _relative_to_project(p).lower())
    return [_runtime_option(p) for p in files]


def _list_gallery_dirs(root: Path, extensions: set[str]) -> list[dict[str, str]]:
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and _has_supported_file(p, extensions)]
    dirs.sort(key=lambda p: _relative_to_project(p).lower())
    return [_runtime_option(p) for p in dirs]


def get_runtime_options() -> dict[str, Any]:
    ensure_runtime_dirs()
    return {
        "roots": {
            "query_images": _relative_to_project(DEFAULT_QUERY_DIR),
            "image_galleries": _relative_to_project(DEFAULT_GALLERY_IMAGES),
            "video_galleries": _relative_to_project(DEFAULT_GALLERY_VIDEOS),
        },
        "query_images": _list_query_images(DEFAULT_QUERY_DIR),
        "image_galleries": _list_gallery_dirs(DEFAULT_GALLERY_IMAGES, IMAGE_EXTENSIONS),
        "video_galleries": _list_gallery_dirs(DEFAULT_GALLERY_VIDEOS, VIDEO_EXTENSIONS),
    }


def resolve_query_path(path_value: str) -> Path:
    path = _resolve_input_path(path_value, must_exist=True)
    if not path.is_file():
        raise ValueError(f"query_path must be a file: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported query image type. Allowed extensions: {allowed}")
    return path


def resolve_gallery_path(path_value: str) -> Path:
    path = _resolve_input_path(path_value, must_exist=True)
    if not path.is_file() and not path.is_dir():
        raise ValueError(f"gallery_path must be a file or directory: {path}")
    return path


def _guess_library_type(gallery_path: Path) -> str:
    if gallery_path.is_file():
        ext = gallery_path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        if ext in VIDEO_EXTENSIONS:
            return "video"
        raise ValueError(f"unsupported gallery file type: {gallery_path}")

    has_image = any(p.suffix.lower() in IMAGE_EXTENSIONS for p in gallery_path.rglob("*") if p.is_file())
    has_video = any(p.suffix.lower() in VIDEO_EXTENSIONS for p in gallery_path.rglob("*") if p.is_file())
    if has_image and not has_video:
        return "image"
    if has_video and not has_image:
        return "video"
    if has_image and has_video:
        raise ValueError("gallery contains both image and video files; split them before indexing")
    raise ValueError(f"no supported media found in gallery: {gallery_path}")


def _index_dir_for_library(library_type: str) -> Path:
    if library_type == "image":
        return IMAGE_INDEX_DIR
    if library_type == "video":
        return VIDEO_INDEX_DIR
    raise ValueError(f"Unsupported library_type: {library_type}")


def _index_paths(index_dir: Path, index_name: str, feature_mode: str) -> dict[str, str]:
    suffix = f"{index_name}_{feature_mode}"
    return {
        "features_path": str(index_dir / f"{suffix}_features.npy"),
        "meta_path": str(index_dir / f"{suffix}_meta.csv"),
        "info_path": str(index_dir / f"{suffix}_info.json"),
    }


def path_to_url(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(WEB_OUTPUT_DIR.resolve())
        return f"/outputs-static/{rel.as_posix()}"
    except ValueError:
        pass
    try:
        rel = resolved.relative_to(DATA_RUNTIME_DIR.resolve())
        return f"/data-runtime-static/{rel.as_posix()}"
    except ValueError:
        return ""


def _to_search_options(options: SearchOptions | RebuildIndexOptions) -> dict[str, Any]:
    mode = _resolve_feature_mode(options.feature_mode)
    device = options.device or _default_device()
    return {
        "feature_mode": mode,
        "topk": max(1, int(getattr(options, "topk", 5))),
        "device": device,
        "sample_fps": max(0.1, float(options.sample_fps)),
        "arcface_weight_path": str(_resolve_input_path(options.arcface_weight_path, must_exist=False))
        if options.arcface_weight_path
        else str(ARCFACE_WEIGHTS),
        "yolo_weights": str(_resolve_input_path(options.yolo_weights, must_exist=False))
        if options.yolo_weights
        else str(_default_yolo_weights()),
        "yolo_conf": float(options.yolo_conf),
        "yolo_iou": float(options.yolo_iou),
        "yolo_max_det": int(options.yolo_max_det),
        "person_model": resolve_person_model(options.person_model) if mode == "person" else "resnet",
        "resnet_backbone": str(options.resnet_backbone),
        "resnet_pretrained": bool(options.resnet_pretrained),
        "resnet_weight_path": (
            str(_resolve_input_path(options.resnet_weight_path, must_exist=False)) if options.resnet_weight_path else None
        ),
        "person_input_size": int(options.person_input_size),
    }


def _count_indexes(index_dir: Path, mode: str) -> int:
    pattern = f"*_{mode}_features.npy"
    return len(list(index_dir.glob(pattern))) if index_dir.exists() else 0


def get_index_status(options: IndexStatusOptions) -> dict[str, Any]:
    mode = _resolve_feature_mode(options.feature_mode)
    person_model = resolve_person_model(options.person_model) if mode == "person" else "resnet"
    gallery_path = resolve_gallery_path(options.gallery_path)
    library_type = _guess_library_type(gallery_path)
    fallback = gallery_path.name if gallery_path.is_dir() else gallery_path.stem
    base_index_name = _normalize_index_name(options.index_name, fallback)
    resolved_index_name = resolve_effective_index_name(
        index_name=base_index_name,
        feature_mode=mode,
        person_model=person_model,
        resnet_backbone=options.resnet_backbone,
    )
    index_dir = _index_dir_for_library(library_type)
    index_paths = _index_paths(index_dir=index_dir, index_name=resolved_index_name, feature_mode=mode)
    missing_paths = [path for path in index_paths.values() if not Path(path).exists()]

    return {
        "exists": not missing_paths,
        "library_type": library_type,
        "feature_mode": mode,
        "person_model": person_model,
        "index_name": resolved_index_name,
        "index_dir": str(index_dir),
        "index_paths": index_paths,
        "missing_paths": missing_paths,
    }


def _is_module_importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def get_status() -> dict[str, Any]:
    return {
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "ultralytics_importable": _is_module_importable("ultralytics"),
            "uvicorn_importable": _is_module_importable("uvicorn"),
            "torchreid_importable": _is_module_importable("torchreid"),
        },
        "weights": {
            "arcface": {
                "path": str(ARCFACE_WEIGHTS),
                "exists": ARCFACE_WEIGHTS.exists(),
            },
            "yolo_default": {
                "path": str(_default_yolo_weights()),
                "exists": _default_yolo_weights().exists(),
                "candidates": [str(p) for p in YOLO_WEIGHT_CANDIDATES],
            },
        },
        "defaults": {
            "gallery_images": str(DEFAULT_GALLERY_IMAGES),
            "gallery_videos": str(DEFAULT_GALLERY_VIDEOS),
            "query_images": str(DEFAULT_QUERY_DIR),
            "indexes_root": str(INDEXES_ROOT),
            "web_output_root": str(WEB_OUTPUT_DIR),
            "retrieval_output_root": str(RETRIEVAL_OUTPUT_DIR),
        },
        "frontend": {
            "root": str(FRONTEND_DIR),
            "index_exists": (FRONTEND_DIR / "index.html").exists(),
            "app_js_exists": (FRONTEND_DIR / "app.js").exists(),
            "styles_exists": (FRONTEND_DIR / "styles.css").exists(),
        },
        "indexes": {
            "image": {
                "path": str(IMAGE_INDEX_DIR),
                "exists": IMAGE_INDEX_DIR.exists(),
                "face_count": _count_indexes(IMAGE_INDEX_DIR, "face"),
                "person_count": _count_indexes(IMAGE_INDEX_DIR, "person"),
            },
            "video": {
                "path": str(VIDEO_INDEX_DIR),
                "exists": VIDEO_INDEX_DIR.exists(),
                "face_count": _count_indexes(VIDEO_INDEX_DIR, "face"),
                "person_count": _count_indexes(VIDEO_INDEX_DIR, "person"),
            },
        },
        "feature_modes": sorted(FEATURE_MODES),
        "person_models": ["resnet", "osnet"],
        "default_device": _default_device(),
    }


def rebuild_gallery_index(options: RebuildIndexOptions) -> dict[str, Any]:
    ensure_runtime_dirs()
    search_options = _to_search_options(options)
    gallery_path = resolve_gallery_path(options.gallery_path)
    library_type = _guess_library_type(gallery_path)

    base_index_name = _normalize_index_name(options.index_name, gallery_path.name if gallery_path.is_dir() else gallery_path.stem)
    resolved_index_name = resolve_effective_index_name(
        index_name=base_index_name,
        feature_mode=search_options["feature_mode"],
        person_model=search_options["person_model"],
        resnet_backbone=search_options["resnet_backbone"],
    )
    index_dir = _index_dir_for_library(library_type)

    result = build_feature_index(
        library_path=str(gallery_path),
        output_dir=str(index_dir),
        arcface_weight_path=search_options["arcface_weight_path"],
        feature_mode=search_options["feature_mode"],
        library_type=library_type,
        prefix=resolved_index_name,
        device=search_options["device"],
        sample_fps=search_options["sample_fps"],
        yolo_weights=search_options["yolo_weights"],
        yolo_conf=search_options["yolo_conf"],
        yolo_iou=search_options["yolo_iou"],
        yolo_max_det=search_options["yolo_max_det"],
        person_model=search_options["person_model"],
        resnet_backbone=search_options["resnet_backbone"],
        resnet_pretrained=search_options["resnet_pretrained"],
        resnet_weight_path=search_options["resnet_weight_path"],
        person_input_size=search_options["person_input_size"],
    )

    return {
        "gallery_path": str(gallery_path),
        "library_type": result.library_type,
        "feature_mode": result.feature_mode,
        "person_model": search_options["person_model"],
        "index_name": resolved_index_name,
        "index_paths": result.output_paths,
        "index_dir": str(index_dir),
        "total_items": result.total_items,
        "feature_dim": result.feature_dim,
    }


def _load_result_payload(result_json_path: str | Path) -> dict[str, Any]:
    path = Path(result_json_path)
    if not path.exists():
        raise FileNotFoundError(f"result_json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in payload.get("results", []):
        item = dict(row)
        item["crop_url"] = path_to_url(item.get("crop_path", ""))
        item["annotated_url"] = path_to_url(item.get("annotated_path", ""))
        flattened.append(item)
    return flattened


def search_gallery(query_path: Path | str, gallery_path: str, options: SearchOptions) -> dict[str, Any]:
    ensure_runtime_dirs()
    search_options = _to_search_options(options)
    resolved_query = resolve_query_path(str(query_path))
    resolved_gallery = resolve_gallery_path(gallery_path)
    fallback_name = resolved_gallery.name if resolved_gallery.is_dir() else resolved_gallery.stem
    base_index_name = _normalize_index_name(options.index_name, fallback_name)
    resolved_index_name = resolve_effective_index_name(
        index_name=base_index_name,
        feature_mode=search_options["feature_mode"],
        person_model=search_options["person_model"],
        resnet_backbone=search_options["resnet_backbone"],
    )

    summary = run_app_retrieval_flow(
        query_path=str(resolved_query),
        gallery_path=str(resolved_gallery),
        index_name=resolved_index_name,
        topk=search_options["topk"],
        arcface_weight_path=search_options["arcface_weight_path"],
        device=search_options["device"],
        indexes_root=str(INDEXES_ROOT),
        retrieval_output_root=str(RETRIEVAL_OUTPUT_DIR),
        feature_mode=search_options["feature_mode"],
        sample_fps=search_options["sample_fps"],
        yolo_weights=search_options["yolo_weights"],
        yolo_conf=search_options["yolo_conf"],
        yolo_iou=search_options["yolo_iou"],
        yolo_max_det=search_options["yolo_max_det"],
        person_model=search_options["person_model"],
        resnet_backbone=search_options["resnet_backbone"],
        resnet_pretrained=search_options["resnet_pretrained"],
        resnet_weight_path=search_options["resnet_weight_path"],
        person_input_size=search_options["person_input_size"],
    )

    payload = _load_result_payload(summary["retrieval"]["result_json"])
    results = _flatten_results(payload)

    return {
        "query_url": path_to_url(resolved_query),
        "query_path": str(resolved_query),
        "gallery_path": str(resolved_gallery),
        "library_type": summary["library_type"],
        "feature_mode": summary["feature_mode"],
        "person_model": summary["person_model"],
        "index_name": summary["index_name"],
        "build": summary["build"],
        "retrieval": summary["retrieval"],
        "results": results,
        "result_count": len(results),
    }


def clear_web_outputs() -> dict[str, Any]:
    if RETRIEVAL_OUTPUT_DIR.exists():
        shutil.rmtree(RETRIEVAL_OUTPUT_DIR)
    RETRIEVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "cleared": True,
        "retrieval_output_dir": str(RETRIEVAL_OUTPUT_DIR),
    }
