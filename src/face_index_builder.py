"""特征索引构建函数（face/person，图像库/视频库路径输入）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .face_feature_pipeline import FaceFeatureBundle, FaceFeaturePipeline, FaceFeaturePipelineConfig
from .person_feature_pipeline import PersonFeatureBundle, PersonFeaturePipeline, PersonFeaturePipelineConfig
from .tools.feature_extractor import FeatureMode


LibraryType = Literal["image", "video", "auto"]
FeatureModeLiteral = Literal["face", "person"]


@dataclass(slots=True)
class FeatureIndexBuildResult:
    """索引构建结果。"""

    library_path: str
    library_type: str
    feature_mode: str
    total_items: int
    feature_dim: int
    output_paths: dict[str, str]
    bundle: FaceFeatureBundle | PersonFeatureBundle

    @property
    def total_faces(self) -> int:
        """兼容旧字段名（历史 face-only 调用）。"""
        return self.total_items

    def summary(self) -> dict[str, object]:
        """返回简要统计信息。"""
        return {
            "library_path": self.library_path,
            "library_type": self.library_type,
            "feature_mode": self.feature_mode,
            "total_items": self.total_items,
            "feature_dim": self.feature_dim,
            "output_paths": self.output_paths,
        }


def _resolve_library_type(path: str, library_type: LibraryType) -> str:
    if library_type in ("image", "video"):
        return library_type

    p = Path(path)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if p.is_file():
        ext = p.suffix.lower()
        if ext in image_exts:
            return "image"
        if ext in video_exts:
            return "video"
        raise ValueError(f"Unsupported file type: {path}")

    if p.is_dir():
        has_image = any(file.suffix.lower() in image_exts for file in p.rglob("*") if file.is_file())
        has_video = any(file.suffix.lower() in video_exts for file in p.rglob("*") if file.is_file())
        if has_image and not has_video:
            return "image"
        if has_video and not has_image:
            return "video"
        if has_image and has_video:
            raise ValueError("Both image and video files exist in library_path. Please set library_type explicitly.")
        raise ValueError(f"No supported media files found in directory: {path}")

    raise FileNotFoundError(f"library_path does not exist: {path}")


def _extract_face_bundle(
    library_path: str,
    resolved_type: str,
    arcface_weight_path: str,
    device: str,
    sample_fps: float,
) -> FaceFeatureBundle:
    path_obj = Path(library_path)
    pipeline = FaceFeaturePipeline(
        FaceFeaturePipelineConfig(
            arcface_weight_path=arcface_weight_path,
            device=device,
        )
    )
    if resolved_type == "image":
        if path_obj.is_file():
            return pipeline.extract_image(image_path=str(path_obj), source_name=str(path_obj))
        return pipeline.extract_image_library(image_dir=str(path_obj))
    if path_obj.is_file():
        return pipeline.extract_video(video_path=str(path_obj), sample_fps=sample_fps)
    return pipeline.extract_video_library(video_dir=str(path_obj), sample_fps=sample_fps)


def _extract_person_bundle(
    library_path: str,
    resolved_type: str,
    arcface_weight_path: str,
    device: str,
    sample_fps: float,
    yolo_weights: str,
    yolo_conf: float,
    yolo_iou: float,
    yolo_max_det: int,
    person_model: str,
    resnet_backbone: str,
    resnet_pretrained: bool,
    resnet_weight_path: str | None,
    person_input_size: int,
) -> PersonFeatureBundle:
    path_obj = Path(library_path)
    pipeline = PersonFeaturePipeline(
        PersonFeaturePipelineConfig(
            arcface_weight_path=arcface_weight_path,
            device=device,
            person_model=person_model,
            resnet_backbone=resnet_backbone,
            resnet_pretrained=resnet_pretrained,
            resnet_weight_path=resnet_weight_path,
            person_input_size=person_input_size,
            yolo_weights=yolo_weights,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            yolo_max_det=yolo_max_det,
        )
    )
    if resolved_type == "image":
        if path_obj.is_file():
            return pipeline.extract_image(image_path=str(path_obj), source_name=str(path_obj))
        return pipeline.extract_image_library(image_dir=str(path_obj))
    if path_obj.is_file():
        return pipeline.extract_video(video_path=str(path_obj), sample_fps=sample_fps)
    return pipeline.extract_video_library(video_dir=str(path_obj), sample_fps=sample_fps)


def _resolve_feature_mode(feature_mode: FeatureMode | FeatureModeLiteral) -> FeatureMode:
    if isinstance(feature_mode, FeatureMode):
        return feature_mode
    try:
        return FeatureMode(str(feature_mode).strip().lower())
    except Exception as exc:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}") from exc


def build_feature_index(
    library_path: str,
    output_dir: str,
    arcface_weight_path: str,
    feature_mode: FeatureMode | FeatureModeLiteral = FeatureMode.FACE,
    library_type: LibraryType = "auto",
    prefix: str = "index",
    device: str = "cpu",
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
) -> FeatureIndexBuildResult:
    """建立特征索引并落盘。

    Args:
        library_path: 图像库/视频库路径（文件或目录）。
        output_dir: 索引输出目录。
        arcface_weight_path: ArcFace 权重文件路径。
        feature_mode: `face` / `person`。
        library_type: `image` / `video` / `auto`。
        prefix: 输出文件名前缀。
        device: 推理设备。
        sample_fps: 视频采样帧率（仅视频生效）。
    """
    resolved_type = _resolve_library_type(library_path, library_type)
    resolved_mode = _resolve_feature_mode(feature_mode)

    if resolved_mode == FeatureMode.FACE:
        bundle = _extract_face_bundle(
            library_path=library_path,
            resolved_type=resolved_type,
            arcface_weight_path=arcface_weight_path,
            device=device,
            sample_fps=sample_fps,
        )
    else:
        bundle = _extract_person_bundle(
            library_path=library_path,
            resolved_type=resolved_type,
            arcface_weight_path=arcface_weight_path,
            device=device,
            sample_fps=sample_fps,
            yolo_weights=yolo_weights,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            yolo_max_det=yolo_max_det,
            person_model=person_model,
            resnet_backbone=resnet_backbone,
            resnet_pretrained=resnet_pretrained,
            resnet_weight_path=resnet_weight_path,
            person_input_size=person_input_size,
        )
        if len(bundle) == 0:
            raise ValueError(
                "No person feature extracted from gallery. "
                "Please check YOLO weights/threshold or gallery content."
            )

    output_prefix = f"{prefix}_{resolved_mode.value}"
    output_paths = bundle.dump(output_dir=output_dir, prefix=output_prefix)
    mat = bundle.feature_matrix()
    feature_dim = int(mat.shape[1]) if mat.ndim == 2 and mat.shape[0] > 0 else 0

    return FeatureIndexBuildResult(
        library_path=library_path,
        library_type=resolved_type,
        feature_mode=resolved_mode.value,
        total_items=len(bundle),
        feature_dim=feature_dim,
        output_paths=output_paths,
        bundle=bundle,
    )


def build_face_feature_index(
    library_path: str,
    output_dir: str,
    arcface_weight_path: str,
    library_type: LibraryType = "auto",
    prefix: str = "face_index",
    device: str = "cpu",
    sample_fps: float = 1.0,
) -> FeatureIndexBuildResult:
    """建立人脸特征索引并落盘（兼容旧接口）。

    新命名规则会落盘为：`<prefix>_face_{features,meta,info}`。
    """
    return build_feature_index(
        library_path=library_path,
        output_dir=output_dir,
        arcface_weight_path=arcface_weight_path,
        feature_mode=FeatureMode.FACE,
        library_type=library_type,
        prefix=prefix,
        device=device,
        sample_fps=sample_fps,
    )


__all__ = [
    "LibraryType",
    "FeatureIndexBuildResult",
    "build_feature_index",
    "build_face_feature_index",
]
