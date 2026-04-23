"""人脸特征索引构建函数（图像库/视频库路径输入）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .face_feature_pipeline import FaceFeatureBundle, FaceFeaturePipeline, FaceFeaturePipelineConfig


LibraryType = Literal["image", "video", "auto"]


@dataclass(slots=True)
class FaceIndexBuildResult:
    """索引构建结果。"""

    library_path: str
    library_type: str
    total_faces: int
    feature_dim: int
    output_paths: dict[str, str]
    bundle: FaceFeatureBundle

    def summary(self) -> dict[str, object]:
        """返回简要统计信息。"""
        return {
            "library_path": self.library_path,
            "library_type": self.library_type,
            "total_faces": self.total_faces,
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


def build_face_feature_index(
    library_path: str,
    output_dir: str,
    arcface_weight_path: str,
    library_type: LibraryType = "auto",
    prefix: str = "face_index",
    device: str = "cpu",
    sample_fps: float = 1.0,
) -> FaceIndexBuildResult:
    """建立人脸特征索引并落盘。

    Args:
        library_path: 图像库/视频库路径（文件或目录）。
        output_dir: 索引输出目录。
        arcface_weight_path: ArcFace 权重文件路径。
        library_type: `image` / `video` / `auto`。
        prefix: 输出文件名前缀。
        device: 推理设备。
        sample_fps: 视频采样帧率（仅视频生效）。
    """
    resolved_type = _resolve_library_type(library_path, library_type)
    path_obj = Path(library_path)

    pipeline = FaceFeaturePipeline(
        FaceFeaturePipelineConfig(
            arcface_weight_path=arcface_weight_path,
            device=device,
        )
    )

    if resolved_type == "image":
        if path_obj.is_file():
            bundle = pipeline.extract_image(image_path=str(path_obj), source_name=str(path_obj))
        else:
            bundle = pipeline.extract_image_library(image_dir=str(path_obj))
    else:
        if path_obj.is_file():
            bundle = pipeline.extract_video(video_path=str(path_obj), sample_fps=sample_fps)
        else:
            bundle = pipeline.extract_video_library(video_dir=str(path_obj), sample_fps=sample_fps)

    output_paths = bundle.dump(output_dir=output_dir, prefix=prefix)
    mat = bundle.feature_matrix()
    feature_dim = int(mat.shape[1]) if mat.ndim == 2 and mat.shape[0] > 0 else 0

    return FaceIndexBuildResult(
        library_path=library_path,
        library_type=resolved_type,
        total_faces=len(bundle),
        feature_dim=feature_dim,
        output_paths=output_paths,
        bundle=bundle,
    )

