"""人脸特征提取封装接口（MTCNN + ArcFace）。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .tools.feature_extractor import FeatureExtractor, FeatureExtractorConfig, FeatureMode
from .tools.mtcnn_detector import FaceBox, MTCNNDetector, MTCNNDetectorConfig


@dataclass(slots=True)
class FaceFeaturePipelineConfig:
    """人脸特征提取流程配置。"""

    arcface_weight_path: str
    device: str = "cpu"
    face_flip_test: bool = True
    face_blur_threshold: float = 0.0
    face_min_size: int = 0
    mtcnn_min_face_size: float = 20.0
    mtcnn_thresholds: tuple[float, float, float] = (0.6, 0.7, 0.8)
    mtcnn_nms_thresholds: tuple[float, float, float] = (0.7, 0.7, 0.7)


@dataclass(slots=True)
class FaceFeatureRecord:
    """单个人脸特征记录。"""

    source_type: str
    source_name: str
    frame_index: int
    face_index: int
    bbox: FaceBox
    feature: np.ndarray


@dataclass(slots=True)
class FaceFeatureBundle:
    """批量人脸特征结果。"""

    records: List[FaceFeatureRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def feature_matrix(self) -> np.ndarray:
        if not self.records:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([record.feature for record in self.records]).astype(np.float32)

    def dump(self, output_dir: str, prefix: str = "face_index") -> dict[str, str]:
        """落盘保存特征矩阵与元数据。"""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        features_path = out_dir / f"{prefix}_features.npy"
        meta_path = out_dir / f"{prefix}_meta.csv"
        info_path = out_dir / f"{prefix}_info.json"

        features = self.feature_matrix()
        np.save(features_path, features)

        with meta_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_id", "source_type", "source_name", "frame_index", "face_index", "x", "y", "w", "h"])
            for row_id, record in enumerate(self.records):
                writer.writerow(
                    [
                        row_id,
                        record.source_type,
                        record.source_name,
                        record.frame_index,
                        record.face_index,
                        record.bbox.x,
                        record.bbox.y,
                        record.bbox.w,
                        record.bbox.h,
                    ]
                )

        info = {
            "total_faces": len(self.records),
            "feature_dim": int(features.shape[1]) if features.ndim == 2 and features.shape[0] > 0 else 0,
            "features_path": str(features_path),
            "meta_path": str(meta_path),
        }
        with info_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        return {
            "features_path": str(features_path),
            "meta_path": str(meta_path),
            "info_path": str(info_path),
        }


class FaceFeaturePipeline:
    """人脸特征提取流程。

    流程：读图 -> MTCNN 检测 -> 人脸裁剪 -> ArcFace 特征提取。
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, config: FaceFeaturePipelineConfig):
        self.config = config
        self.detector = MTCNNDetector(
            MTCNNDetectorConfig(
                min_face_size=config.mtcnn_min_face_size,
                thresholds=config.mtcnn_thresholds,
                nms_thresholds=config.mtcnn_nms_thresholds,
            )
        )
        self.extractor = FeatureExtractor(
            FeatureExtractorConfig(
                arcface_weight_path=config.arcface_weight_path,
                device=config.device,
                detect_face=False,
                face_flip_test=config.face_flip_test,
                face_blur_threshold=config.face_blur_threshold,
                face_min_size=config.face_min_size,
            )
        )

    @staticmethod
    def _crop(image_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        x1 = max(0, box.x)
        y1 = max(0, box.y)
        x2 = min(w, box.x + box.w)
        y2 = min(h, box.y + box.h)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=image_bgr.dtype)
        return image_bgr[y1:y2, x1:x2]

    def _extract_from_frame(self, image_bgr: np.ndarray, source_type: str, source_name: str, frame_index: int) -> List[FaceFeatureRecord]:
        if image_bgr is None or image_bgr.size == 0:
            return []

        records: List[FaceFeatureRecord] = []
        boxes = self.detector.detect(image_bgr)
        for face_index, box in enumerate(boxes):
            crop = self._crop(image_bgr, box)
            if crop.size == 0:
                continue
            feat = self.extractor.extract(FeatureMode.FACE, crop)
            if feat is None:
                continue
            records.append(
                FaceFeatureRecord(
                    source_type=source_type,
                    source_name=source_name,
                    frame_index=frame_index,
                    face_index=face_index,
                    bbox=box,
                    feature=feat,
                )
            )
        return records

    @staticmethod
    def _read_image(image_path: str) -> Optional[np.ndarray]:
        image = cv2.imread(image_path)
        return image

    @classmethod
    def _list_files_by_ext(cls, root_dir: str, exts: set[str]) -> List[str]:
        root = Path(root_dir)
        if not root.exists():
            return []
        files = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        files.sort()
        return files

    @staticmethod
    def _frame_step(native_fps: float, sample_fps: float) -> int:
        if sample_fps <= 0:
            return 1
        if native_fps <= 0:
            return 1
        return max(1, int(round(native_fps / sample_fps)))

    def extract_image(
        self,
        image_bgr: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> FaceFeatureBundle:
        """单图提取（内存/磁盘兼容）。"""
        if (image_bgr is None) == (image_path is None):
            raise ValueError("Exactly one of image_bgr or image_path must be provided.")

        if image_bgr is None:
            image_bgr = self._read_image(image_path or "")
        if image_bgr is None or image_bgr.size == 0:
            return FaceFeatureBundle()

        src_name = source_name or image_path or "memory_image"
        records = self._extract_from_frame(image_bgr=image_bgr, source_type="image", source_name=src_name, frame_index=-1)
        return FaceFeatureBundle(records=records)

    def extract_images(self, images_bgr: Sequence[np.ndarray], source_prefix: str = "memory_image") -> FaceFeatureBundle:
        """内存图片批量提取。"""
        records: List[FaceFeatureRecord] = []
        for idx, image_bgr in enumerate(images_bgr):
            records.extend(
                self._extract_from_frame(
                    image_bgr=image_bgr,
                    source_type="image",
                    source_name=f"{source_prefix}_{idx}",
                    frame_index=-1,
                )
            )
        return FaceFeatureBundle(records=records)

    def extract_image_library(self, image_paths: Optional[Sequence[str]] = None, image_dir: Optional[str] = None) -> FaceFeatureBundle:
        """图库提取（磁盘路径列表或目录）。"""
        if image_paths is None and image_dir is None:
            raise ValueError("Provide image_paths or image_dir.")
        paths = [str(p) for p in image_paths] if image_paths is not None else self._list_files_by_ext(image_dir or "", self.IMAGE_EXTS)

        records: List[FaceFeatureRecord] = []
        for path in paths:
            records.extend(self.extract_image(image_path=path, source_name=path).records)
        return FaceFeatureBundle(records=records)

    def extract_video(self, video_path: str, sample_fps: float = 1.0) -> FaceFeatureBundle:
        """单视频提取（磁盘读取）。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return FaceFeatureBundle()

        native_fps = float(cap.get(cv2.CAP_PROP_FPS))
        step = self._frame_step(native_fps=native_fps, sample_fps=sample_fps)

        records: List[FaceFeatureRecord] = []
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                records.extend(
                    self._extract_from_frame(
                        image_bgr=frame,
                        source_type="video",
                        source_name=video_path,
                        frame_index=frame_index,
                    )
                )
            frame_index += 1

        cap.release()
        return FaceFeatureBundle(records=records)

    def extract_video_frames(self, frames_bgr: Sequence[np.ndarray], source_name: str = "memory_video") -> FaceFeatureBundle:
        """内存视频帧序列提取。"""
        records: List[FaceFeatureRecord] = []
        for frame_index, frame in enumerate(frames_bgr):
            records.extend(
                self._extract_from_frame(
                    image_bgr=frame,
                    source_type="video",
                    source_name=source_name,
                    frame_index=frame_index,
                )
            )
        return FaceFeatureBundle(records=records)

    def extract_video_library(self, video_paths: Optional[Sequence[str]] = None, video_dir: Optional[str] = None, sample_fps: float = 1.0) -> FaceFeatureBundle:
        """视频库提取（磁盘路径列表或目录）。"""
        if video_paths is None and video_dir is None:
            raise ValueError("Provide video_paths or video_dir.")
        paths = [str(p) for p in video_paths] if video_paths is not None else self._list_files_by_ext(video_dir or "", self.VIDEO_EXTS)

        records: List[FaceFeatureRecord] = []
        for path in paths:
            records.extend(self.extract_video(video_path=path, sample_fps=sample_fps).records)
        return FaceFeatureBundle(records=records)
