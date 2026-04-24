"""人体特征提取封装接口（YOLO Person + person backbone 特征）。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .tools.feature_extractor import FeatureExtractor, FeatureExtractorConfig, FeatureMode
from .tools.yolo_person_detector import PersonBox, YOLOPersonDetector, YOLOPersonDetectorConfig


@dataclass(slots=True)
class PersonFeaturePipelineConfig:
    """人体特征提取流程配置。"""

    arcface_weight_path: str
    device: str = "cpu"
    person_model: str = "resnet"
    resnet_backbone: str = "resnet18"
    resnet_pretrained: bool = False
    resnet_weight_path: Optional[str] = None
    person_input_size: int = 224

    yolo_weights: str = "./models/weights/yolo11n.pt"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    yolo_max_det: int = 100


@dataclass(slots=True)
class PersonFeatureRecord:
    """单个人体特征记录。"""

    source_type: str
    source_name: str
    frame_index: int
    person_index: int
    bbox: PersonBox
    feature: np.ndarray


@dataclass(slots=True)
class PersonFeatureBundle:
    """批量人体特征结果。"""

    records: List[PersonFeatureRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def feature_matrix(self) -> np.ndarray:
        if not self.records:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([record.feature for record in self.records]).astype(np.float32)

    def dump(self, output_dir: str, prefix: str = "person_index") -> dict[str, str]:
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
            writer.writerow(
                [
                    "row_id",
                    "source_type",
                    "source_name",
                    "frame_index",
                    "person_index",
                    "x",
                    "y",
                    "w",
                    "h",
                    "conf",
                    "cls",
                ]
            )
            for row_id, record in enumerate(self.records):
                writer.writerow(
                    [
                        row_id,
                        record.source_type,
                        record.source_name,
                        record.frame_index,
                        record.person_index,
                        record.bbox.x,
                        record.bbox.y,
                        record.bbox.w,
                        record.bbox.h,
                        float(record.bbox.conf),
                        int(record.bbox.cls),
                    ]
                )

        info = {
            "total_persons": len(self.records),
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


class PersonFeaturePipeline:
    """人体特征提取流程。

    流程：读图 -> YOLO 人体检测 -> 人体裁剪 -> person backbone 特征提取。
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, config: PersonFeaturePipelineConfig):
        self.config = config
        self.detector = YOLOPersonDetector(
            YOLOPersonDetectorConfig(
                weights=config.yolo_weights,
                conf=config.yolo_conf,
                iou=config.yolo_iou,
                max_det=config.yolo_max_det,
                device=config.device,
            )
        )
        self.extractor = FeatureExtractor(
            FeatureExtractorConfig(
                arcface_weight_path=config.arcface_weight_path,
                device=config.device,
                detect_face=False,
                face_flip_test=False,
                person_model=config.person_model,
                resnet_backbone=config.resnet_backbone,
                resnet_pretrained=config.resnet_pretrained,
                resnet_weight_path=config.resnet_weight_path,
                person_input_size=config.person_input_size,
            )
        )

    @staticmethod
    def _crop(image_bgr: np.ndarray, box: PersonBox) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        x1 = max(0, box.x)
        y1 = max(0, box.y)
        x2 = min(w, box.x + box.w)
        y2 = min(h, box.y + box.h)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=image_bgr.dtype)
        return image_bgr[y1:y2, x1:x2]

    def _extract_from_frame(self, image_bgr: np.ndarray, source_type: str, source_name: str, frame_index: int) -> List[PersonFeatureRecord]:
        if image_bgr is None or image_bgr.size == 0:
            return []

        records: List[PersonFeatureRecord] = []
        boxes = self.detector.detect(image_bgr)
        for person_index, box in enumerate(boxes):
            crop = self._crop(image_bgr, box)
            if crop.size == 0:
                continue
            feat = self.extractor.extract(FeatureMode.PERSON, crop)
            if feat is None:
                continue
            records.append(
                PersonFeatureRecord(
                    source_type=source_type,
                    source_name=source_name,
                    frame_index=frame_index,
                    person_index=person_index,
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
    ) -> PersonFeatureBundle:
        """单图提取（内存/磁盘兼容）。"""
        if (image_bgr is None) == (image_path is None):
            raise ValueError("Exactly one of image_bgr or image_path must be provided.")

        if image_bgr is None:
            image_bgr = self._read_image(image_path or "")
        if image_bgr is None or image_bgr.size == 0:
            return PersonFeatureBundle()

        src_name = source_name or image_path or "memory_image"
        records = self._extract_from_frame(image_bgr=image_bgr, source_type="image", source_name=src_name, frame_index=-1)
        return PersonFeatureBundle(records=records)

    def extract_images(self, images_bgr: Sequence[np.ndarray], source_prefix: str = "memory_image") -> PersonFeatureBundle:
        """内存图片批量提取。"""
        records: List[PersonFeatureRecord] = []
        for idx, image_bgr in enumerate(images_bgr):
            records.extend(
                self._extract_from_frame(
                    image_bgr=image_bgr,
                    source_type="image",
                    source_name=f"{source_prefix}_{idx}",
                    frame_index=-1,
                )
            )
        return PersonFeatureBundle(records=records)

    def extract_image_library(self, image_paths: Optional[Sequence[str]] = None, image_dir: Optional[str] = None) -> PersonFeatureBundle:
        """图库提取（磁盘路径列表或目录）。"""
        if image_paths is None and image_dir is None:
            raise ValueError("Provide image_paths or image_dir.")
        paths = [str(p) for p in image_paths] if image_paths is not None else self._list_files_by_ext(image_dir or "", self.IMAGE_EXTS)

        records: List[PersonFeatureRecord] = []
        for path in paths:
            records.extend(self.extract_image(image_path=path, source_name=path).records)
        return PersonFeatureBundle(records=records)

    def extract_video(self, video_path: str, sample_fps: float = 1.0) -> PersonFeatureBundle:
        """单视频提取（磁盘读取）。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return PersonFeatureBundle()

        native_fps = float(cap.get(cv2.CAP_PROP_FPS))
        step = self._frame_step(native_fps=native_fps, sample_fps=sample_fps)

        records: List[PersonFeatureRecord] = []
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
        return PersonFeatureBundle(records=records)

    def extract_video_frames(self, frames_bgr: Sequence[np.ndarray], source_name: str = "memory_video") -> PersonFeatureBundle:
        """内存视频帧序列提取。"""
        records: List[PersonFeatureRecord] = []
        for frame_index, frame in enumerate(frames_bgr):
            records.extend(
                self._extract_from_frame(
                    image_bgr=frame,
                    source_type="video",
                    source_name=source_name,
                    frame_index=frame_index,
                )
            )
        return PersonFeatureBundle(records=records)

    def extract_video_library(self, video_paths: Optional[Sequence[str]] = None, video_dir: Optional[str] = None, sample_fps: float = 1.0) -> PersonFeatureBundle:
        """视频库提取（磁盘路径列表或目录）。"""
        if video_paths is None and video_dir is None:
            raise ValueError("Provide video_paths or video_dir.")
        paths = [str(p) for p in video_paths] if video_paths is not None else self._list_files_by_ext(video_dir or "", self.VIDEO_EXTS)

        records: List[PersonFeatureRecord] = []
        for path in paths:
            records.extend(self.extract_video(video_path=path, sample_fps=sample_fps).records)
        return PersonFeatureBundle(records=records)
