"""统一特征提取接口（ArcFace + ResNet，内存输入版）。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

from models import ArcFace, MTCNNFaceDetector


class FeatureMode(str, Enum):
    """特征提取类型。"""

    FACE = "face"
    PERSON = "person"


@dataclass(slots=True)
class FeatureExtractorConfig:
    """统一提取器配置。"""

    arcface_weight_path: str
    device: str = "cpu"
    detect_face: bool = False
    face_flip_test: bool = True
    face_blur_threshold: float = 0.0
    face_min_size: int = 0

    resnet_backbone: str = "resnet50"
    resnet_pretrained: bool = False
    resnet_weight_path: Optional[str] = None
    person_input_size: int = 224


class FeatureExtractor:
    """统一提特征入口（输入只接受内存 BGR 图像）。"""

    def __init__(self, config: FeatureExtractorConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.arcface = ArcFace(weight_path=config.arcface_weight_path, device=config.device)
        self.face_detector = (
            MTCNNFaceDetector(min_face_size=max(float(config.face_min_size), 20.0)) if config.detect_face else None
        )
        self.person_model: Optional[torch.nn.Module] = None
        self.person_dim: int = 0

    @staticmethod
    def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < eps:
            return vec.astype(np.float32, copy=False)
        return (vec / norm).astype(np.float32, copy=False)

    @staticmethod
    def _largest_bbox(raw_boxes: Sequence[object]) -> Optional[tuple[int, int, int, int]]:
        if not raw_boxes:
            return None
        normalized: List[tuple[int, int, int, int]] = []
        for box in raw_boxes:
            if hasattr(box, "x") and hasattr(box, "y") and hasattr(box, "w") and hasattr(box, "h"):
                normalized.append((int(box.x), int(box.y), int(box.w), int(box.h)))
            else:
                x, y, w, h = box
                normalized.append((int(x), int(y), int(w), int(h)))
        return max(normalized, key=lambda b: b[2] * b[3])

    @staticmethod
    def _crop_with_bbox(image_bgr: np.ndarray, bbox: Optional[tuple[int, int, int, int]]) -> np.ndarray:
        if bbox is None:
            return image_bgr
        x, y, w, h = bbox
        height, width = image_bgr.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=image_bgr.dtype)
        return image_bgr[y1:y2, x1:x2]

    def _passes_face_quality(self, crop: np.ndarray, bbox: Optional[tuple[int, int, int, int]]) -> bool:
        if crop is None or crop.size == 0:
            return False

        min_size = int(self.config.face_min_size)
        if min_size > 0:
            if bbox is not None:
                _, _, bw, bh = bbox
                if bw < min_size or bh < min_size:
                    return False
            else:
                h, w = crop.shape[:2]
                if w < min_size or h < min_size:
                    return False

        blur_threshold = float(self.config.face_blur_threshold)
        if blur_threshold > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < blur_threshold:
                return False
        return True

    def _preprocess_face(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (112, 112)).astype(np.float32)
        image_rgb = (image_rgb - 127.5) / 128.0
        chw = np.transpose(image_rgb, (2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _face_forward(self, image_bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess_face(image_bgr)
        feat = self.arcface.model(x)[0].detach().cpu().numpy().astype(np.float32)
        return self._l2_normalize(feat)

    def _extract_face_from_bgr(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        if image_bgr is None or image_bgr.size == 0:
            return None

        bbox: Optional[tuple[int, int, int, int]] = None
        if self.face_detector is not None:
            boxes = self.face_detector.detect(image_bgr)
            bbox = self._largest_bbox(boxes)
            if bbox is None:
                return None

        crop = self._crop_with_bbox(image_bgr, bbox)
        if not self._passes_face_quality(crop, bbox):
            return None

        feat = self._face_forward(crop)
        if not self.config.face_flip_test:
            return feat

        flipped = cv2.flip(crop, 1)
        feat_flip = self._face_forward(flipped)
        return self._l2_normalize((feat + feat_flip) / 2.0)

    def _ensure_person_model(self) -> None:
        if self.person_model is not None:
            return
        self.person_model, self.person_dim = self._build_person_model()

    def _build_person_model(self) -> Tuple[torch.nn.Module, int]:
        backbone = self.config.resnet_backbone.lower().strip()
        if backbone == "resnet18":
            try:
                weights = ResNet18_Weights.DEFAULT if self.config.resnet_pretrained else None
                model = torchvision.models.resnet18(weights=weights)
            except Exception:
                model = torchvision.models.resnet18(weights=None)
            feature_dim = 512
        elif backbone == "resnet34":
            try:
                weights = ResNet34_Weights.DEFAULT if self.config.resnet_pretrained else None
                model = torchvision.models.resnet34(weights=weights)
            except Exception:
                model = torchvision.models.resnet34(weights=None)
            feature_dim = 512
        elif backbone == "resnet50":
            try:
                weights = ResNet50_Weights.DEFAULT if self.config.resnet_pretrained else None
                model = torchvision.models.resnet50(weights=weights)
            except Exception:
                model = torchvision.models.resnet50(weights=None)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported resnet_backbone: {self.config.resnet_backbone}")

        model.fc = torch.nn.Identity()
        if self.config.resnet_weight_path:
            ckpt = torch.load(self.config.resnet_weight_path, map_location=self.device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
            cleaned = {}
            for key, value in state_dict.items():
                cleaned[key[7:] if key.startswith("module.") else key] = value
            model.load_state_dict(cleaned, strict=False)

        model.eval()
        model.to(self.device)
        return model, feature_dim

    def _preprocess_person(self, image_bgr: np.ndarray) -> torch.Tensor:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("empty person image")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.config.person_input_size, self.config.person_input_size))
        image_rgb = image_rgb.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_rgb = (image_rgb - mean) / std

        chw = np.transpose(image_rgb, (2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _extract_person_from_bgr(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        if image_bgr is None or image_bgr.size == 0:
            return None
        try:
            x = self._preprocess_person(image_bgr)
        except ValueError:
            return None

        self._ensure_person_model()
        if self.person_model is None:
            return None
        feat = self.person_model(x)[0].detach().cpu().numpy().astype(np.float32)
        return self._l2_normalize(feat)

    def extract(self, mode: FeatureMode, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """单条提特征。失败返回 None。"""
        if image_bgr is None or image_bgr.size == 0:
            return None

        if mode == FeatureMode.FACE:
            return self._extract_face_from_bgr(image_bgr)
        if mode == FeatureMode.PERSON:
            return self._extract_person_from_bgr(image_bgr)
        raise ValueError(f"Unsupported mode: {mode}")

    def extract_batch(self, mode: FeatureMode, images_bgr: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
        """批量提特征，返回与输入等长的特征列表（失败位置为 None）。"""
        return [self.extract(mode=mode, image_bgr=image_bgr) for image_bgr in images_bgr]

    def extract_batch_matrix(self, mode: FeatureMode, images_bgr: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        """批量提特征并压缩成矩阵，返回 (features, valid_indices)。"""
        feats = self.extract_batch(mode=mode, images_bgr=images_bgr)
        valid_indices: List[int] = [idx for idx, feat in enumerate(feats) if feat is not None]
        valid_feats = [feats[idx] for idx in valid_indices]
        if not valid_feats:
            return np.empty((0, 0), dtype=np.float32), valid_indices
        return np.vstack(valid_feats).astype(np.float32), valid_indices
