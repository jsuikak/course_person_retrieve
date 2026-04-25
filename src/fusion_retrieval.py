"""Dual-branch face/person feature extraction and late fusion retrieval."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .tools.feature_extractor import FeatureExtractor, FeatureExtractorConfig, FeatureMode


@dataclass(slots=True)
class FusionFeatureConfig:
    """Configuration for ArcFace + OSNet feature fusion."""

    arcface_weight_path: str = "./models/weights/arcface.pt"
    device: str = "cpu"
    face_weight: float = 0.35
    person_weight: float = 0.65
    face_dim: int = 512
    face_flip_test: bool = True
    face_blur_threshold: float = 0.0
    face_min_size: int = 20
    person_model: str = "osnet"
    person_input_size: int = 224


@dataclass(slots=True)
class FusionFeatureRecord:
    """One image and its dual-branch features."""

    path: str
    identity: str
    split: str
    face_feature: np.ndarray | None
    person_feature: np.ndarray | None

    @property
    def face_missing(self) -> bool:
        return self.face_feature is None

    @property
    def person_missing(self) -> bool:
        return self.person_feature is None


@dataclass(slots=True)
class FusionFeatureMatrices:
    """Dense matrices used by retrieval and metrics."""

    paths: list[str]
    identities: list[str]
    splits: list[str]
    face_features: np.ndarray
    person_features: np.ndarray
    fused_features: np.ndarray
    face_valid: np.ndarray
    person_valid: np.ndarray
    fused_valid: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.astype(np.float32, copy=False)
    return (vec / norm).astype(np.float32, copy=False)


def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each matrix row; zero rows remain zero."""
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("mat must be 2-D.")
    if arr.shape[0] == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True).astype(np.float32)
    norms = np.maximum(norms, eps)
    return (arr / norms).astype(np.float32, copy=False)


def build_feature_extractor(config: FusionFeatureConfig) -> FeatureExtractor:
    """Create the shared extractor used by both branches."""
    return FeatureExtractor(
        FeatureExtractorConfig(
            arcface_weight_path=config.arcface_weight_path,
            device=config.device,
            detect_face=True,
            face_flip_test=config.face_flip_test,
            face_blur_threshold=config.face_blur_threshold,
            face_min_size=config.face_min_size,
            person_model=config.person_model,
            person_input_size=config.person_input_size,
        )
    )


def read_image_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        raise FileNotFoundError(f"failed to read image: {path}")
    return image


def extract_dual_feature(
    image_path: str | Path,
    identity: str,
    split: str,
    extractor: FeatureExtractor,
) -> FusionFeatureRecord:
    """Extract ArcFace face feature and OSNet/person feature for one cropped person image."""
    image = read_image_bgr(image_path)
    face_feature = extractor.extract(FeatureMode.FACE, image)
    person_feature = extractor.extract(FeatureMode.PERSON, image)
    return FusionFeatureRecord(
        path=str(image_path),
        identity=str(identity),
        split=str(split),
        face_feature=face_feature,
        person_feature=person_feature,
    )


def extract_dual_feature_records(
    items: Iterable[tuple[str | Path, str, str]],
    config: FusionFeatureConfig,
) -> list[FusionFeatureRecord]:
    """Extract features for many images with one shared model instance."""
    extractor = build_feature_extractor(config)
    records: list[FusionFeatureRecord] = []
    for image_path, identity, split in items:
        records.append(
            extract_dual_feature(
                image_path=image_path,
                identity=identity,
                split=split,
                extractor=extractor,
            )
        )
    return records


def _infer_feature_dim(records: Sequence[FusionFeatureRecord], attr: str, fallback: int | None = None) -> int:
    for record in records:
        feat = getattr(record, attr)
        if feat is not None:
            return int(np.asarray(feat).shape[0])
    if fallback is None:
        raise ValueError(f"cannot infer feature dim for {attr}; all features are missing")
    return int(fallback)


def build_fused_features(
    face_features: np.ndarray,
    person_features: np.ndarray,
    face_weight: float = 0.35,
    person_weight: float = 0.65,
) -> np.ndarray:
    """Create weighted concatenation fusion features.

    Inputs should already contain zero rows for missing branches. The output is
    L2-normalized after concatenation.
    """
    if face_features.ndim != 2 or person_features.ndim != 2:
        raise ValueError("face_features and person_features must be 2-D.")
    if face_features.shape[0] != person_features.shape[0]:
        raise ValueError("face/person row counts must match.")
    if face_weight < 0 or person_weight < 0:
        raise ValueError("feature weights must be non-negative.")
    if face_weight == 0 and person_weight == 0:
        raise ValueError("at least one feature weight must be positive.")

    face_part = np.sqrt(float(face_weight)) * face_features.astype(np.float32, copy=False)
    person_part = np.sqrt(float(person_weight)) * person_features.astype(np.float32, copy=False)
    return l2_normalize_rows(np.hstack([face_part, person_part]).astype(np.float32, copy=False))


def records_to_matrices(
    records: Sequence[FusionFeatureRecord],
    config: FusionFeatureConfig,
) -> FusionFeatureMatrices:
    """Convert sparse optional branch features to dense matrices and masks."""
    paths = [record.path for record in records]
    identities = [record.identity for record in records]
    splits = [record.split for record in records]
    face_dim = _infer_feature_dim(records, "face_feature", fallback=config.face_dim)
    person_dim = _infer_feature_dim(records, "person_feature", fallback=None)

    face_rows: list[np.ndarray] = []
    person_rows: list[np.ndarray] = []
    face_valid: list[bool] = []
    person_valid: list[bool] = []
    for record in records:
        if record.face_feature is None:
            face_rows.append(np.zeros((face_dim,), dtype=np.float32))
            face_valid.append(False)
        else:
            face_rows.append(_l2_normalize(np.asarray(record.face_feature, dtype=np.float32)))
            face_valid.append(True)

        if record.person_feature is None:
            person_rows.append(np.zeros((person_dim,), dtype=np.float32))
            person_valid.append(False)
        else:
            person_rows.append(_l2_normalize(np.asarray(record.person_feature, dtype=np.float32)))
            person_valid.append(True)

    face_features = np.vstack(face_rows).astype(np.float32) if face_rows else np.empty((0, face_dim), dtype=np.float32)
    person_features = (
        np.vstack(person_rows).astype(np.float32) if person_rows else np.empty((0, person_dim), dtype=np.float32)
    )
    face_mask = np.asarray(face_valid, dtype=bool)
    person_mask = np.asarray(person_valid, dtype=bool)
    fused_features = build_fused_features(
        face_features=face_features,
        person_features=person_features,
        face_weight=config.face_weight,
        person_weight=config.person_weight,
    )
    fused_mask = np.logical_or(face_mask, person_mask)

    return FusionFeatureMatrices(
        paths=paths,
        identities=identities,
        splits=splits,
        face_features=face_features,
        person_features=person_features,
        fused_features=fused_features,
        face_valid=face_mask,
        person_valid=person_mask,
        fused_valid=fused_mask,
    )


def compute_similarity_matrix(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix for query/gallery features."""
    if query_features.ndim != 2 or gallery_features.ndim != 2:
        raise ValueError("query_features and gallery_features must be 2-D.")
    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError(
            f"feature dim mismatch: query={query_features.shape[1]}, gallery={gallery_features.shape[1]}"
        )
    return l2_normalize_rows(query_features) @ l2_normalize_rows(gallery_features).T


def save_feature_matrices(matrices: FusionFeatureMatrices, output_dir: str | Path, prefix: str) -> dict[str, str]:
    """Save matrices and metadata for a split."""
    out_dir = Path(output_dir)
    feature_dir = out_dir / "features"
    metadata_dir = out_dir / "metadata"
    feature_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "face_features": feature_dir / f"{prefix}_face.npy",
        "person_features": feature_dir / f"{prefix}_person.npy",
        "fused_features": feature_dir / f"{prefix}_fused.npy",
        "face_valid": feature_dir / f"{prefix}_face_valid.npy",
        "person_valid": feature_dir / f"{prefix}_person_valid.npy",
        "fused_valid": feature_dir / f"{prefix}_fused_valid.npy",
        "metadata": metadata_dir / f"{prefix}.csv",
    }
    np.save(paths["face_features"], matrices.face_features)
    np.save(paths["person_features"], matrices.person_features)
    np.save(paths["fused_features"], matrices.fused_features)
    np.save(paths["face_valid"], matrices.face_valid)
    np.save(paths["person_valid"], matrices.person_valid)
    np.save(paths["fused_valid"], matrices.fused_valid)

    with paths["metadata"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "split", "identity", "path", "face_missing", "person_missing"])
        for row_id, path in enumerate(matrices.paths):
            writer.writerow(
                [
                    row_id,
                    matrices.splits[row_id],
                    matrices.identities[row_id],
                    path,
                    int(not bool(matrices.face_valid[row_id])),
                    int(not bool(matrices.person_valid[row_id])),
                ]
            )

    return {key: str(path) for key, path in paths.items()}


def load_feature_matrices(output_dir: str | Path, prefix: str) -> FusionFeatureMatrices:
    """Load a previously saved split cache."""
    out_dir = Path(output_dir)
    feature_dir = out_dir / "features"
    metadata_path = out_dir / "metadata" / f"{prefix}.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata cache not found: {metadata_path}")

    paths: list[str] = []
    identities: list[str] = []
    splits: list[str] = []
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            splits.append(str(row["split"]))
            identities.append(str(row["identity"]))
            paths.append(str(row["path"]))

    return FusionFeatureMatrices(
        paths=paths,
        identities=identities,
        splits=splits,
        face_features=np.load(feature_dir / f"{prefix}_face.npy").astype(np.float32, copy=False),
        person_features=np.load(feature_dir / f"{prefix}_person.npy").astype(np.float32, copy=False),
        fused_features=np.load(feature_dir / f"{prefix}_fused.npy").astype(np.float32, copy=False),
        face_valid=np.load(feature_dir / f"{prefix}_face_valid.npy").astype(bool, copy=False),
        person_valid=np.load(feature_dir / f"{prefix}_person_valid.npy").astype(bool, copy=False),
        fused_valid=np.load(feature_dir / f"{prefix}_fused_valid.npy").astype(bool, copy=False),
    )


__all__ = [
    "FusionFeatureConfig",
    "FusionFeatureMatrices",
    "FusionFeatureRecord",
    "build_feature_extractor",
    "build_fused_features",
    "compute_similarity_matrix",
    "extract_dual_feature",
    "extract_dual_feature_records",
    "l2_normalize_rows",
    "load_feature_matrices",
    "read_image_bgr",
    "records_to_matrices",
    "save_feature_matrices",
]
